"""Translate agent provider messages to the selected Tinker renderer.

Only the text and tool interfaces used by the vendored agents are supported.
Unsupported inputs fail explicitly instead of disappearing from the context.
Native Responses shell/apply_patch calls are represented as renderer functions
and converted back to native calls before the original SDK executes them.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from typing import Any

from tinker_cookbook import renderers
from tinker_cookbook.renderers.base import ToolCall


def identifier(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def text_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        raise TypeError("expected text content")
    parts = []
    for part in value:
        if part.get("type") not in {"text", "input_text", "output_text"}:
            raise ValueError(f"unsupported content type: {part.get('type')}")
        parts.append(part["text"])
    return "\n".join(parts)


def call(name: str, arguments: Any, call_id: str | None = None) -> ToolCall:
    return ToolCall(
        id=call_id or identifier("call"),
        function=ToolCall.FunctionBody(
            name=name,
            arguments=arguments if isinstance(arguments, str) else json.dumps(arguments),
        ),
    )


NATIVE_TOOLS = {
    "shell": {
        "name": "shell",
        "description": "Execute commands using the agent's shell executor.",
        "parameters": {
            "type": "object",
            "properties": {
                "commands": {"type": "array", "items": {"type": "string"}},
                "timeout_ms": {"type": "integer"},
                "max_output_length": {"type": "integer"},
            },
            "required": ["commands"],
        },
    },
    "apply_patch": {
        "name": "apply_patch",
        "description": "Apply a V4A diff using the agent's file editor.",
        "parameters": {
            "type": "object",
            "properties": {
                "type": {"enum": ["create_file", "update_file", "delete_file"]},
                "path": {"type": "string"},
                "diff": {"type": "string"},
            },
            "required": ["type", "path"],
        },
    },
}


@dataclass
class Request:
    protocol: str
    body: dict
    messages: list
    tools: list
    native_tools: set[str]

    def render(self, renderer):
        messages = list(self.messages)
        if self.tools:
            # Merge only the initial system messages; preserve later message order.
            system = []
            while messages and messages[0]["role"] == "system":
                system.append(text_content(messages.pop(0)["content"]))
            messages = (
                renderer.create_conversation_prefix_with_tools(self.tools, "\n\n".join(system))
                + messages
            )
        return renderer.build_generation_prompt(messages)


def parse_request(protocol: str, body: dict) -> Request:
    if not isinstance(body, dict):
        raise TypeError("request body must be a JSON object")
    if body.get("n", 1) != 1:
        raise ValueError("one completion per request is required")
    if body.get("previous_response_id"):
        raise ValueError("send complete Responses input; server-side history is unsupported")
    if body.get("response_format", {}).get("type", "text") != "text":
        raise ValueError("structured response formats are unsupported")
    messages, specs, native = [], [], set()
    for tool in body.get("tools", []):
        kind = tool.get("type", "custom")
        if protocol == "responses" and kind in NATIVE_TOOLS:
            specs.append(NATIVE_TOOLS[kind])
            native.add(kind)
        elif protocol == "messages" and kind == "custom":
            specs.append(
                {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool["input_schema"],
                }
            )
        elif kind == "function":
            definition = tool.get("function", tool)
            specs.append(
                {
                    key: definition.get(key, {} if key == "parameters" else "")
                    for key in ("name", "description", "parameters")
                }
            )
        else:
            raise ValueError(f"unsupported tool type: {kind}")
    choice = body.get("tool_choice", "auto")
    if choice not in ("auto", None) and choice != {"type": "auto"}:
        # Forced selection needs constrained decoding; pretending to enforce it
        # would change the policy distribution recorded by the RL sampler.
        raise ValueError("only automatic tool selection is supported")

    if protocol == "chat":
        for item in body["messages"]:
            message = {"role": item["role"], "content": text_content(item.get("content"))}
            if message["role"] == "developer":
                message["role"] = "system"
            for key in ("tool_call_id", "name"):
                if key in item:
                    message[key] = item[key]
            if item.get("tool_calls"):
                message["tool_calls"] = [ToolCall.model_validate(tc) for tc in item["tool_calls"]]
            if item.get("reasoning_content"):
                message["content"] = [
                    {"type": "thinking", "thinking": item["reasoning_content"]},
                    {"type": "text", "text": message["content"]},
                ]
            messages.append(message)
    elif protocol == "messages":
        if body.get("system"):
            messages.append({"role": "system", "content": text_content(body["system"])})
        for item in body["messages"]:
            if isinstance(item["content"], str):
                messages.append({"role": item["role"], "content": item["content"]})
                continue
            message = {"role": item["role"], "content": [], "tool_calls": []}
            for block in item["content"]:
                kind = block["type"]
                if kind == "text":
                    message["content"].append({"type": "text", "text": block["text"]})
                elif kind == "thinking":
                    message["content"].append({"type": "thinking", "thinking": block["thinking"]})
                elif kind == "tool_use":
                    message["tool_calls"].append(call(block["name"], block["input"], block["id"]))
                elif kind == "tool_result":
                    if message["content"] or message["tool_calls"]:
                        messages.append(message)
                        message = {"role": item["role"], "content": [], "tool_calls": []}
                    result = text_content(block.get("content"))
                    if block.get("is_error"):
                        result = "Tool error: " + result
                    messages.append(
                        {"role": "tool", "content": result, "tool_call_id": block["tool_use_id"]}
                    )
                else:
                    raise ValueError(f"unsupported Anthropic block: {kind}")
            if message["content"] or message["tool_calls"]:
                messages.append(message)
    elif protocol == "responses":
        if body.get("instructions"):
            messages.append({"role": "system", "content": body["instructions"]})
        items = body.get("input", [])
        if isinstance(items, str):
            items = [{"role": "user", "content": items}]
        for item in items:
            kind = item.get("type", "message")
            if kind == "message":
                role = item["role"]
                messages.append(
                    {
                        "role": "system" if role == "developer" else role,
                        "content": text_content(item.get("content")),
                    }
                )
            elif kind in {"function_call", "shell_call", "apply_patch_call"}:
                name = item["name"] if kind == "function_call" else kind.removesuffix("_call")
                args = (
                    item.get("arguments")
                    if kind == "function_call"
                    else item.get("action" if kind == "shell_call" else "operation")
                )
                messages.append(
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [call(name, args, item["call_id"])],
                    }
                )
            elif kind in {"function_call_output", "shell_call_output", "apply_patch_call_output"}:
                output = item.get("output", "")
                if kind == "apply_patch_call_output":
                    output = {"status": item.get("status"), "output": output}
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": item["call_id"],
                        "content": output if isinstance(output, str) else json.dumps(output),
                    }
                )
            elif kind == "reasoning":
                # Our gateway emits no opaque/encrypted reasoning items.
                if item.get("encrypted_content"):
                    raise ValueError("encrypted reasoning cannot be rendered")
                messages.append(
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "thinking", "thinking": part["text"]}
                            for part in item.get("summary", [])
                        ],
                    }
                )
            else:
                raise ValueError(f"unsupported Responses input: {kind}")
    else:
        raise ValueError("unknown provider protocol")
    return Request(protocol, body, messages, specs, native)


def completion(request: Request, message: dict, input_tokens: int, output_tokens: int) -> dict:
    content = renderers.get_text_content(message)
    calls = [tc.model_dump() for tc in message.get("tool_calls", [])]
    for tc in calls:
        tc["id"] = tc["id"] or identifier("call")
    model = request.body.get("model", "tinker")
    if request.protocol == "chat":
        reply = {"role": "assistant", "content": content or None}
        if calls:
            reply["tool_calls"] = calls
        return {
            "id": identifier("chatcmpl"),
            "object": "chat.completion",
            "model": model,
            "created": int(time.time()),
            "choices": [
                {"index": 0, "message": reply, "finish_reason": "tool_calls" if calls else "stop"}
            ],
            "usage": {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
        }
    if request.protocol == "messages":
        blocks = [{"type": "text", "text": content}] if content else []
        blocks.extend(
            {
                "type": "tool_use",
                "id": tc["id"],
                "name": tc["function"]["name"],
                "input": json.loads(tc["function"]["arguments"]),
            }
            for tc in calls
        )
        return {
            "id": identifier("msg"),
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": blocks,
            "stop_reason": "tool_use" if calls else "end_turn",
            "stop_sequence": None,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
        }
    output = []
    if content:
        output.append(
            {
                "id": identifier("msg"),
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {"type": "output_text", "text": content, "annotations": [], "logprobs": []}
                ],
            }
        )
    for tc in calls:
        fn = tc["function"]
        item = {"id": identifier("fc"), "call_id": tc["id"], "status": "completed"}
        if fn["name"] in request.native_tools:
            item.update(type=f"{fn['name']}_call")
            item["action" if fn["name"] == "shell" else "operation"] = json.loads(fn["arguments"])
        else:
            item.update(type="function_call", name=fn["name"], arguments=fn["arguments"])
        output.append(item)
    return {
        "id": identifier("resp"),
        "object": "response",
        "created_at": int(time.time()),
        "model": model,
        "status": "completed",
        "output": output,
        "error": None,
        "incomplete_details": None,
        "parallel_tool_calls": True,
        "tools": [],
        "tool_choice": "auto",
        "temperature": 1.0,
        "top_p": 1.0,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }


def stream_events(protocol: str, reply: dict):
    """Yield provider SSE payloads; sampling is complete before this transport flush."""
    if protocol == "chat":
        choice = reply["choices"][0]
        delta = dict(choice["message"])
        if "tool_calls" in delta:
            delta["tool_calls"] = [dict(tc, index=i) for i, tc in enumerate(delta["tool_calls"])]
        base = {key: reply[key] for key in ("id", "created", "model")}
        base["object"] = "chat.completion.chunk"
        yield None, dict(base, choices=[{"index": 0, "delta": delta, "finish_reason": None}])
        yield (
            None,
            dict(
                base,
                choices=[{"index": 0, "delta": {}, "finish_reason": choice["finish_reason"]}],
                usage=reply["usage"],
            ),
        )
        yield None, "[DONE]"
    elif protocol == "messages":
        yield (
            "message_start",
            {
                "type": "message_start",
                "message": dict(
                    reply, content=[], stop_reason=None, usage=dict(reply["usage"], output_tokens=0)
                ),
            },
        )
        for index, block in enumerate(reply["content"]):
            is_text = block["type"] == "text"
            start = dict(block, text="") if is_text else dict(block, input={})
            yield (
                "content_block_start",
                {"type": "content_block_start", "index": index, "content_block": start},
            )
            delta = (
                {"type": "text_delta", "text": block["text"]}
                if is_text
                else {"type": "input_json_delta", "partial_json": json.dumps(block["input"])}
            )
            yield (
                "content_block_delta",
                {"type": "content_block_delta", "index": index, "delta": delta},
            )
            yield "content_block_stop", {"type": "content_block_stop", "index": index}
        yield (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": reply["stop_reason"], "stop_sequence": None},
                "usage": reply["usage"],
            },
        )
        yield "message_stop", {"type": "message_stop"}
    else:
        events = [
            ("response.created", {"response": dict(reply, status="in_progress", output=[])}),
            ("response.in_progress", {"response": dict(reply, status="in_progress", output=[])}),
        ]
        for index, item in enumerate(reply["output"]):
            start = dict(item, status="in_progress")
            if item["type"] == "message":
                start["content"] = []
            elif item["type"] == "function_call":
                start["arguments"] = ""
            events.append(("response.output_item.added", {"output_index": index, "item": start}))
            if item["type"] == "message":
                part = item["content"][0]
                meta = {"item_id": item["id"], "output_index": index, "content_index": 0}
                events.extend(
                    [
                        ("response.content_part.added", dict(meta, part=dict(part, text=""))),
                        ("response.output_text.delta", dict(meta, delta=part["text"], logprobs=[])),
                        ("response.output_text.done", dict(meta, text=part["text"], logprobs=[])),
                        ("response.content_part.done", dict(meta, part=part)),
                    ]
                )
            elif item["type"] == "function_call":
                meta = {"item_id": item["id"], "output_index": index}
                events.extend(
                    [
                        (
                            "response.function_call_arguments.delta",
                            dict(meta, delta=item["arguments"]),
                        ),
                        (
                            "response.function_call_arguments.done",
                            dict(meta, arguments=item["arguments"]),
                        ),
                    ]
                )
            events.append(("response.output_item.done", {"output_index": index, "item": item}))
        events.append(("response.completed", {"response": reply}))
        for sequence, (name, payload) in enumerate(events):
            yield name, dict(payload, type=name, sequence_number=sequence)
