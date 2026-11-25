"""Per-attempt provider endpoint driven by Cookbook's token sampler.

The gateway never calls a model itself. Every generation request becomes one
RL observation, and its response is supplied by ``Env.step``. This preserves
the sampled tokens and log probabilities, including auxiliary condenser calls.
"""

from __future__ import annotations

import asyncio
import json
import secrets
from dataclasses import dataclass

from aiohttp import web

from elt_environment.protocol import Request, parse_request, stream_events


@dataclass
class Pending:
    request: Request
    observation: object
    reply: asyncio.Future


class Bridge:
    def __init__(self, renderer, *, bind_host: str = "127.0.0.1"):
        self.renderer = renderer
        self.bind_host = bind_host
        self.token = secrets.token_urlsafe(32)
        self.pending: asyncio.Queue[Pending] = asyncio.Queue()
        self.requests: set[asyncio.Future] = set()
        self.failure: asyncio.Future | None = None
        self.runner = None
        self.port = 0

    async def start(self):
        self.failure = asyncio.get_running_loop().create_future()
        app = web.Application(client_max_size=8 * 1024 * 1024)
        app.router.add_post("/v1/chat/completions", self.handle)
        app.router.add_post("/v1/responses", self.handle)
        app.router.add_post("/v1/messages", self.handle)
        app.router.add_post("/v1/messages/count_tokens", self.handle)
        self.runner = web.AppRunner(app, shutdown_timeout=1)
        await self.runner.setup()
        site = web.TCPSite(self.runner, self.bind_host, 0)
        await site.start()
        self.port = self.runner.addresses[0][1]

    def fail(self, reason: str):
        if self.failure is not None and not self.failure.done():
            self.failure.set_result(reason)

    async def handle(self, http: web.Request):
        bearer = http.headers.get("Authorization", "").removeprefix("Bearer ")
        key = http.headers.get("x-api-key", bearer)
        if not secrets.compare_digest(key, self.token):
            raise web.HTTPUnauthorized()
        protocol = (
            "chat"
            if "chat/completions" in http.path
            else "responses"
            if "responses" in http.path
            else "messages"
        )
        try:
            body = await http.json()
            request = parse_request(protocol, body)
            observation = request.render(self.renderer)
        except (ValueError, KeyError, TypeError, NotImplementedError):
            self.fail("provider_request_unsupported")
            return web.json_response(
                {"error": {"message": "unsupported provider request"}}, status=400
            )
        if http.path.endswith("count_tokens"):
            return web.json_response({"input_tokens": observation.length})
        future = asyncio.get_running_loop().create_future()
        self.requests.add(future)
        await self.pending.put(Pending(request, observation, future))
        try:
            reply = await future
            if not body.get("stream"):
                return web.json_response(reply)
            response = web.StreamResponse(
                headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"}
            )
            await response.prepare(http)
            for name, payload in stream_events(protocol, reply):
                data = payload if isinstance(payload, str) else json.dumps(payload)
                event = (f"event: {name}\n" if name else "") + f"data: {data}\n\n"
                await response.write(event.encode())
            await response.write_eof()
            return response
        except (ConnectionError, asyncio.CancelledError):
            self.fail("provider_connection_lost")
            raise
        finally:
            self.requests.discard(future)

    async def close(self):
        for future in list(self.requests):
            future.cancel()
        if self.runner is not None:
            await self.runner.cleanup()
            self.runner = None
