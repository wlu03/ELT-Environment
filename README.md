# ELT-Environment

Run ELT-Bench agents with a model served by Tinker and train on multi-turn trajectories. The original agents keep their prompts, tools and execution loops the same. The fastest start to this is to use `agent=codex`. This select the type of framework to use. 

## Setup

1. Keep `ELT-taskgen` in the same root directory as `ELT-Environment`, then install the dependencies

```Python
cd ELT-Environment
uv sync
```

2. Create `.env` in the repo and add your Tinker API key and project ID
```markdown
TINKER_API_KEY=urapikey
TINKER_PROJECT_ID=urprojectid
```
3. Put your `training-data/<name>`. Keep each training data the same as `ELT-taskgen`. Visit that [repo](github.com/wlu03/ELT-taskgen). `public/` should be the agent facing files and `private/` keeps all of the verifiers and evaluators. 

4. Install the Codex runner's requirements in a separate Python env. Build its `elt-swe` Docker image using the instructions under `agents/`. Pass the environment's Python interpreter as `agent_python`. 

5. Start the source services, Airbyte, and configured warehouses. Use taskgen's runtime installer to install development inputs under `input_roots/<task_id>/` with runtime credentials. schemas and `data_model.yaml` must match the training-data. 

6. A real cloud grader is not included. 

Agent choices:

- `spider`: original Spider PromptAgent and Snowflake tools; image `elt_agent-image`.
- `swe`: original SWE-agent framework and `elt_fc.yaml`; image `elt-swe`.
- `codex`: original OpenAI Agents SDK runner; image `elt-swe`.
- `claude`: original Claude Agent SDK runner; image `elt-swe-claude`.
- `openhands`: original OpenHands runner, including its condenser; image `elt-swe-openhands`.

All agents except Spider support the original runner's Snowflake, Databricks
and Redshift destinations. See each folder's requirements and Docker build files.


```bash
.venv/bin/python -m elt_environment.probe \
  agent=codex agent_python=/absolute/path/to/agent-env/bin/python \
  inputs_root=/absolute/path/to/installed-development-tasks \
  grader_command='/absolute/path/to/hidden-replay-grader' \
  tasks=1 samples=1 eval_families=0

.venv/bin/python -m elt_environment.train \
  agent=codex agent_python=/absolute/path/to/agent-env/bin/python \
  inputs_root=/absolute/path/to/installed-development-tasks \
  grader_command='/absolute/path/to/hidden-replay-grader' \
  max_turns=100 max_steps=20
```

- `max_turns` bounds model calls per attempt, including condenser calls.
- `max_steps` bounds training updates. 
- `max_tokens` applies to each sampled reply.

Family holdouts remain separate from training. Probes save every sampled turn,
its token IDs, log probabilities and terminal reward.

Claude and OpenHands call the model from their containers. Set
`bridge_bind_host=0.0.0.0`; `bridge_container_host` defaults to
`host.docker.internal` for Docker Desktop. On Linux, supply a host address
reachable from the configured Docker network. Each bridge requires a random
per-attempt token. Real provider keys and the Tinker key are not passed to workers.

The original runners reset the configured warehouse namespace. Attempts sharing
an `inputs_root` are serialized using a host file lock, through grading and cleanup.
Use one installed input root per runtime; separate hosts need separate runtime
resources. Parallel warehouse provisioning is not implemented.

## Grading

The command receives one extra argument: a private JSON request filename:

```json
{"task_id":"...","release_dir":"/absolute/release","destination":"snowflake","files":{"elt/main.tf":"...","elt/models/mart.sql":"..."}}
```

It must replay the submitted source on fresh hidden populations and print one
JSON outcome to stdout, with diagnostics on stderr:

```json
{"label_valid":true,"reward":0.75,"metrics":{"r_el":1.0,"r_t":0.5}}
```

For an infrastructure or harness fault:

```json
{"label_valid":false,"reason_code":"warehouse_unavailable"}
```

Faults discard and resample the entire GRPO group; they never become zero rewards.
The grader request, release private tree and results stay outside the agent mount.
The agent sees only the installed task workspace and its development services.

For an explicit **offline diagnostic**, omit `grader_command` and set
`allow_declarative_grader=true`. This selects the existing taskgen DuckDB replay
scorer. It enforces a narrower Terraform/SQL contract than the copied agents:
valid real-cloud solutions can fail this scorer. It should not be used to claim
ELT-Bench-equivalent pass rates. There is no silent fallback to it.

## Checks and limits

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python testing/test_agent_folders.py
.venv/bin/python testing/test_tinker_integration.py
.venv/bin/python testing/test_tinker_renderer.py
```

`testing/` follows the local unittest script style and stays ignored by Git.
SDK protocol checks additionally use `.state/sdk-tests/bin/python` when installed
with `openai-agents` and `anthropic`; otherwise those cases report a skip.
These checks use scripted completions and do not call paid models or warehouses.
The renderer check uses the locally cached Qwen tokenizer with Hub access disabled.

The bridge supports text, function tools, Responses shell/apply-patch tools,
streaming and Anthropic token counting. Unsupported multimodal, hosted tools or
forced tool-choice requests fail explicitly. The wire formats follow
[OpenAI Responses streaming](https://developers.openai.com/api/reference/resources/responses/streaming-events)
and [Anthropic Messages streaming](https://platform.claude.com/docs/en/build-with-claude/streaming).
Live Tinker sampling, all five Docker runners and full cloud replay still require
an end-to-end canary with configured infrastructure.

```text
agents/                        unchanged upstream agents
training-data/                 frozen public/private releases (ignored)
src/elt_environment/
  agent_worker.py              launch original agents and redirect model transport
  protocol.py                 translate provider messages and streaming responses
  bridge.py                   provider requests <-> Tinker observations/actions
  runner.py                   workspace, runtime lock, process and container cleanup
  env.py                      multi-turn Cookbook environment and group retries
  grading.py                  trusted grader command or explicit offline diagnostic
  admission.py, tasks.py      verify inputs, discover tasks and split families
  train.py, probe.py          training and evaluation entry points
.state/                        private attempts, worker logs and training output
```
