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
3. Put taskgen bundles under `training-data/<name>/`, with matching `public/<task_id>/` and `private/<task_id>/` directories. Discovery reads `family_id` from private `task_ir.json` (or `semantic/task_ir.json`) and the destination from public `config.yaml`. Public EL/T variants without a private parent record are not separate training examples. You can also point `releases_root` at a single bundle.

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

Training shows an overall progress bar with completed steps, elapsed time and
estimated time remaining, alongside the per-batch sampling bar. It respects
`max_steps` and resumes from the checkpoint's completed step count. Add
`progress_bar=false` to disable the overall bar.

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

### DuckDB development target

Use `agent=swe` and replace the warehouse section of both the task's public
config and installed development config with:

```yaml
duckdb:
  config:
    definition_id: 94bd199c-2ff0-4aa2-b98e-17f0acb72610
    schema: main
```

Use the DuckDB connector definition ID in your Airbyte workspace. For sources
that emit global checkpoints, including MongoDB, build and register the
patched connector at `runtime/airbyte-duckdb/Dockerfile`. Use that custom
definition ID in both the public and installed configs. Keep the Airbyte
workspace and source settings in the installed development config.
Install the scoped workspace credentials and seed development data before
running a probe. The public bundle contains task inputs and templates.

Build the DuckDB solver image from the existing `elt-swe` image.

```bash
docker build -t elt-swe-duckdb -f agents/SWE-agent/docker/elt-swe/DuckDB.Dockerfile agents/SWE-agent/docker/elt-swe
```

Set `duckdb_local_root` to a dedicated host directory mounted as `/local` in
the Airbyte destination worker. The host user and the Airbyte worker need
write access through the mount. The solver receives only its attempt's
subdirectory at `/duckdb`. Use Airbyte access scoped to the development
workspace and restrict destination writes to the attempt directory.

Airbyte and the solver use the same `warehouse.duckdb` file. Finish each
Airbyte sync before starting the next sync or running dbt. With the DuckDB
destination, source rows land in `_airbyte_raw_<stream>` as JSON in
`_airbyte_data`. Use `profile: elt_environment` and `dbt run --profiles-dir
/workspace` to build the models in `elt_models`.

Run the bundled DuckDB replay grader with the same Python environment as the
coordinator.

```bash
grader_command='/absolute/path/to/.venv/bin/python -m elt_environment.duckdb_grader --runtime /private/attempt-runtime.json'
```

The private runtime JSON describes one provisioned task. Set `release_dir`,
`task_id`, `workspace_id`, `definition_id`, `local_root`, `airbyte_url`, and
`airbyte_credentials_file` (a file containing the Airbyte client ID and
secret). Set `development` to the serialized taskgen `SourceEnvironment`
used to seed that workspace. The grader reuses the attempt's Terraform
connections in the task-owned workspace. It stops development, syncs each
hidden population into a fresh DuckDB file, checks raw row counts, and runs
dbt without network access. Give this runtime one live rollout at a time.

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
DuckDB requests include a private `runtime` object with the attempt storage
path and the Airbyte connection IDs found in Terraform state. The grader
intersects those IDs with the task workspace's registered connections.
The grader request, release private tree and results stay outside the agent mount.
The agent sees only the installed task workspace and its development services.

`grader_command` is required for training and probes. The grader owns validation
of private replay assets and must report missing or invalid assets as an
unlabelled outcome. Admission checks task metadata and installed development
inputs; it does not certify hidden replay assets. Keep bundles immutable during
a run. There is no bundled offline diagnostic scorer.

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
  grading.py                  trusted hidden-replay grader command
  admission.py, tasks.py      verify inputs, discover tasks and split families
  train.py, probe.py          training and evaluation entry points
.state/                        private attempts, worker logs and training output
```
