# ELT-Environment

GRPO training on Tinker for the two-stage ELT tasks produced by
[`ELT-taskgen`](../ELT-taskgen).

> **Status:** runnable single-turn trainer. The policy reads a task's public
> bundle and replies with the whole project in one message. ELT-taskgen's
> grader scores it locally with DuckDB on four hidden populations. There is
> no interactive tool loop yet.

# Rollout

One rollout is one parent task and one continuous attempt:
1. public task
2. Extract/Load
3. raw tables
4. Transform
5. marts
6. hidden replay

`<task_id>__el` and `<task_id>__t` are not separate public tasks, dataset rows, episodes, or GRPO groups.

The policy writes `elt/main.tf`, `elt/dbt_project.yml`, `elt/models/sources.yml`
and one `elt/models/<mart>.sql` per mart, each in a
`<file path="elt/...">` block. The reward is ELT-taskgen's
`workspace-signal-v1`: 0 when Extract/Load fails, 0.5 when Extract/Load passes
and no mart matches, up to 1.0 when every mart matches on every hidden
population. A grader fault is not a reward: it discards the whole group, which
is sampled again. If three attempts at a group all fault, the run stops rather
than train on a smaller batch.

The prompt holds four of the task's public files: `documentation/README.md`,
`config.yaml`, `data_model.yaml` and the starter `elt/main.tf`. Credential
values in `config.yaml` are replaced by `<supplied as a Terraform variable>`;
the grader requires credentials to be Terraform variables, so the model never
needs them. The other public files are left out:

- The Airbyte and Terraform reference documents are the same for every task
  (13,425 tokens). The example reply in the system prompt uses every Terraform
  attribute that the 50 known-correct replies use.
- Each `schemas/<table>.csv` repeats the README's Source tables entry for that
  table, which also gives each column's type. A CSV is left out only when its
  rows equal the rows taskgen writes from that entry.
- Other destinations, credential templates and the job-status script do not
  affect the reply.

## Run

`.env` holds `TINKER_API_KEY` and `TINKER_PROJECT_ID`.

Measure the base model before training. A task whose group rewards are all
equal gives no gradient.

```bash
.venv/bin/python -m elt_environment.probe tasks=6 samples=4
```

Train:

```bash
.venv/bin/python -m elt_environment.train max_steps=20
```

Main settings, all command-line arguments of `elt_environment.train`:

| Setting | Default |
| --- | --- |
| `model_name` | `Qwen/Qwen3.8-27B`, thinking off |
| `group_size` x `groups_per_batch` | 8 x 8 |
| `learning_rate`, `lora_rank` | 1e-5, 32 |
| `max_tokens` | 10240 |
| `eval_families` | 5 families held out, evaluated at step 0 and every 5 steps |
| `grader_concurrency` | 8 graders at once |
| `releases_root` | `../ELT-taskgen/runs/batch50c_20260923/workspace/releases` |

Runs, checkpoints and grader attempts go under `.state/`, which is not tracked.

## Measured on batch50c (2026-09-28)

All 50 tasks fit a 64K context with the 10,240-token reply budget. Prompts
average 23,046 tokens; the longest is 47,770. Base models, 24 replies on six
training tasks, before any training, measured with the earlier prompt that
still included the reference documents:

| Model | Extract/Load pass | Groups with reward variance | Transform credit |
| --- | --- | --- | --- |
| `Qwen/Qwen3.6-35B-A3B` | 12-21% | 2-3 of 6 | none seen |
| `Qwen/Qwen3.8-27B` | 75% | 4 of 6 | seen in a one-step run: rewards 0, 0.5, 0.75 and 1.0 |

With the current prompt, `Qwen/Qwen3.8-27B` on the same six tasks passed
Extract/Load in 14 of 24 replies, with a mean reward of 0.34 and two replies at
1.0. Three of the ten failures merged several `flat_files` tables into one File
source, which the earlier prompt's File source document had prevented.

## Known limits

- The policy writes the project in one reply. ELT-Bench agents run Terraform,
  Airbyte syncs and dbt over many turns with a shell; that loop is not trained.
- The offline Terraform grader accepts a narrower form than a real Airbyte run.
  It refused all 150 Terraform files written by ELT-Bench agents on these
  tasks, although every one loaded the right data into Snowflake. The system
  prompt states the rules it enforces, including one value the task bundle
  never publishes: `number_data_type = "NUMBER(38,9)"` on the Snowflake
  destination.
- Model SQL must stay inside the grader's portable subset, which refuses
  functions such as `UPPER`, `ABS` and `GREATEST`. The prompt lists the subset;
  `tests/test_prompt_rules.py` checks that list against the grader.

## Repository

```text
ELT-Environment/
  configs/
    pilot.example.toml       earlier settings draft; the train CLI replaces it
  docs/
    TASK_ANATOMY.md          inspected generated/released task shape
    ARCHITECTURE.md          ownership, episode, sandbox, data flow
    TINKER_INTEGRATION.md    exact Cookbook interfaces and GRPO retry rule
    BUILD_PLAN.md            workstreams, tests, blockers, exit criteria
  src/elt_environment/
    tasks.py                 release discovery and family split
    prompt.py                public bundle to chat messages, grader rules
    example_reply.txt        correct reply for the taskgen fixture task
    artifact.py              reply to files and back
    admission.py             release check and context fit per task
    grading.py               ELT-taskgen grader to reward or discard
    env.py                   Cookbook Env, group builder, RetryWholeGroup, dataset
    train.py                 GRPO entry point
    probe.py                 base-model pass rates without training
    outcomes.py              no-label versus numeric-reward control flow
  tests/                     ELT_ENVIRONMENT_GRADING_TESTS=1 runs real dbt grading
  pyproject.toml
  uv.lock                    Python 3.12; ELT-taskgen installed from ../ELT-taskgen
```
