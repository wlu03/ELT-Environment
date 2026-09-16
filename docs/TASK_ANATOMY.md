# Task anatomy and admission boundary

This review separates two things that look similar on disk but have different
trust properties:

1. an active generator workspace under `ELT-taskgen/runs/.../tasks/`; and
2. an immutable combined release under `release/public` and `release/private`.

Only the second is an environment input. Generator workspaces contain oracle
material and may be incomplete while a pipeline is running.

## Inspected examples

### Released contract fixture

The clearest current combined release is:

```text
ELT-taskgen/runs/runtime_canary_20260901/snowflake/release/
  public/gate__five_backend_probe/
```

`elt-taskgen verify` accepted all 101 pinned release files. Its manifest is
schema 3.3 with `public_layout: combined`. It is a runtime canary, not an
admitted training task: its solver prompt is empty, its transformation prose is
stub-like, and it has no attack cases. Use it to test loading and interfaces,
not to train a policy.

### Generator-workspace anatomy

The richer current task inspected was:

```text
ELT-taskgen/runs/current_2026-09-05_five_sources_retry2/tasks/dlt__workable/
```

It is at `gold_frozen`, not `accepted` and not a combined frozen release. It
contains 17 source tables, three marts, five populations, nine relationships,
and 28 attack cases. It is useful for understanding what Taskgen produces, but
the environment must never hand this directory to a rollout.

## Combined release shape

```text
release/
├── release_manifest.json
├── checksums.sha256
├── public/<task_id>/
│   ├── config.yaml
│   ├── data_model.yaml
│   ├── schemas/
│   ├── documentation/
│   │   ├── README.md
│   │   └── source, destination, connection, and job guides
│   ├── check_job_status.py
│   ├── <destination>_credential.json       # blank template only
│   └── elt/main.tf
└── private/
    ├── <task_id>/
    │   ├── semantic/task_ir.json
    │   ├── populations/
    │   │   ├── development/rendered/
    │   │   ├── primary/rendered/
    │   │   ├── resampled/rendered/
    │   │   ├── counterfactual/rendered/
    │   │   └── stress/rendered/
    │   ├── oracle/<population>.duckdb
    │   └── answer_key/
    │       ├── manifest.json
    │       ├── table.json
    │       ├── sort_key.json
    │       ├── reference and evaluation SQL
    │       ├── gold/<population>/
    │       └── runtime/airbyte_connector_contract.json
    ├── <task_id>__el/reward.json            # private stage evidence
    └── <task_id>__t/reward.json             # private stage evidence
```

The canonical public ID is `<task_id>`. The suffixed private directories do not
create two training examples.

## What the public fixture asks the agent to build

`gate__five_backend_probe` combines five extraction backends:

| Backend | Source table | Private rendered form |
| --- | --- | --- |
| PostgreSQL | `customers` | `postgres/customers.sql` |
| MongoDB | `orders` | `mongodb/orders.jsonl` |
| Flat file | `order_items` | `files/order_items.csv` |
| REST | `events` | paginated `rest/events/` |
| S3 | `metrics` | `s3/metrics/part-*.jsonl` |

It requests two marts:

| Mart | Grain/key | Required result |
| --- | --- | --- |
| `customer_rollup` | one row per `customer_id` | completed-order count and item spend, retaining customers without orders |
| `event_wide` | one row per `event_id` | event fields joined to optional measurement fields with exact type/null behavior |

An artifact-level candidate ultimately supplies:

```text
elt/main.tf
elt/dbt_project.yml
elt/models/sources.yml
elt/models/*.sql
```

The harness supplies the dbt profile. Candidate profiles, dependency packages,
macros/hooks, Python models, symlinks, transient state, secrets, and writes
outside the candidate `elt/` surface are rejected by `workspace-v1`.

## TaskIR and generated workspace

The private TaskIR is the generator/scorer authority. Its major fields are:

```text
schema_version, task_id, family_id, cluster_id, origin, license, attribution,
title, solver_prompt, tables, relationships, backends, marts, populations,
reference, attack_cases
```

A raw generated task expands this into:

```text
task_ir.json
populations/<population>/
  rows/*.jsonl
  rendered/{files,mongodb,postgres,rest,s3}/...
answer_key/
  reference/<mart>.sql
  gold/<population>/{stage1_counts.json,<mart>.csv}
attacks/<mutation>/...
reports/...
```

This entire tree is curation-side. Serializing TaskIR, even after deleting a
few obvious keys, is not an acceptable prompt renderer.

## Populations and visibility

Each task has five deterministic populations:

- `development`: available only through policy-safe development tools or an
  approved public observation; never part of the aggregate reward;
- `primary`, `resampled`, `counterfactual`, and `stress`: hidden graded
  populations, replayed independently with fresh state.

The submitted artifact is identical across hidden replays. The scorer uses the
minimum hidden-population result, preventing a solution that only works on the
most convenient population.

## One task, two stages

### Stage 1: Extraction/Load

The Terraform/Airbyte intent selects every required stream and lands raw tables
into a fresh attempt warehouse. `workspace-v1` checks the admitted Terraform
surface, sync lifecycle, required table/schema/content behavior, and strict EL
pass. The historical cloud contract is count-oriented; the local workspace
proxy adds stricter raw checks where its versioned contract permits them.

### Stage 2: Transformation

Against that same attempt state, the submitted dbt project is parsed, compiled,
and run. Each required mart is compared with private gold using the existing
ELT-taskgen comparator, including columns, rows, nulls, numeric behavior, and
private sort order. Raw tables must remain unchanged.

No stage reset occurs between EL and T. A later interactive environment should
make the transition observable when development raw state is ready, but it
must preserve the same episode, task ID, attempt, and submitted artifact.

## Agent-visible boundary

The policy may receive:

- approved files beneath `public/<task_id>/`;
- a normalized public file listing;
- bounded development table/schema/sample/query results;
- sanitized tool status and stable error codes;
- its own candidate files and development execution output.

The policy and Tinker telemetry must never receive:

- `private/`, TaskIR, gold CSVs, reference/evaluation SQL, expected counts, or
  hidden population data;
- real credentials, Airbyte IDs, cloud administrator state, Terraform state,
  dbt profiles, or host environment values;
- absolute private/attempt/sealed paths or raw grader exceptions;
- cross-rollout files, databases, logs, tool output, or cache state.

## Admission rules for a training catalog

A catalog entry is eligible only if all of these are true:

1. It comes from `release_manifest.json`, not directory discovery.
2. The release checksum/signature verification succeeds.
3. The manifest declares the current supported schema-3 combined layout.
4. The parent task is accepted for training; canaries and incomplete statuses
   are excluded unless an explicit harness-test catalog selects them.
5. Public documentation fully states the task without relying on private
   reference semantics.
6. Prompt and a known-correct response fit the selected model renderer,
   tokenizer, context window, and output limit.
7. All required hidden populations and scorer assets load through
   `load_workspace_package`.
8. Train/eval membership is family- and cluster-disjoint and recorded in the
   manifest or an immutable companion catalog.
9. Known-correct and known-bad artifacts produce expected labels before the
   task enters a training batch.
10. No public-file or rendered-observation leak scan fires.

The environment should quarantine a task that fails admission. Context overflow,
missing private evidence, and package incompatibility are harness/task defects,
not policy failures and therefore never reward `0.0`.

