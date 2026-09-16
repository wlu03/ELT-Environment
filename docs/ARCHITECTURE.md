# Environment architecture

## Decision

Build `ELT-Environment` as a thin Tinker coordinator around the verified
`ELT-taskgen` environment. Do not fork the comparator, replay logic, or reward
formula.

Start with the L1 artifact environment and then preserve its terminal grader
while replacing the one-shot action with an L2 tool loop. This gives one
vertical slice that can prove task selection, GRPO grouping, remote sampling,
grading, labels, optimizer updates, checkpoint/resume, and held-out evaluation.

## Trust and process boundaries

```text
┌──────────────── Tinker service (remote model compute) ────────────────┐
│ sample actions/tokens       train on labelled trajectory groups       │
└───────────────────────────────┬───────────────────────────────────────┘
                                │ model I/O only
┌───────────────────────────────▼───────────────────────────────────────┐
│ ELT-Environment coordinator (trusted)                                │
│ catalog · public renderer · dataset · group retry · limits · metrics │
└──────────────┬───────────────────────────────┬────────────────────────┘
               │ policy-safe tools             │ sealed artifact + IDs
┌──────────────▼────────────────┐  ┌───────────▼────────────────────────┐
│ fresh rollout sandbox        │  │ trusted taskgen grader worker      │
│ public task RO · elt/ RW     │  │ private release · hidden replays  │
│ development state · no egress│  │ gold/comparator · numeric signal  │
└───────────────────────────────┘  └────────────────────────────────────┘
```

The sandbox cannot mount the private release. The grader cannot return private
details to the policy. The coordinator receives only an authenticated sealed
artifact, numeric heads, and sanitized status codes.

## Ownership

| Concern | Authority |
| --- | --- |
| Task generation, acceptance, release verification | `ELT-taskgen` |
| TaskIR, populations, gold, attacks | `ELT-taskgen` private release |
| Workspace admission/sealing/replay | `elt_taskgen.training.workspace` |
| Terraform intent and trusted sync | `elt_taskgen.training` |
| dbt execution and semantic comparison | `elt_taskgen.training.scorer` |
| Reward projection and no-label classification | `elt_taskgen.training.signal` |
| Catalog/splits, observations, actions, Tinker objects | `ELT-Environment` |
| Group retry, concurrency, cleanup, telemetry | `ELT-Environment` |
| Adversarial interactive isolation | `ELT-Environment` sandbox backend |
| Remote sampling, optimization, model checkpoints | Tinker |

The grader adapter must be narrow: input is a release/task/sealed-artifact
identity, output is taskgen's versioned result projected to either
`ValidReward` or `DiscardGroup`. There is no fallback scorer.

## Fidelity levels

### L0: semantic JSON diagnostic

The existing `semantic-v1` scorer accepts a load plan plus SQL by mart. It is
cheap and useful for debugging task/corpus semantics, but it does not train an
agent to create the real Terraform/dbt artifact. Keep it as an optional
curriculum or harness diagnostic, not the target environment.

### L1: terminal artifact pilot

The model emits one strict JSON envelope. Proposed version-one shape:

```json
{
  "schema_version": "artifact-envelope-v1",
  "files": [
    {"path": "main.tf", "content": "..."},
    {"path": "dbt_project.yml", "content": "..."},
    {"path": "models/sources.yml", "content": "..."},
    {"path": "models/example.sql", "content": "..."}
  ]
}
```

Paths are relative to candidate `elt/`. The parser must require exact keys,
reject duplicates and Unicode/case collisions, perform no Markdown extraction
or repair, preserve content bytes deterministically, and apply taskgen's file
and byte caps before grading. The envelope is an adapter format only; the
authoritative submission is the sealed workspace artifact.

`initial_observation()` contains the approved public task, exact response
schema, limits, and no hidden facts. `step()` parses once, calls the existing
one-step `DeclarativeEltEnv`, returns a float only for a real label, and ends
the episode.

### L2: interactive agent target

One episode moves through these states:

```text
CREATED -> EL_WORKING -> RAW_READY -> T_WORKING -> SUBMITTED -> GRADED
    └────────────── abort(infeasible) ────────────────┘
any state -> HARNESS_FAILED -> discard complete GRPO group
```

The state is harness-owned. Merely claiming that EL is complete does not move
to `RAW_READY`; a successful development sync/validation does. The terminal
submission seals the complete artifact and privately replays it over every
graded population.

Initial capability-scoped tools should be:

- `list_files`, `read_file`, `write_file`, `delete_file` under candidate
  `elt/` only;
- `list_tables`, `describe_table`, `sample_rows`, `query_readonly` against the
  development warehouse with row/byte/time limits;
- `validate_terraform`, `apply_development_sync`, and `get_sync_status` through
  the trusted offline Airbyte proxy;
- `run_dbt` with an allowlisted command/selector grammar and harness profile;
- `submit` and `abort(reason_code="infeasible")`.

Execute one tool call at a time. Tinker Cookbook's general tool helpers may run
multiple calls concurrently; that is unsafe for a shared mutable attempt.
Avoid a general `bash` tool in the first interactive version. If shell access is
later justified, it stays inside the same hardened sandbox and adds no network,
mount, or credential authority.

## Observation renderer

The renderer is a versioned security boundary, not string concatenation over a
task object. It should:

1. start from the verified public task tree;
2. allowlist files and fields per renderer version;
3. normalize ordering, newlines, encodings, and path display;
4. render development observations through bounded typed tool results;
5. scan the final message for private paths, forbidden key names, gold values,
   credentials, and known canary tokens;
6. preflight both prompt and known-correct response with the exact Tinker model
   tokenizer/chat renderer/stop conditions;
7. store a renderer version and prompt digest with every rollout.

Do not render private TaskIR and then try to redact it.

## Grading and reward

Use `workspace-signal-v1` unchanged:

```text
R = (r_EL + 1[EL_pass] * w_T * r_T) / (1 + w_T)
```

With `w_T = 1`, EL-only success is `0.5`, end-to-end success can reach `1.0`,
and failed EL receives `0.0`. `r_T` is the taskgen scorer's minimum hidden
end-to-end reward; development is not aggregated. Policy violations and valid
but incorrect submissions receive numeric zero.

Task, harness, and infrastructure faults have `reward=None`. The adapter maps
that to `DiscardGroup`, raises before constructing a Tinker terminal result,
and retries every sibling from fresh state. It never maps no-label to zero or
NaN.

Stage heads are diagnostics for curriculum/evaluation. Standard Cookbook GRPO
assigns the episode-level advantage to all sampled action tokens. True
stage-local credit would require an explicitly different datum construction and
should not be claimed by merely logging `r_EL` and `r_T`.

## Sandbox contract

L1's existing sealed artifact admission is strong, but it is not an isolation
boundary for arbitrary model-authored commands. L2 needs a fresh rootless
container, VM, or microVM per rollout with:

- public task mounted read-only and one empty candidate directory writable;
- no private release, host repository, home directory, SSH agent, Docker
  socket, cloud metadata, or inherited secret environment;
- no external network; only explicit in-sandbox/trusted proxy capabilities;
- non-root user, read-only root, dropped capabilities, no-new-privileges;
- CPU, memory, PID, disk/inode, output, tool-call, token, and wall-clock limits;
- a killable process group and cancellation-safe, unconditional teardown;
- unique attempt, database, namespace, cache, temp, and service identifiers.

Every hidden population is also a fresh replay. Never reuse Terraform state,
Airbyte IDs/jobs, DuckDB files, dbt `target/`, logs, or append-mode raw tables.

## Dataset and split contract

The Tinker dataset yields one pickleable/lightweight group builder per parent
task. `make_envs()` lazily creates `group_size` independent attempts for that
same task. Heavy clients, package objects, database handles, locks, and event
loops are created in the worker, not serialized in the builder.

Catalog selection is manifest-driven and records:

- release and package digests;
- parent task, origin, family, cluster, destination, and difficulty strata;
- renderer/action/reward/scorer versions;
- train/validation/test split and quarantine reason;
- known-correct preflight and admission evidence.

Splits must be family/cluster disjoint, not only task-ID disjoint. Evaluation
uses fixed tasks, seeds, generation settings, and checkpointed sampler state.

## Concurrency and process model

Effective rollout demand is approximately:

```text
group_size * groups_per_batch * in-flight batches
```

Each member can fan out into four hidden replays and a dbt subprocess, so start
with one concurrent group and measure CPU, RAM, disk, and latency before
raising it. Run synchronous on-policy GRPO first; asynchronous training adds
policy staleness and makes failure accounting harder.

Blocking taskgen grading runs outside the async event loop behind a bounded
semaphore/worker pool. Any multiprocessing pool must use `spawn` or
`forkserver`; never fork a process after Tinker has started background threads.

## Checkpoint and evidence contract

A resumable run needs more than model weights:

- Tinker sampler checkpoint;
- Tinker training/optimizer checkpoint;
- dataset cursor, shuffle seed, and task sampler RNG;
- release manifest/package digest and admitted task IDs;
- model/tokenizer, renderer, action, environment, scorer, and reward versions;
- group retry and concurrency configuration;
- completed evaluation step IDs and aggregate metrics.

On resume, mismatch should fail closed unless an explicit migration creates a
new run identity.

Telemetry has two levels: policy-safe numeric metrics/stable codes and
restricted harness diagnostics. Hidden rows, expected counts, reference SQL,
credentials, absolute private paths, and raw exceptions never enter Tinker
logs.

