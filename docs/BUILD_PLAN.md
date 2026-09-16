# Build plan and readiness review

## Bottom line

The trusted ELT execution/reward core already exists. The largest new work is
not another scorer; it is a production-quality agent coordinator around that
core: corpus admission, policy-safe observations, Tinker adapters, strict group
retry, lifecycle/concurrency, and eventually an adversarial tool sandbox.

Two prerequisites block a meaningful training pilot today:

1. a reviewed, immutable `ELT-taskgen` build; and
2. fresh accepted schema-current combined task releases with real public
   specifications and family-disjoint splits.

The verified canary can still drive loader and failure-path integration tests.

## Readiness matrix

| Area | Reuse from Taskgen | Build in this project | State |
| --- | --- | --- | --- |
| Combined task/release contract | schema-3 manifest and package loader | supported-version/admission policy | partial; canary only |
| Corpus | generation, gates, populations, attacks | immutable catalog and family splits | blocked on releases |
| Public observation | public release files | versioned renderer, leak scan, token preflight | missing |
| L1 action | workspace artifact mapping | exact text envelope/parser | missing |
| L2 actions | Airbyte proxy/dev query/dbt primitives | tool schemas, state machine, sequential dispatcher | missing |
| Workspace lifecycle | install, admit, seal, clean replay | async adapter and attempt registry | mostly reusable |
| EL execution | Terraform intent and trusted local sync | optional interactive development controls | reusable for L1 |
| T execution | pinned dbt runner | optional interactive development controls | reusable for L1 |
| Grading | hidden replay and comparator | narrow worker/client adapter | mostly reusable |
| Reward | `workspace-signal-v1` | `ValidReward`/`DiscardGroup` bridge only | foundation added |
| GRPO groups | taskgen's framework-neutral rule | Cookbook builder and `RetryWholeGroup` | missing |
| Training | none | Tinker dataset/config/loop/checkpoints | missing |
| Isolation | sealed artifact checks/process limits | L2 rootless container or microVM | missing for L2 |
| Concurrency | killable scorer child | async backpressure and capacity accounting | missing |
| Telemetry | stable scorer codes/heads | safe event schema and restricted diagnostics | missing |
| Evaluation | gold/attacks/calibration evidence | fixed held-out runner and reports | missing |
| Reproducibility | release hashes/scorer versions | full run manifest and resume validation | missing |
| Cloud transfer | separate runtime certification | trigger/report only, never train by default | separate path |

## Phase 0 — stabilize inputs

Deliverables:

- create a reviewed initial commit for `ELT-taskgen`;
- build a wheel and record its source/wheel digest;
- prove installed-package resource loading and run the targeted training tests;
- freeze at least 2–4 real accepted combined tasks under the current release
  schema, across more than one family;
- record immutable train/validation/test family assignments in a manifest or
  companion catalog;
- designate the synthetic canary as `harness_test`, never `train`.

Exit tests:

- release verification succeeds from a clean checkout;
- `load_workspace_package` loads every admitted task;
- known-correct artifacts label `1.0` and curated attacks receive their
  expected lower scores;
- public task prose alone contains all semantics needed for the expected
  solution;
- rebuilding the environment from recorded digests yields the same labels.

## Phase 1 — catalog, renderer, and preflight

Add:

```text
src/elt_environment/
  config.py
  catalog.py
  prompts.py
  admission.py
```

Responsibilities:

- read tasks/splits only from verified metadata;
- preserve parent identity and family/cluster provenance;
- allowlist public files and render them deterministically;
- scan observations for private material and known leak canaries;
- preflight prompt plus known-correct response against the selected Tinker
  model's exact tokenizer, renderer, stops, and limits;
- quarantine tasks with stable reason codes before batching.

Exit tests cover corrupted manifests, split overlap, symlinks, private path/key
leaks, credential-shaped values, deterministic rendering, and context overflow.

## Phase 2 — L1 artifact environment

Add:

```text
src/elt_environment/
  actions.py
  env.py
  grader.py
  lifecycle.py
  backends/workspace.py
```

Responsibilities:

- freeze and strictly parse `artifact-envelope-v1`;
- translate relative files to the existing taskgen artifact mapping;
- install exactly one fresh workspace per Env;
- grade terminally outside the event loop with a hard deadline;
- translate taskgen signal to `ValidReward` or `DiscardGroup` without
  recalculating it;
- sanitize all policy observations and telemetry;
- always close/retain evidence according to an explicit retention policy.

Exit tests include a known-correct completion, malformed envelopes, every
workspace policy violation, grader timeout/crash/no-label, cancellation during
dbt, artifact cap boundaries, and no private detail in model-visible output.

## Phase 3 — Tinker/GRPO bridge

Add:

```text
src/elt_environment/
  dataset.py
  group.py
  tinker_env.py
  train.py
  checkpoints.py
```

Responsibilities:

- implement Cookbook `Env`, `EnvGroupBuilder`, `RLDataset`, and dataset builder;
- ensure each group is one parent task with independent fresh attempts;
- implement `RetryWholeGroup` exactly as specified in
  [`TINKER_INTEGRATION.md`](TINKER_INTEGRATION.md);
- start in synchronous on-policy mode with constant-group filtering disabled;
- bound group, grader, dbt, and hidden-replay concurrency;
- save both Tinker checkpoint kinds plus the local deterministic run manifest.

Exit tests include pickle/spawn, complete-group retry, zero-versus-no-label,
retry exhaustion, batch shape, deterministic dataset order, and exact resume.

## Phase 4 — inference-only pilot

Before optimization:

1. run the admitted catalog through initial-observation preflight;
2. sample complete groups from the real Tinker client without training;
3. exercise numeric zero, partial reward, full reward, and synthetic no-label;
4. confirm resource peaks and teardown after forced cancellation;
5. inspect only sanitized traces for prompt/action correctness;
6. compare taskgen CLI labels with environment labels byte-for-byte/field-for-field.

Exit criterion: at least 100 consecutive labelled rollouts with no leaked data,
orphan process/container/database, silent batch shrink, or reward disagreement.

## Phase 5 — small synchronous GRPO run

Start with one concurrent group and a small fixed train/eval catalog. Record:

- reward mean/distribution and constant-group rate;
- `r_EL`, `r_T`, EL-gated final reward, and first failed phase;
- no-label causes, whole-group retry count, and exhaustion count;
- tokens/turns/tool calls, sampling/grading latency, CPU/RAM/disk peaks;
- task/family sampling balance and held-out performance;
- policy-violation and canary-hit rates.

Promotion requires a successful optimizer-resume test, stable eval tasks, no
reward drift from direct taskgen scoring, and a manual trace/privacy audit.

## Phase 6 — L2 interactive sandbox

Add only after L1 is reliable:

```text
src/elt_environment/
  tools/
    files.py
    warehouse.py
    terraform.py
    sync.py
    dbt.py
    terminal.py
  sandbox/
    base.py
    container.py       # or a remote/microVM backend
```

Implement the episode state machine and capability-scoped tools from
`ARCHITECTURE.md`. Tool execution is sequential. Development execution may
help the policy debug, but the terminal reward still comes from sealed hidden
replay of the same artifact.

Security exit tests include path/symlink/Unicode escapes, fork bombs, process
leaks, disk/output bombs, environment/credential probing, network/metadata
access, Docker socket access, private mount discovery, SQL external-access
attempts, dbt macro/package abuse, cancellation races, and cross-rollout state.

## Phase 7 — evaluation and anti-reward-hacking

Add fixed reports for:

- held-out families/clusters and origins;
- all four hidden populations and minimum aggregation;
- attack/mutation suites, including hardcoded-primary and skipped-extraction
  shortcuts;
- planted infeasible canaries and honest-abort behavior;
- stage-specific success, end-to-end success, and raw immutability;
- length/tool efficiency as diagnostics only, unless a future reward version
  explicitly and empirically adds them;
- L1-to-L2 transfer and sparse real-runtime certification.

No evaluation query or diagnostic should expose gold values to the policy.

## Phase 8 — operations and CI

Add:

- unit CI without credentials;
- integration CI against the synthetic combined canary;
- scheduled acceptance runs against the pinned taskgen wheel/dbt runtime;
- lockfile generation under Python 3.12 and dependency/source attestations;
- capacity preflight, disk cleanup, checkpoint retention, and run cancellation;
- separate public summaries and access-controlled grader evidence;
- explicit API-key injection and redaction tests.

Cloud runtime certification remains a separate, sparse job with scoped
credentials and its own claim. A local RLVR pass must never be relabelled as a
Snowflake, Databricks, or Redshift certification.

## Target source tree

```text
ELT-Environment/
├── configs/
│   ├── pilot.example.toml
│   ├── train.example.toml
│   └── eval.example.toml
├── docs/
├── src/elt_environment/
│   ├── actions.py
│   ├── admission.py
│   ├── catalog.py
│   ├── checkpoints.py
│   ├── config.py
│   ├── dataset.py
│   ├── env.py
│   ├── grader.py
│   ├── group.py
│   ├── lifecycle.py
│   ├── outcomes.py
│   ├── prompts.py
│   ├── telemetry.py
│   ├── tinker_env.py
│   ├── train.py
│   ├── evaluate.py
│   ├── backends/
│   │   ├── workspace.py
│   │   └── semantic.py
│   ├── sandbox/
│   └── tools/
└── tests/
    ├── unit/
    ├── integration/
    ├── security/
    └── fixtures/
```

## Pilot definition of done

The L1 pilot is done only when:

- a clean installation is pinned to immutable taskgen and release digests;
- at least two real tasks pass admission and held-out splitting;
- the environment produces identical taskgen labels for correct/bad fixtures;
- a same-task Tinker group reaches training with exactly `group_size` members;
- injected no-label faults retry the entire group with all-new IDs;
- a small GRPO job updates, evaluates, checkpoints, stops, and resumes;
- privacy, cancellation, cleanup, and capacity tests pass;
- every claim is explicitly `artifact_workflow_proxy`, not cloud/runtime parity.

## Decisions still needed before implementation

- the initial Tinker base model and its context/output limits;
- training/evaluation budget and desired group/batch sizes;
- the first immutable real release/catalog path;
- local rootless container versus remote/microVM backend for L2;
- rollout evidence retention period and access controls;
- whether L0 semantic JSON is worth a curriculum experiment after L1 works.

These choices tune the implementation; they do not change the one-task,
two-stage, single-scorer, whole-group-retry architecture.

