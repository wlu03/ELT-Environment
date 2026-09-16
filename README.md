# ELT-Environment

Tinker/GRPO orchestration for the two-stage ELT tasks produced by
[`ELT-taskgen`](../ELT-taskgen).

> **Status:** architecture and safety scaffold. The task/reward audit is
> complete; the Tinker adapter and interactive sandbox are not implemented
> yet. This directory must not be described as a runnable trainer until the
> pilot exit criteria in [`docs/BUILD_PLAN.md`](docs/BUILD_PLAN.md) pass.

## Core contract

One rollout is one parent task and one continuous attempt:

```text
public task -> Extract/Load -> raw tables -> Transform -> marts -> hidden replay
                    r_EL                         r_T          final reward
```

`<task_id>__el` and `<task_id>__t` are private evidence names. They are not
separate public tasks, dataset rows, episodes, or GRPO groups.

This project owns the Tinker-facing coordinator:

- manifest-driven train/eval task selection;
- leak-checked observations and prompt/token preflight;
- one fresh attempt and sandbox per rollout;
- the policy action/tool protocol;
- GRPO group construction and whole-group retry;
- bounded concurrency, cleanup, telemetry, checkpointing, and evaluation.

It deliberately does **not** own task generation, gold data, comparison, or
reward math. Those remain in `ELT-taskgen`, whose verified package loader,
workspace lifecycle, scorer, and `workspace-signal-v1` projection are the
single authorities.

## Recommended delivery path

1. **L1 artifact pilot:** one completion emits an exact, bounded artifact
   envelope containing `main.tf`, a dbt project, sources YAML, and SQL models.
   Adapt that envelope to `DeclarativeEltEnv.step()` and prove the full Tinker
   data/reward/checkpoint path without exposing a shell.
2. **L2 interactive agent:** retain the same task and terminal grader, but add
   sequential, capability-scoped file, development-warehouse, Terraform-proxy,
   sync, and dbt tools inside a rootless container or microVM.
3. **Sparse runtime certification:** keep real Airbyte/cloud-warehouse checks
   outside ordinary GRPO. They certify transfer; they do not replace the local
   semantic reward.

The L1 pilot is intentionally an integration milestone, not the final agent
experience. It removes Tinker and grouping uncertainty before introducing an
adversarial code-execution surface.

## Repository map

```text
ELT-Environment/
  configs/
    pilot.example.toml       documented initial settings
  docs/
    TASK_ANATOMY.md          inspected generated/released task shape
    ARCHITECTURE.md          ownership, episode, sandbox, data flow
    TINKER_INTEGRATION.md    exact Cookbook interfaces and GRPO retry rule
    BUILD_PLAN.md            workstreams, tests, blockers, exit criteria
  src/elt_environment/
    outcomes.py              no-label versus numeric-reward control flow
  tests/
    test_outcomes.py         frozen coordinator boundary tests
  pyproject.toml
  uv.lock                    Python 3.12 dependency resolution
```

The eventual source tree is specified in the build plan. Empty placeholder
modules are not included: each module should arrive with its contract tests.

## What is already reusable

`ELT-taskgen/src/elt_taskgen/training/` already provides:

- verified combined-release loading;
- fresh workspace installation, admission, sealing, and clean replay;
- normalized Terraform-to-Airbyte intent compilation and trusted local sync;
- real pinned `dbt-duckdb` execution against the same fresh state;
- hidden-population scoring, strict raw immutability, and minimum aggregation;
- the canonical two-head training signal and `reward=None` semantics;
- a one-step artifact environment and a framework-neutral group adapter.

The missing work is the actual Tinker Cookbook adapter, a policy-safe renderer,
corpus admission, async/process isolation, and the L2 tool sandbox.

## Current blockers

- `ELT-taskgen` has no committed `HEAD` and its active worktree contains many
  uncommitted files. Production training needs a reviewed wheel/source digest.
- The verified schema-3 release inspected here is a synthetic runtime canary,
  useful for interface tests but not suitable training content.
- The current rich generated task inspected here is not an accepted frozen
  release. A fresh schema-current corpus with family-disjoint splits is needed.
- Tinker is not installed in this workspace, so no integration has yet been
  exercised against a real service client.

See [`docs/BUILD_PLAN.md`](docs/BUILD_PLAN.md) for the ordered path through
those blockers.
