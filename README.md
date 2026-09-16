# ELT-Environment

Tinker/GRPO orchestration for the two-stage ELT tasks produced by
[`ELT-taskgen`](../ELT-taskgen).

> **Status:** architecture and safety scaffold. The task/reward audit is
> complete; the Tinker adapter and interactive sandbox are not implemented
> yet. This directory must not be described as a runnable trainer until the
> pilot exit criteria in [`docs/BUILD_PLAN.md`](docs/BUILD_PLAN.md) pass.

## Core contract

One rollout is one parent task and one continuous attempt:
1. public task
2. Extract/Load
3. raw tables
4. Transform
5. marts
6. hidden replay

`<task_id>__el` and `<task_id>__t` are not separate public tasks, dataset rows, episodes, or GRPO groups.

## Repository

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

