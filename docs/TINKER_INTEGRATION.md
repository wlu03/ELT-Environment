# Tinker and GRPO integration contract

This records the current API assumptions reviewed on 2026-09-06. Re-audit them
when changing either pin.

## Pins and runtime

The scaffold pins:

```text
Python             3.12.x
tinker             0.27.1
tinker-cookbook    0.5.7
```

The Tinker service performs model sampling and training remotely. The local
Python process still owns the dataset, environment, tools, sandbox, grader
coordination, and logs. Installing Tinker does not isolate model-authored code.

References:

- [Tinker package](https://pypi.org/project/tinker/)
- [Tinker Cookbook package](https://pypi.org/project/tinker-cookbook/)
- [Cookbook Env API](https://tinker-docs.thinkingmachines.ai/cookbook/api-reference/rl/env/)
- [EnvGroupBuilder API](https://tinker-docs.thinkingmachines.ai/cookbook/api-reference/rl/envgroupbuilder/)
- [Env and EnvGroupBuilder tutorial](https://tinker-docs.thinkingmachines.ai/tutorials/cookbook-abstractions/env-and-envgroupbuilder/)
- [Harbor RL reference recipe](https://tinker-docs.thinkingmachines.ai/cookbook/recipes/harbor-rl/)

## Objects to implement

### `EltEnv`

A single-use asynchronous Cookbook `Env`:

```text
initial_observation() -> (ModelInput, StopCondition) | InitialObservationOverflow
step(action, extra=None) -> StepResult
```

For L1, `step` is terminal. For L2, each tool result becomes the next
observation and only `submit`/`abort` is terminal. Every terminal
`StepResult.reward` is a real float. A taskgen no-label never reaches this
dataclass.

### `EltEnvGroupBuilder`

A pickleable builder containing only immutable task/config references:

```text
make_envs()              create a complete same-task group lazily
compute_group_rewards()  optional; not needed for independent ELT labels
cleanup()                idempotent and cancellation safe
logging_tags()           public, low-cardinality tags only
```

`make_envs()` must work again after cleanup because a whole group can be
retried. Every call returns fresh sandbox, attempt, database, and runtime IDs.

### `EltRLDataset` and builder

The dataset's `get_batch(index)` selects parent tasks and returns builders;
`__len__` is deterministic for a fixed admitted catalog. The dataset builder
creates family-disjoint train and evaluation datasets. Task members are never
mixed within a group.

### Coordinator/trainer entrypoint

The first trainer uses synchronous on-policy Cookbook training, explicit
rollout strategy settings, bounded rollout concurrency, and an owned checkpoint
manifest. Inference-only rollout must pass before the first optimizer update.

## No-label is control flow

Cookbook reward and trajectory fields are typed as floats. Taskgen's
`reward=None` means the attempt could not be measured; it is neither failure
reward nor a float to smuggle through as NaN.

The grader adapter returns one of:

```python
ValidReward(value=0.75, metrics={"r_el": 1.0, "r_t": 0.5})
DiscardGroup(reason_code="grader_timeout")
```

The environment calls `require_tinker_reward()` before constructing
`StepResult`. `DiscardGroup` raises `DiscardGroupSignal` and begins complete
group control flow.

## Why Cookbook defaults are insufficient

- `RetryOnFailure` replaces only an individual failed trajectory and retains
  successful siblings. That splices samples from different group attempts.
- `MinViableGroup` may train on a partial group after roughly 75% of siblings
  succeed.
- A caught group error becomes a missing group, and the standard synchronous,
  asynchronous, and streaming trainers can continue with a smaller batch
  rather than replenishing it.
- Constant-reward filtering can also silently shrink a batch.

All of those conflict with taskgen's contract: if any sibling is unlabelled,
discard every sibling and resample the same parent task from fresh state.

## Required `RetryWholeGroup` strategy

Implement a custom pickleable Cookbook `RolloutStrategy` with this behavior:

```text
for attempt in 1..max_attempts:
    envs = builder.make_envs()              # exactly group_size, all fresh
    launch every single rollout
    if any rollout raises DiscardGroupSignal:
        cancel and await every sibling
        retain no trajectory from this attempt
        builder.cleanup()
        continue
    if any unrelated exception occurs:
        cancel/await siblings, clean up, fail loudly
    return the complete, labelled group
raise GroupRetryExhausted
```

Important details:

- set `catches_group_errors = False`;
- use a distinct exhaustion exception, not Cookbook's
  `AllTrajectoriesFailedError`, which is caught and converted into a missing
  group;
- await cancelled tasks before tearing down their resources;
- verify that every successful result contains a finite reward in `[0,1]`;
- never splice a previously successful sibling into a retry;
- use the same task builder on retry, unless an owned outer trainer is
  explicitly responsible for resampling the complete task/group operation.

The custom strategy is sufficient when no-label is discovered inside terminal
`Env.step` and reward is per trajectory. If no-label is discovered only in
`compute_group_rewards`, wrap and retry the complete `do_group_rollout`
operation in an owned trainer layer; Cookbook has no general refill callback at
that later point.

Configure rollout handling explicitly. Do not use the boolean convenience
values whose meaning varies by preset/model. For the first run:

```text
rollout strategy                RetryWholeGroup(max_attempts=4)
remove_constant_reward_groups  false
training mode                  synchronous
```

If constant groups are later filtered, the owned trainer must refill them to
preserve the configured batch shape and record the filter rate.

## Token and action handling

Use the native Tinker sampling path used by Cookbook RL, retaining sampled
tokens and sampling log probabilities for training. Decode action content once
with the same renderer used in preflight. The environment must not repair JSON,
extract code fences, or ask a second model to reinterpret an action.

Preflight every admitted task with:

- selected model and tokenizer;
- exact chat renderer/system prompt;
- initial observation and stop conditions;
- max context and output tokens;
- a known-correct complete response for L1, or worst-case bounded tool
  observations for L2.

`InitialObservationOverflow` quarantines the task/configuration. It is not a
negative rollout label.

## Async and cleanup rules

- Run blocking grader work in a bounded worker, not the event loop.
- Make cleanup idempotent and safe after partial initialization.
- Propagate cancellation into process-group termination and sandbox teardown.
- Bound model sampling, active groups, dbt workers, and hidden replays
  independently so nested fan-out cannot exhaust the host.
- Use multiprocessing `spawn` or `forkserver`, never default `fork` after a
  Tinker client has background threads.
- Builders must be pickleable; instantiate clients and heavy taskgen package
  state lazily in their execution process.

## Checkpoint/resume

Sampler and optimizer/training checkpoints are distinct. Save both, together
with local dataset cursor/RNG and the environment manifest described in
`ARCHITECTURE.md`. A model-only checkpoint is suitable for inference but not a
bitwise/semantic training resume.

## Contract tests

The Tinker adapter is not complete until tests prove:

1. group members share one task ID and have unique attempt/sandbox IDs;
2. one no-label cancels all siblings and trains on none of them;
3. a retry creates exactly `group_size` entirely fresh members;
4. retry exhaustion fails loudly and cannot shrink a batch;
5. numeric zero remains a valid policy label and does not retry;
6. cleanup runs once-or-more safely after success, failure, timeout, and
   cancellation;
7. a builder survives pickle/spawn and creates heavy state lazily;
8. context overflow excludes a task without yielding reward;
9. checkpoint/resume restores sampler, optimizer, dataset order, and task/env
   version bindings.

