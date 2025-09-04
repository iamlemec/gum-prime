# haiku

### Overview

- **Environment ID**: `haiku`
- **Short description**: Write a valid haiku given a user prompt.
- **Tags**: poetry, haiku

### Datasets

- **Primary dataset(s)**: Instruction Poems from [CheckAI](https://huggingface.co/datasets/checkai/instruction-poems)
- **Source links**: https://huggingface.co/datasets/checkai/instruction-poems
- **Split sizes**: Uses the `train` split for evaluation.

### Task

- **Type**: single-turn
- **Parser**: Parse the response into a list of syllable counts for each line.
- **Rubric overview**: (1) Difference of line count from 3, (2) Difference of first three line syllable counts from 5, 7, 5.

### Quickstart

Run an evaluation with default settings:

```bash
uv run vf-eval haiku
```

Configure model and sampling:

```bash
uv run vf-eval haiku -m gpt-4.1-mini -n 20 -r 3 -t 1024 -T 0.7
```

Notes:
- Use `-a` / `--env-args` to pass environment-specific configuration as a JSON object.

### Environment Arguments

Document any supported environment arguments and their meaning. Example:

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `max_examples` | int | `-1` | Limit on dataset size (use -1 for all) |

### Metrics

Summarize key metrics your rubric emits and how they’re interpreted.

| Metric | Meaning |
| ------ | ------- |
| `reward` | Main scalar reward (weighted sum of criteria) |
