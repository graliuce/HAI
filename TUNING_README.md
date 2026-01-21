# Hyperparameter Tuning Guide

This guide explains how to use the `tune_hyperparameters.py` script to optimize hyperparameters for different experimental settings.

## Overview

The script supports tuning two different hyperparameters:

1. **`plackett-luce-learning-rate`**: For additive valuations **without** queries
   - Controls the learning rate for Plackett-Luce belief updates in the belief-based agent
   - Default: 0.1

2. **`linear-gaussian-noise-variance`**: For additive valuations **with** queries
   - Controls noise variance for linear-Gaussian updates from queries
   - Default: 1.0

## Basic Usage

### Tune plackett-luce-learning-rate (without queries)

```bash
python tune_hyperparameters.py \
    --mode plackett-luce-lr \
    --plackett-luce-values "0.01,0.05,0.1,0.2,0.5"
```

### Tune linear-gaussian-noise-variance (with queries)

```bash
python tune_hyperparameters.py \
    --mode linear-gaussian-noise \
    --linear-gaussian-values "0.1,0.5,1.0,2.0,5.0"
```

### Tune both hyperparameters

```bash
python tune_hyperparameters.py \
    --mode both \
    --plackett-luce-values "0.01,0.05,0.1,0.2,0.5" \
    --linear-gaussian-values "0.1,0.5,1.0,2.0,5.0"
```

## Advanced Options

### Custom experiment settings

```bash
python tune_hyperparameters.py \
    --mode plackett-luce-lr \
    --plackett-luce-values "0.01,0.05,0.1,0.2,0.5" \
    --grid-size 10 \
    --num-objects 30 \
    --train-episodes 2000 \
    --eval-episodes 20 \
    --num-seeds 3 \
    --property-counts "1,2,3,4,5"
```

### With verbose output

```bash
python tune_hyperparameters.py \
    --mode plackett-luce-lr \
    --plackett-luce-values "0.01,0.05,0.1,0.2,0.5" \
    --verbose
```

### Custom output directory

```bash
python tune_hyperparameters.py \
    --mode plackett-luce-lr \
    --plackett-luce-values "0.01,0.05,0.1,0.2,0.5" \
    --output-dir my_tuning_results
```

## Quick Start Examples

### Quick test (faster, fewer episodes)

```bash
python tune_hyperparameters.py \
    --mode plackett-luce-lr \
    --plackett-luce-values "0.05,0.1,0.2" \
    --train-episodes 500 \
    --eval-episodes 5 \
    --num-seeds 1
```

### Full tuning run (more thorough, takes longer)

```bash
python tune_hyperparameters.py \
    --mode plackett-luce-lr \
    --plackett-luce-values "0.01,0.02,0.05,0.1,0.15,0.2,0.3,0.5" \
    --train-episodes 2000 \
    --eval-episodes 20 \
    --num-seeds 3
```

## Output

The script creates a timestamped directory structure:

```
tuning_results/
  20260120_143022/  # timestamp
    plackett_luce_lr_no_queries/
      plackett-luce-learning-rate_0.01/
        summary.json
        training_curve.png
        eval_returns_vs_properties.png
        episode_*.gif
      plackett-luce-learning-rate_0.05/
        ...
      tuning_plackett-luce-learning-rate_performance.png
      tuning_plackett-luce-learning-rate_by_props.png
      tuning_summary.json
    linear_gaussian_noise_with_queries/
      linear-gaussian-noise-variance_0.1/
        ...
      tuning_linear-gaussian-noise-variance_performance.png
      tuning_linear-gaussian-noise-variance_queries.png
      tuning_linear-gaussian-noise-variance_by_props.png
      tuning_summary.json
    overall_summary.json
```

### Key files:

- **`tuning_summary.json`**: Summary of all tested values and their performance
- **`tuning_*_performance.png`**: Plot showing performance vs hyperparameter value
- **`tuning_*_by_props.png`**: Performance across different property counts
- **`tuning_*_queries.png`**: Query usage vs hyperparameter (for query-enabled mode)
- **`overall_summary.json`**: Combined results for all tuning runs

## Interpreting Results

The script automatically:
1. Runs experiments for each hyperparameter value
2. Plots performance curves
3. Identifies the best hyperparameter value
4. Saves detailed results for each configuration

Look for the output at the end of the run:

```
================================================================================
BEST plackett-luce-learning-rate: 0.1
Average evaluation return: 8.45
================================================================================
```

## Command-Line Options

### Required:
- `--mode`: Which hyperparameter(s) to tune
  - `plackett-luce-lr`: Tune plackett-luce-learning-rate (no queries)
  - `linear-gaussian-noise`: Tune linear-gaussian-noise-variance (with queries)
  - `both`: Tune both hyperparameters

### Hyperparameter ranges:
- `--plackett-luce-values`: Comma-separated values to test (default: "0.01,0.05,0.1,0.2,0.5")
- `--linear-gaussian-values`: Comma-separated values to test (default: "0.1,0.5,1.0,2.0,5.0")

### Experiment parameters:
- `--grid-size`: Size of the grid (default: 10)
- `--num-objects`: Number of objects in the grid (default: 20)
- `--train-episodes`: Training episodes per run (default: 1000)
- `--eval-episodes`: Evaluation episodes per run (default: 10)
- `--num-seeds`: Number of random seeds to average over (default: 1)
- `--property-counts`: Property counts to test (default: "1,2,3,4,5")
- `--query-budget`: Query budget for query-enabled experiments (default: 5)

### Output parameters:
- `--output-dir`: Base directory for results (default: "tuning_results")
- `--verbose`: Print detailed output from each experiment

## Notes

1. **Time requirements**: Each experiment run can take several minutes to hours depending on:
   - Number of training episodes
   - Number of seeds
   - Number of property counts
   - Number of hyperparameter values to test

2. **Recommended values to test**:
   - `plackett-luce-learning-rate`: Try values in range [0.01, 0.5]
     - Smaller values (0.01-0.05): More stable, slower learning
     - Larger values (0.2-0.5): Faster learning, potentially less stable
   
   - `linear-gaussian-noise-variance`: Try values in range [0.1, 5.0]
     - Smaller values (0.1-0.5): Trust queries more (low noise assumption)
     - Larger values (2.0-5.0): Trust queries less (high noise assumption)

3. **For faster testing**: Reduce `--train-episodes`, `--eval-episodes`, and `--num-seeds`

4. **For more reliable results**: Increase `--num-seeds` to 3 or 5

## Example Workflow

1. **Quick exploration** (test a few values quickly):
```bash
python tune_hyperparameters.py \
    --mode plackett-luce-lr \
    --plackett-luce-values "0.05,0.1,0.2" \
    --train-episodes 500 \
    --eval-episodes 5
```

2. **Fine-tuning** (narrow down around best value):
```bash
# If 0.1 was best from step 1, test around it
python tune_hyperparameters.py \
    --mode plackett-luce-lr \
    --plackett-luce-values "0.08,0.09,0.1,0.11,0.12" \
    --train-episodes 1000 \
    --eval-episodes 10 \
    --num-seeds 2
```

3. **Final validation** (validate with more seeds):
```bash
# Use best value with more seeds for final confirmation
python run_experiment.py \
    --additive-valuation \
    --use-belief-based-agent \
    --plackett-luce-learning-rate 0.1 \
    --train-episodes 2000 \
    --eval-episodes 20 \
    --num-seeds 5
```

