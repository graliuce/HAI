# Human-AI Interaction: Bayesian Preference Learning in Gridworld

A research codebase for studying how robots can learn human preferences through observation and natural language queries in a multi-agent gridworld environment.

## Overview

This project implements a gridworld environment where a **human** and a **robot** agent collect objects with various properties. The human has hidden preferences over object properties, and the robot must infer these preferences to maximize reward.

Key features:
- **Bayesian Belief Modeling**: Robot maintains Gaussian beliefs over feature preference weights
- **Active Learning**: Robot can query the human using natural language (via LLM) to learn preferences faster
- **Multiple Query Modes**: Including Expected Information Gain (EIG), state-based, and Thompson sampling-based queries
- **Additive Valuation**: Object rewards are computed as sums of their property value rewards
- **Rich Visualizations**: Generate GIFs of episodes and performance plots

## Installation

### Requirements

- Python 3.8+
- pip

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd HAI
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) For query-enabled experiments, set up OpenAI API key:
```bash
# Create a .env file
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

Or set as environment variable:
```bash
export OPENAI_API_KEY=your-api-key-here
```

## Quick Start

### Basic Experiment (No Queries)

Run a simple experiment without queries:

```bash
python run_experiment.py --grid-size 10 --num-objects 20 --eval-episodes 10
```

### With Natural Language Queries

Enable queries with different modes:

```bash
# Default mode: Thompson sampling-based queries
python run_experiment.py --allow-queries --query-budget 5

# Expected Information Gain (EIG) mode
python run_experiment.py --allow-queries --query-budget 5 --ask-eig

# State-based queries
python run_experiment.py --allow-queries --query-budget 5 --ask-with-state

# Belief-based queries
python run_experiment.py --allow-queries --query-budget 5 --ask-with-beliefs

# Preference queries (pairwise comparisons)
python run_experiment.py --allow-queries --query-budget 5 --ask-preference-with-state
```

### Generate GIFs Only

Generate visualizations without running full experiments:

```bash
python run_experiment.py --gen-gifs --output-dir results_gifs
```

## Architecture

### Environment (`gridworld/environment.py`)

The `GridWorld` class implements a 2D grid where:
- Objects have 5 property categories: **color**, **shape**, **size**, **pattern**, **opacity**
- Each property can have 1-5 values (configurable)
- Each property value has a Gaussian-sampled reward
- Object rewards are additive: sum of all property value rewards
- The human knows the true rewards and acts optimally
- The robot must infer preferences from observations

### Agents

#### Human Agent (`gridworld/agents/human.py`)
- Has access to true property value rewards
- Uses greedy policy to collect highest-reward objects
- Responds to robot's natural language queries

#### Belief-Based Robot (`gridworld/agents/belief_based_robot.py`)
- Maintains Gaussian belief over feature preference weights: N(μ, Σ)
- Prior: N(0, I)
- Updates beliefs via:
  - **Plackett-Luce model**: From observing human's object collection choices
  - **Linear-Gaussian updates**: From natural language query responses
- Decides when to query based on action confidence threshold
- Multiple query modes:
  - **Thompson Sampling**: Sample weights, ask about most uncertain features
  - **EIG**: Select queries that maximize expected information gain
  - **State-based**: Query about specific objects on the board
  - **Preference**: Ask pairwise comparison questions

### LLM Interface (`gridworld/llm_interface.py`)

Handles natural language query generation and interpretation:
- Generates contextual queries based on current beliefs and game state
- Interprets human responses to extract preference information
- Supports OpenAI API integration

### Experiment Runner (`gridworld/experiment.py`)

Orchestrates experiments:
- Runs multiple episodes with varying property counts
- Collects statistics (rewards, queries, information gain)
- Supports multi-seed averaging
- Generates episode results and summaries

## Configuration Options

### Environment Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--grid-size` | 10 | Size of the square grid |
| `--num-objects` | 20 | Number of objects in the environment |
| `--num-property-values` | 5 | Number of values per property (1-5) |

### Evaluation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--eval-episodes` | 10 | Number of evaluation episodes |
| `--property-counts` | "1,2,3,4,5" | Property counts to test |
| `--num-seeds` | 1 | Number of random seeds to average |

### Query Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--allow-queries` | False | Enable natural language queries |
| `--query-budget` | 5 | Maximum queries per episode |
| `--llm-model` | "gpt-5-chat-latest" | LLM model to use |
| `--action-confidence-threshold` | 0.3 | Query when confidence < this |

### Belief Update Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--plackett-luce-learning-rate` | 0.2 | Learning rate for Plackett-Luce updates |
| `--plackett-luce-gradient-steps` | 5 | Gradient steps per update |
| `--linear-gaussian-noise-variance` | 0.5 | Noise variance for query updates |

### EIG Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--eig-mc-samples` | 100 | Monte Carlo samples for EIG estimation |

### Output Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--output-dir` | "results" | Directory for results |
| `--no-plot` | False | Disable plotting |
| `--verbose` | False | Print detailed progress |

## Query Modes

### 1. Sampled Actions (Default)
Uses Thompson sampling to measure action confidence. Queries include vote frequencies over features.

### 2. Expected Information Gain (--ask-eig)
Selects queries that maximize expected information gain using Bayesian Optimal Experimental Design (BOED).

### 3. State-Based (--ask-with-state)
Queries include information about objects on the board with properties and distances.

### 4. Beliefs-Based (--ask-with-beliefs)
Queries include robot's current belief mean and variance for each feature.

### 5. Preference (--ask-preference-with-state)
Asks pairwise comparison questions between two objects.


## Example Commands

### Full Experiment with EIG

```bash
python run_experiment.py --allow-queries --query-budget 5 --eval-episodes 30 --action-confidence-threshold 0.3 --output-dir results_preference_state --plackett-luce-learning-rate 0.05 --linear-gaussian-noise-variance 2 --ask-preference-with-state > run_experiment.out
```

## Project Structure

```
HAI/
├── gridworld/                      # Main package
│   ├── __init__.py
│   ├── environment.py              # GridWorld environment
│   ├── experiment.py               # Experiment orchestration
│   ├── llm_interface.py            # LLM query generation/interpretation
│   ├── objects.py                  # Object definitions and properties
│   └── agents/                     # Agent implementations
│       ├── __init__.py
│       ├── human.py                # Human agent (optimal policy)
│       └── belief_based_robot.py   # Bayesian belief-based robot
├── run_experiment.py               # Main entry point
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── results/                        # Output directory (created at runtime)
```

## Key Algorithms

### Plackett-Luce Belief Update

When the human collects objects, the robot updates beliefs using gradient ascent on the Plackett-Luce likelihood:

```
P(o₁ > o₂ > ... > oₖ | w) ∝ ∏ᵢ exp(wᵀφ(oᵢ)) / ∑ⱼ≥ᵢ exp(wᵀφ(oⱼ))
```

### Expected Information Gain (EIG)

For query selection, compute expected information gain:

```
EIG(q) = H[p(θ)] - E_{y~p(y|q)}[H[p(θ|y,q)]]
```

where:
- `θ`: preference weights
- `q`: query (pairwise comparison)
- `y`: human response

### Action Confidence

Confidence is measured using Thompson sampling:
- Sample N weight vectors from current belief
- For each sample, find best object
- Confidence = fraction agreeing on top choice
- Query if confidence < threshold


