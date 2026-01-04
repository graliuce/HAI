#!/usr/bin/env python3
"""
Main script to run the gridworld multi-agent experiment.

This experiment investigates how the number of distinct object properties
affects the robot's ability to infer reward specifications from human behavior.

The robot is trained with variable property counts (1-5) each episode,
then evaluated separately on each property count to measure generalization.

With --allow_queries flag, the robot can query the human at test time
to learn preferences faster through natural language.
"""

import argparse
import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from gridworld.experiment import (
    ExperimentConfig,
    run_variable_property_experiment,
    summarize_variable_results,
    render_episode_gif,
)


def load_dotenv(path=".env"):
    """Load environment variables from .env file."""
    try:
        with open(path) as f:
            for line in f:
                if line.strip() and not line.strip().startswith("#"):
                    if "=" in line:
                        k, v = line.strip().split("=", 1)
                        os.environ.setdefault(k, v)
    except FileNotFoundError:
        pass


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run gridworld multi-agent experiment"
    )

    # Environment parameters
    parser.add_argument(
        "--grid-size", type=int, default=10,
        help="Size of the grid (default: 10)"
    )
    parser.add_argument(
        "--num-objects", type=int, default=20,
        help="Number of objects in the grid (default: 20)"
    )
    parser.add_argument(
        "--reward-ratio", type=float, default=0.5,
        help="Proportion of objects that are rewarding (default: 0.5)"
    )
    parser.add_argument(
        "--num-rewarding-properties", type=int, default=2,
        help="Number of properties that give reward (K) (default: 2)"
    )
    parser.add_argument(
        "--num-property-values", type=int, default=5,
        help="Number of values per property category, 1-5 (default: 5)"
    )

    # Training parameters
    parser.add_argument(
        "--train-episodes", type=int, default=1000,
        help="Number of training episodes (default: 1000)"
    )
    parser.add_argument(
        "--eval-episodes", type=int, default=10,
        help="Number of evaluation episodes (default: 10)"
    )

    # DQN parameters
    parser.add_argument(
        "--learning-rate", type=float, default=2.5e-4,
        help="Learning rate (default: 2.5e-4)"
    )
    parser.add_argument(
        "--buffer-size", type=int, default=100000,
        help="Size of replay buffer (default: 100000)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=128,
        help="Batch size for training (default: 128)"
    )
    parser.add_argument(
        "--target-update-freq", type=int, default=100,
        help="How often to update target network (default: 100)"
    )
    parser.add_argument(
        "--learning-starts", type=int, default=500,
        help="Number of steps before training starts (default: 500)"
    )
    parser.add_argument(
        "--hidden-dims", type=str, default="128,128",
        help="Comma-separated hidden layer dimensions (default: 128,128)"
    )

    # Hierarchical policy parameters
    parser.add_argument(
        "--hierarchical", action="store_true",
        help="Use hierarchical policy (required for queries)"
    )
    parser.add_argument(
        "--high-level-interval", type=int, default=3,
        help="Steps between high-level goal decisions (default: 3)"
    )

    # Query parameters
    parser.add_argument(
        "--allow-queries", action="store_true",
        help="Allow robot to query human at test time (requires --hierarchical)"
    )
    parser.add_argument(
        "--query-budget", type=int, default=5,
        help="Maximum queries per episode (default: 5)"
    )
    parser.add_argument(
        "--query-threshold", type=float, default=0.5,
        help="Max Q-value threshold for querying - query when best Q-value is below this (default: 0.5)"
    )
    parser.add_argument(
        "--blend-factor", type=float, default=0.5,
        help="Blend factor for beliefs vs Q-values (default: 0.5)"
    )
    parser.add_argument(
        "--llm-model", type=str, default="gpt-4o-mini",
        help="LLM model for queries (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--api-key", type=str, default=None,
        help="OpenAI API key (or set OPENAI_API_KEY env var)"
    )

    # Experiment parameters
    parser.add_argument(
        "--num-seeds", type=int, default=1,
        help="Number of random seeds to average over (default: 1)"
    )
    parser.add_argument(
        "--property-counts", type=str, default="1,2,3,4,5",
        help="Comma-separated list of property counts to test (default: 1,2,3,4,5)"
    )

    # Output parameters
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help="Directory to save results (default: results)"
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Disable plotting"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print detailed progress"
    )

    # Random seed
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Base random seed (default: 42)"
    )

    return parser.parse_args()


def plot_results(
    summary: dict,
    output_dir: str,
    config: ExperimentConfig
):
    """Plot results from variable property training experiment."""
    prop_counts = sorted([k for k in summary.keys() if isinstance(k, int)])
    eval_means = [summary[p]['eval_mean'] for p in prop_counts]
    eval_sems = [summary[p]['eval_sem'] for p in prop_counts]

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(
        prop_counts, eval_means,
        yerr=eval_sems,
        capsize=5,
        color='tab:blue',
        edgecolor='black',
        alpha=0.8
    )

    query_info = f", Query Budget={config.query_budget}" if config.allow_queries else ""
    ax.set_xlabel('Number of Distinct Properties (Test Condition)', fontsize=12)
    ax.set_ylabel('Evaluation Return (Mean +/- SEM)', fontsize=12)
    ax.set_title(
        f'Variable Training: Performance vs. Test Property Count\n'
        f'(K={config.num_rewarding_properties}, Objects={config.num_objects}{query_info})',
        fontsize=14
    )
    ax.set_xticks(prop_counts)
    ax.grid(True, alpha=0.3, axis='y')

    for bar, mean in zip(bars, eval_means):
        height = bar.get_height()
        ax.annotate(f'{mean:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    suffix = "_with_queries" if config.allow_queries else ""
    plot_path = os.path.join(output_dir, f'eval_returns_vs_properties{suffix}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nEvaluation plot saved to: {plot_path}")
    plt.close()

    # Plot queries if enabled
    if config.allow_queries:
        avg_queries = [summary[p].get('avg_queries', 0) for p in prop_counts]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(prop_counts, avg_queries, color='tab:orange', edgecolor='black', alpha=0.8)
        ax.set_xlabel('Number of Distinct Properties', fontsize=12)
        ax.set_ylabel('Average Queries per Episode', fontsize=12)
        ax.set_title(f'Query Usage vs. Property Count\n(Budget={config.query_budget})', fontsize=14)
        ax.set_xticks(prop_counts)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'query_usage_vs_properties.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Query usage plot saved to: {plot_path}")
        plt.close()


def plot_training_curve(results: list, output_dir: str):
    """Plot the training curve."""
    fig, ax = plt.subplots(figsize=(12, 6))

    all_curves = [r.train_rewards for r in results]
    min_len = min(len(c) for c in all_curves)
    truncated = [c[:min_len] for c in all_curves]
    mean_curve = np.mean(truncated, axis=0)
    std_curve = np.std(truncated, axis=0)

    window = 50
    if len(mean_curve) >= window:
        smoothed_mean = np.convolve(mean_curve, np.ones(window) / window, mode='valid')
        smoothed_std = np.convolve(std_curve, np.ones(window) / window, mode='valid')
        x = np.arange(window - 1, len(mean_curve))

        ax.plot(x, smoothed_mean, color='tab:blue', linewidth=2, label='Mean')
        ax.fill_between(x, smoothed_mean - smoothed_std, smoothed_mean + smoothed_std,
                        alpha=0.3, color='tab:blue', label='±1 Std Dev')

    ax.set_xlabel('Training Episode', fontsize=12)
    ax.set_ylabel('Episode Return (Smoothed)', fontsize=12)
    ax.set_title('Training Curve', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'training_curve.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Training curve saved to: {plot_path}")
    plt.close()


def render_episode_gifs(property_counts: list, output_dir: str,
                        config: ExperimentConfig, trained_robot):
    """Render episode GIFs."""
    print("\nRendering episode GIFs...")

    for num_props in property_counts:
        output_path = os.path.join(output_dir, f'episode_{num_props}_props.gif')
        render_episode_gif(
            num_distinct_properties=num_props,
            config=config,
            output_path=output_path,
            max_steps=50,
            fps=4,
            trained_robot=trained_robot
        )
        print(f"  Saved GIF for {num_props} distinct properties: {output_path}")


def main():
    """Main entry point."""
    args = parse_args()

    # Load .env file
    load_dotenv()

    # Parse property counts
    property_counts = [int(x) for x in args.property_counts.split(',')]

    # Parse hidden dimensions
    hidden_dims = [int(x) for x in args.hidden_dims.split(',')]

    # Get API key
    api_key = args.api_key or os.environ.get('OPENAI_API_KEY')
    
    # Validate query settings
    if args.allow_queries:
        if not api_key:
            print("WARNING: --allow-queries requires OpenAI API key.")
            print("Set via --api-key or OPENAI_API_KEY environment variable.")
            print("Continuing without queries...")
            args.allow_queries = False
        else:
            # Force hierarchical for queries
            args.hierarchical = True

    # Create config
    config = ExperimentConfig(
        grid_size=args.grid_size,
        num_objects=args.num_objects,
        reward_ratio=args.reward_ratio,
        num_rewarding_properties=args.num_rewarding_properties,
        num_property_values=args.num_property_values,
        num_train_episodes=args.train_episodes,
        num_eval_episodes=args.eval_episodes,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        target_update_freq=args.target_update_freq,
        learning_starts=args.learning_starts,
        hidden_dims=hidden_dims,
        use_hierarchical=args.hierarchical,
        high_level_interval=args.high_level_interval,
        allow_queries=args.allow_queries,
        query_budget=args.query_budget,
        query_threshold=args.query_threshold,
        blend_factor=args.blend_factor,
        llm_model=args.llm_model,
        seed=args.seed
    )

    policy_type = "Hierarchical" if config.use_hierarchical else "Flat DQN"
    query_status = "ENABLED" if config.allow_queries else "DISABLED"
    
    print("=" * 60)
    print(f"Gridworld Variable Property Training Experiment")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Policy type: {policy_type}")
    print(f"  Grid size: {config.grid_size}x{config.grid_size}")
    print(f"  Number of objects: {config.num_objects}")
    print(f"  Reward ratio: {config.reward_ratio}")
    print(f"  Rewarding properties (K): {config.num_rewarding_properties}")
    print(f"  Property values per category: {config.num_property_values}")
    print(f"  Training episodes: {config.num_train_episodes}")
    print(f"  Evaluation episodes: {config.num_eval_episodes}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Hidden dims: {config.hidden_dims}")
    print(f"  Property counts to test: {property_counts}")
    print(f"  Number of seeds: {args.num_seeds}")
    print(f"  Base seed: {args.seed}")
    print(f"\n  === QUERY SETTINGS ===")
    print(f"  Queries: {query_status}")
    if config.allow_queries:
        print(f"  Query budget: {config.query_budget}")
        print(f"  Query threshold: {config.query_threshold}")
        print(f"  Blend factor: {config.blend_factor}")
        print(f"  LLM model: {config.llm_model}")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)

    # Run experiment
    print("\n" + "=" * 60)
    print("Running experiment...")
    print("=" * 60)

    results, trained_robot = run_variable_property_experiment(
        config=config,
        property_counts=property_counts,
        num_seeds=args.num_seeds,
        verbose=args.verbose,
        api_key=api_key
    )

    # Summarize results
    summary = summarize_variable_results(results, property_counts)

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    if config.allow_queries:
        print(f"\n{'Props':<8} {'Eval Mean':<12} {'Eval Std':<12} {'Avg Queries':<15}")
        print("-" * 50)
        prop_counts_only = sorted([k for k in summary.keys() if isinstance(k, int)])
        for num_props in prop_counts_only:
            s = summary[num_props]
            print(f"{num_props:<8} {s['eval_mean']:<12.2f} {s['eval_std']:<12.2f} {s.get('avg_queries', 0):<15.2f}")
        
        # Print additional query statistics
        print("\n" + "-" * 60)
        print("QUERY STATISTICS PER PROPERTY COUNT")
        print("-" * 60)
        total_queries = sum(s.get('avg_queries', 0) for s in [summary[p] for p in prop_counts_only])
        avg_overall = total_queries / len(prop_counts_only) if prop_counts_only else 0
        print(f"Average queries per episode (across all property counts): {avg_overall:.2f}")
        print(f"Query budget per episode: {config.query_budget}")
    else:
        print(f"\n{'Props':<8} {'Eval Mean':<12} {'Eval Std':<12} {'Eval SEM':<12}")
        print("-" * 48)
        prop_counts_only = sorted([k for k in summary.keys() if isinstance(k, int)])
        for num_props in prop_counts_only:
            s = summary[num_props]
            print(f"{num_props:<8} {s['eval_mean']:<12.2f} {s['eval_std']:<12.2f} {s['eval_sem']:<12.2f}")

    if 'training' in summary:
        print(f"\nOverall training mean (last 100 eps): {summary['training']['train_mean']:.2f}")

    # Correlation
    prop_counts_arr = np.array(prop_counts_only)
    eval_means_arr = np.array([summary[p]['eval_mean'] for p in prop_counts_only])
    correlation = np.corrcoef(prop_counts_arr, eval_means_arr)[0, 1]
    print(f"\nCorrelation between property count and eval return: {correlation:.3f}")

    # Save summary
    summary_path = os.path.join(args.output_dir, 'summary.json')
    config_dict = {
        'grid_size': config.grid_size,
        'num_objects': config.num_objects,
        'reward_ratio': config.reward_ratio,
        'num_rewarding_properties': config.num_rewarding_properties,
        'num_property_values': config.num_property_values,
        'num_train_episodes': config.num_train_episodes,
        'num_eval_episodes': config.num_eval_episodes,
        'learning_rate': config.learning_rate,
        'hidden_dims': config.hidden_dims,
        'use_hierarchical': config.use_hierarchical,
        'allow_queries': config.allow_queries,
        'query_budget': config.query_budget,
        'query_threshold': config.query_threshold,
        'blend_factor': config.blend_factor,
        'llm_model': config.llm_model,
        'num_seeds': args.num_seeds,
        'base_seed': args.seed,
    }

    summary_json = {str(k): v for k, v in summary.items()}

    with open(summary_path, 'w') as f:
        json.dump({
            'config': config_dict,
            'summary': summary_json,
            'correlation': correlation,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")

    # Plot results
    if not args.no_plot:
        try:
            plot_results(summary, args.output_dir, config)
            plot_training_curve(results, args.output_dir)
            if config.allow_queries:
                render_episode_gifs(property_counts, args.output_dir, config, trained_robot)
        except Exception as e:
            print(f"Warning: Could not create plots: {e}")

    print("\nExperiment complete!")


if __name__ == "__main__":
    main()
