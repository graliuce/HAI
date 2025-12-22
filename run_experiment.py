#!/usr/bin/env python3
"""
Main script to run the gridworld multi-agent experiment.

This experiment investigates how the number of distinct object properties
affects the robot's ability to infer reward specifications from human behavior.

Hypothesis: As the number of distinct properties increases, the robot's
evaluation returns will decrease because it becomes harder to infer which
properties are rewarding based on the human's collected items.
"""

import argparse
import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from gridworld.experiment import (
    ExperimentConfig,
    run_property_variation_experiment,
    summarize_results,
    render_episode_gif
)


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
        "--reward-ratio", type=float, default=0.4,
        help="Proportion of objects that are rewarding (default: 0.4)"
    )
    parser.add_argument(
        "--num-rewarding-properties", type=int, default=2,
        help="Number of properties that give reward (K) (default: 2)"
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
        "--learning-rate", type=float, default=1e-4,
        help="Learning rate (default: 1e-4)"
    )
    parser.add_argument(
        "--buffer-size", type=int, default=100000,
        help="Size of replay buffer (default: 100000)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Batch size for training (default: 64)"
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

    # Experiment parameters
    parser.add_argument(
        "--num-seeds", type=int, default=5,
        help="Number of random seeds to average over (default: 5)"
    )
    parser.add_argument(
        "--property-counts", type=str, default="1,2,3,4,5",
        help="Comma-separated list of distinct property counts to test (default: 1,2,3,4,5)"
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
    """
    Plot the experiment results.

    Creates plots showing evaluation returns and training returns
    vs. number of distinct properties.
    """
    prop_counts = sorted(summary.keys())
    eval_means = [summary[p]['eval_mean'] for p in prop_counts]
    eval_sems = [summary[p]['eval_sem'] for p in prop_counts]
    train_means = [summary[p]['train_mean'] for p in prop_counts if 'train_mean' in summary[p]]
    train_sems = [summary[p]['train_sem'] for p in prop_counts if 'train_sem' in summary[p]]

    # Plot evaluation results
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(
        prop_counts, eval_means, yerr=eval_sems,
        marker='o', markersize=10, linewidth=2, capsize=5,
        color='tab:blue', ecolor='tab:blue', elinewidth=2,
        label='Evaluation'
    )
    ax.set_xlabel('Number of Distinct Properties', fontsize=12)
    ax.set_ylabel('Evaluation Return (Mean +/- SEM)', fontsize=12)
    ax.set_title(
        f'Robot Performance vs. Property Complexity\n'
        f'(K={config.num_rewarding_properties}, '
        f'Objects={config.num_objects}, '
        f'Ratio={config.reward_ratio})',
        fontsize=14
    )
    ax.set_xticks(prop_counts)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'eval_returns_vs_properties.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {plot_path}")
    plt.close()

    # Plot training results (mean +/- sem)
    if train_means and train_sems:
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.errorbar(
            prop_counts, train_means, yerr=train_sems,
            marker='o', markersize=10, linewidth=2, capsize=5,
            color='tab:green', ecolor='tab:green', elinewidth=2,
            label='Training'
        )
        ax2.set_xlabel('Number of Distinct Properties', fontsize=12)
        ax2.set_ylabel('Training Return (Mean +/- SEM)', fontsize=12)
        ax2.set_title(
            f'Training Performance vs. Property Complexity\n'
            f'(K={config.num_rewarding_properties}, '
            f'Objects={config.num_objects}, '
            f'Ratio={config.reward_ratio})',
            fontsize=14
        )
        ax2.set_xticks(prop_counts)
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        train_plot_path = os.path.join(output_dir, 'train_returns_vs_properties.png')
        plt.savefig(train_plot_path, dpi=150, bbox_inches='tight')
        print(f"Train results plot saved to: {train_plot_path}")
        plt.close()


def render_episode_gifs(
    property_counts: list,
    output_dir: str,
    config: ExperimentConfig
):
    """
    Render and save episode GIFs for each property count.

    Args:
        property_counts: List of distinct property counts
        output_dir: Directory to save GIFs
        config: Experiment configuration
    """
    print("\nRendering episode GIFs...")

    for num_props in property_counts:
        output_path = os.path.join(
            output_dir,
            f'episode_{num_props}_props.gif'
        )
        render_episode_gif(
            num_distinct_properties=num_props,
            config=config,
            output_path=output_path,
            max_steps=50,
            fps=4
        )
        print(f"  Saved GIF for {num_props} distinct properties: {output_path}")


def plot_training_curves(
    results: dict,
    output_dir: str
):
    """
    Plot training curves for each property count.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))

    for (num_props, result_list), color in zip(sorted(results.items()), colors):
        # Average training curves across seeds
        all_curves = [r.train_rewards for r in result_list]
        min_len = min(len(c) for c in all_curves)
        truncated = [c[:min_len] for c in all_curves]
        mean_curve = np.mean(truncated, axis=0)

        # Smooth with rolling average
        window = 50
        if len(mean_curve) >= window:
            smoothed = np.convolve(
                mean_curve,
                np.ones(window) / window,
                mode='valid'
            )
            x = np.arange(window - 1, len(mean_curve))
            ax.plot(x, smoothed, label=f'{num_props} properties', color=color)

    ax.set_xlabel('Training Episode', fontsize=12)
    ax.set_ylabel('Episode Return (Smoothed)', fontsize=12)
    ax.set_title('Training Curves by Property Complexity', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    plot_path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Training curves saved to: {plot_path}")

    plt.close()


def main():
    """Main entry point."""
    args = parse_args()

    # Parse property counts
    property_counts = [int(x) for x in args.property_counts.split(',')]

    # Parse hidden dimensions
    hidden_dims = [int(x) for x in args.hidden_dims.split(',')]

    # Create config
    config = ExperimentConfig(
        grid_size=args.grid_size,
        num_objects=args.num_objects,
        reward_ratio=args.reward_ratio,
        num_rewarding_properties=args.num_rewarding_properties,
        num_train_episodes=args.train_episodes,
        num_eval_episodes=args.eval_episodes,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        target_update_freq=args.target_update_freq,
        learning_starts=args.learning_starts,
        hidden_dims=hidden_dims,
        seed=args.seed
    )

    print("=" * 60)
    print("Gridworld Multi-Agent Experiment (DQN)")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Grid size: {config.grid_size}x{config.grid_size}")
    print(f"  Number of objects: {config.num_objects}")
    print(f"  Reward ratio: {config.reward_ratio}")
    print(f"  Rewarding properties (K): {config.num_rewarding_properties}")
    print(f"  Training episodes: {config.num_train_episodes}")
    print(f"  Evaluation episodes: {config.num_eval_episodes}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Buffer size: {config.buffer_size}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Target update freq: {config.target_update_freq}")
    print(f"  Learning starts: {config.learning_starts}")
    print(f"  Hidden dims: {config.hidden_dims}")
    print(f"  Property counts to test: {property_counts}")
    print(f"  Number of seeds: {args.num_seeds}")
    print(f"  Base seed: {args.seed}")

    # Run experiment
    print("\n" + "=" * 60)
    print("Running experiments...")
    print("=" * 60)

    results = run_property_variation_experiment(
        config=config,
        property_counts=property_counts,
        num_seeds=args.num_seeds,
        verbose=args.verbose
    )

    # Summarize results
    summary = summarize_results(results)

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"\n{'Props':<8} {'Eval Mean':<12} {'Eval Std':<12} {'Train Mean':<12}")
    print("-" * 48)

    for num_props in sorted(summary.keys()):
        s = summary[num_props]
        print(f"{num_props:<8} {s['eval_mean']:<12.2f} {s['eval_std']:<12.2f} {s['train_mean']:<12.2f}")

    # Calculate and report trend
    prop_counts_arr = np.array(sorted(summary.keys()))
    eval_means_arr = np.array([summary[p]['eval_mean'] for p in prop_counts_arr])

    correlation = np.corrcoef(prop_counts_arr, eval_means_arr)[0, 1]
    print(f"\nCorrelation between property count and eval return: {correlation:.3f}")

    if correlation < -0.5:
        print("HYPOTHESIS SUPPORTED: Negative correlation observed!")
        print("  As distinct properties increase, robot performance decreases.")
    elif correlation < 0:
        print("WEAK SUPPORT: Slight negative correlation observed.")
    else:
        print("HYPOTHESIS NOT SUPPORTED: No negative correlation observed.")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)

    # Save summary as JSON
    summary_path = os.path.join(args.output_dir, 'summary.json')
    config_dict = {
        'grid_size': config.grid_size,
        'num_objects': config.num_objects,
        'reward_ratio': config.reward_ratio,
        'num_rewarding_properties': config.num_rewarding_properties,
        'num_train_episodes': config.num_train_episodes,
        'num_eval_episodes': config.num_eval_episodes,
        'learning_rate': config.learning_rate,
        'buffer_size': config.buffer_size,
        'batch_size': config.batch_size,
        'target_update_freq': config.target_update_freq,
        'learning_starts': config.learning_starts,
        'hidden_dims': config.hidden_dims,
        'num_seeds': args.num_seeds,
        'base_seed': args.seed
    }
    with open(summary_path, 'w') as f:
        json.dump({
            'config': config_dict,
            'summary': summary,
            'correlation': correlation,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")

    # Plot results
    if not args.no_plot:
        try:
            plot_results(summary, args.output_dir, config)
            plot_training_curves(results, args.output_dir)
            render_episode_gifs(property_counts, args.output_dir, config)
        except Exception as e:
            print(f"Warning: Could not create plots: {e}")

    print("\nExperiment complete!")


if __name__ == "__main__":
    main()
