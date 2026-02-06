#!/usr/bin/env python3
"""
Main script to run the gridworld multi-agent experiment.

This experiment investigates how a robot can learn reward specifications by
observing human behavior in a multi-agent gridworld environment.

The robot uses a belief-based approach with Bayesian preference modeling:
- Maintains a Gaussian belief over feature preference weights
- Updates beliefs using Plackett-Luce model from observations
- Can query the human to learn preferences faster through natural language
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
    create_belief_based_agent,
)
from gridworld.agents.belief_based_robot import BeliefBasedRobotAgent


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
        "--num-property-values", type=int, default=5,
        help="Number of values per property category, 1-5 (default: 5)"
    )

    # Evaluation parameters
    parser.add_argument(
        "--eval-episodes", type=int, default=10,
        help="Number of evaluation episodes (default: 10)"
    )

    # Query parameters
    parser.add_argument(
        "--allow-queries", action="store_true",
        help="Allow robot to query human at test time"
    )
    parser.add_argument(
        "--query-budget", type=int, default=5,
        help="Maximum queries per episode (default: 5)"
    )
    parser.add_argument(
        "--llm-model", type=str, default="gpt-5-chat-latest",
        help="LLM model for queries (default: gpt-5-chat-latest)"
    )
    parser.add_argument(
        "--api-key", type=str, default=None,
        help="OpenAI API key (or set OPENAI_API_KEY env var)"
    )

    # Query prompting mode (mutually exclusive)
    query_mode_group = parser.add_mutually_exclusive_group()
    query_mode_group.add_argument(
        "--ask-with-sampled-actions", action="store_true",
        help="Query mode: Include Thompson sampling votes in prompt (default)"
    )
    query_mode_group.add_argument(
        "--ask-with-state", action="store_true",
        help="Query mode: Include objects on board with properties/distance and human-collected objects"
    )
    query_mode_group.add_argument(
        "--ask-with-beliefs", action="store_true",
        help="Query mode: Include unique features with distance to closest object and robot beliefs (mean/variance)"
    )
    query_mode_group.add_argument(
        "--ask-preference-with-state", action="store_true",
        help="Query mode: Ask which object the human prefers between exactly two objects on the board"
    )
    query_mode_group.add_argument(
        "--ask-eig", action="store_true",
        help="Query mode: Use Expected Information Gain (EIG) to select optimal pairwise comparison queries. "
             "Based on Bayesian Optimal Experimental Design (BOED) framework."
    )

    # Belief-based agent parameters
    parser.add_argument(
        "--action-confidence-threshold", type=float, default=0.3,
        help="Query when action confidence < this (0-1). Higher = query more. Uses Thompson "
             "sampling to measure agreement on best object. (default: 0.3 = query when <30%% agree)"
    )
    parser.add_argument(
        "--plackett-luce-learning-rate", type=float, default=0.2,
        help="Learning rate for Plackett-Luce belief updates (default: 0.2)"
    )
    parser.add_argument(
        "--plackett-luce-gradient-steps", type=int, default=5,
        help="Number of gradient steps per Plackett-Luce update (default: 5)"
    )
    parser.add_argument(
        "--linear-gaussian-noise-variance", type=float, default=0.1,
        help="Noise variance for linear-Gaussian updates from queries (default: 0.1)"
    )

    # EIG-specific parameters
    parser.add_argument(
        "--eig-mc-samples", type=int, default=100,
        help="Number of Monte Carlo samples for EIG estimation (default: 100)"
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
        "--gen-gifs", action="store_true",
        help="Only generate GIFs (skip full evaluation)"
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
    """Plot results from variable property experiment."""
    prop_counts = sorted([k for k in summary.keys() if isinstance(k, int)])
    eval_means = [summary[p]['eval_mean'] for p in prop_counts]
    eval_ci_errors = [summary[p]['ci_95_error'] for p in prop_counts]

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(
        prop_counts, eval_means,
        yerr=eval_ci_errors,
        capsize=5,
        color='tab:blue',
        edgecolor='black',
        alpha=0.8
    )

    query_info = f", Query Budget={config.query_budget}" if config.allow_queries else ""
    ax.set_xlabel('Number of Distinct Properties (Test Condition)', fontsize=12)
    ax.set_ylabel('Evaluation Return (Mean with 95% CI)', fontsize=12)
    ax.set_title(
        f'Performance vs. Test Property Count\n'
        f'(Objects={config.num_objects}{query_info})',
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

        # Plot information gain if available: separate query vs observation IG
        avg_query_ig = [summary[p].get('avg_query_information_gain', 0) for p in prop_counts]
        avg_obs_ig = [summary[p].get('avg_observation_information_gain', 0) for p in prop_counts]
        if any(ig > 0 for ig in avg_query_ig) or any(ig > 0 for ig in avg_obs_ig):
            x = np.arange(len(prop_counts))
            width = 0.35

            query_ig_ci = [summary[p].get('query_ig_ci_95_error', 0) for p in prop_counts]
            obs_ig_ci = [summary[p].get('obs_ig_ci_95_error', 0) for p in prop_counts]

            fig, ax = plt.subplots(figsize=(10, 6))
            bars_query = ax.bar(
                x - width / 2,
                avg_query_ig,
                width,
                label='Queries',
                color='tab:green',
                edgecolor='black',
                alpha=0.8,
                yerr=query_ig_ci,
                capsize=5,
            )
            bars_obs = ax.bar(
                x + width / 2,
                avg_obs_ig,
                width,
                label='Observations',
                color='tab:purple',
                edgecolor='black',
                alpha=0.8,
                yerr=obs_ig_ci,
                capsize=5,
            )

            ax.set_xlabel('Number of Distinct Properties', fontsize=12)
            ax.set_ylabel('Average Information Gain per Update (nats)', fontsize=12)
            ax.set_title(f'Information Gain vs. Property Count\n(Mode={config.query_mode})', fontsize=14)
            ax.set_xticks(x)
            ax.set_xticklabels(prop_counts)
            ax.grid(True, alpha=0.3, axis='y')
            ax.legend()

            plt.tight_layout()
            plot_path = os.path.join(output_dir, 'information_gain_vs_properties.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"Information gain plot saved to: {plot_path}")
            plt.close()


def render_episode_gifs(property_counts: list, output_dir: str,
                        config: ExperimentConfig, trained_robot):
    """Render episode GIFs."""
    print("\nRendering episode GIFs...")

    for num_props in property_counts:
        output_path = os.path.join(output_dir, f'episode_{num_props}_props.gif')
        robot_reward = render_episode_gif(
            num_distinct_properties=num_props,
            config=config,
            output_path=output_path,
            max_steps=50,
            fps=4,
            trained_robot=trained_robot
        )
        print(f"  Saved GIF for {num_props} distinct properties: {output_path}")
        print(f"    Final robot reward: {robot_reward:.2f}")


def main():
    """Main entry point."""
    args = parse_args()

    # Load .env file
    load_dotenv()

    # Parse property counts
    property_counts = [int(x) for x in args.property_counts.split(',')]

    # Get API key
    api_key = args.api_key or os.environ.get('OPENAI_API_KEY')

    # Validate query settings
    if args.allow_queries and not api_key:
        print("WARNING: --allow-queries requires OpenAI API key.")
        print("Set via --api-key or OPENAI_API_KEY environment variable.")
        print("Continuing without queries...")
        args.allow_queries = False

    # Determine query mode
    if args.ask_with_state:
        query_mode = "state"
    elif args.ask_with_beliefs:
        query_mode = "beliefs"
    elif args.ask_preference_with_state:
        query_mode = "preference"
    elif args.ask_eig:
        query_mode = "eig"
    else:
        # Default to sampled actions mode
        query_mode = "sampled_actions"

    # Create config
    config = ExperimentConfig(
        grid_size=args.grid_size,
        num_objects=args.num_objects,
        num_property_values=args.num_property_values,
        num_eval_episodes=args.eval_episodes,
        allow_queries=args.allow_queries,
        query_budget=args.query_budget,
        llm_model=args.llm_model,
        query_mode=query_mode,
        action_confidence_threshold=args.action_confidence_threshold,
        plackett_luce_learning_rate=args.plackett_luce_learning_rate,
        plackett_luce_gradient_steps=args.plackett_luce_gradient_steps,
        linear_gaussian_noise_variance=args.linear_gaussian_noise_variance,
        eig_num_mc_samples=args.eig_mc_samples,
        seed=args.seed
    )

    query_status = "ENABLED" if config.allow_queries else "DISABLED"

    print("=" * 60)
    print("Gridworld Experiment (Belief-Based Agent)")
    print("=" * 60)
    print("\nConfiguration:")
    print(f"  Grid size: {config.grid_size}x{config.grid_size}")
    print(f"  Number of objects: {config.num_objects}")
    print(f"  Property values per category: {config.num_property_values}")
    print(f"  Evaluation episodes: {config.num_eval_episodes}")
    print(f"  Property counts to test: {property_counts}")
    print(f"  Number of seeds: {args.num_seeds}")
    print(f"  Base seed: {args.seed}")
    print(f"  Output directory: {args.output_dir}")
    print(f"\n  === AGENT SETTINGS ===")
    print(f"  Action confidence threshold: {config.action_confidence_threshold}")
    print(f"  Plackett-Luce learning rate: {config.plackett_luce_learning_rate}")
    print(f"  Plackett-Luce gradient steps: {config.plackett_luce_gradient_steps}")
    print(f"  Linear-Gaussian noise variance: {config.linear_gaussian_noise_variance}")
    print(f"\n  === QUERY SETTINGS ===")
    print(f"  Queries: {query_status}")
    if config.allow_queries:
        print(f"  Query budget: {config.query_budget}")
        print(f"  Query mode: {config.query_mode}")
        print(f"  LLM model: {config.llm_model}")
        if config.query_mode == "eig":
            print(f"\n  === EIG SETTINGS ===")
            print(f"  EIG Monte Carlo samples: {config.eig_num_mc_samples}")
            print(f"  EIG features per option: One per active property category")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)

    # Handle --gen-gifs mode (just generate GIFs, skip full experiment)
    if args.gen_gifs:
        print("\n" + "=" * 60)
        print("GIF Generation Mode")
        print("=" * 60)

        from gridworld.environment import GridWorld

        # Create environment
        env = GridWorld(
            grid_size=config.grid_size,
            num_objects=config.num_objects,
            num_distinct_properties=5,
            num_property_values=config.num_property_values,
            seed=config.seed
        )

        # Create robot
        print("Setting up belief-based robot for GIF generation...")
        robot = create_belief_based_agent(config, env, api_key, verbose=True)
        print("Verbose mode enabled for belief tracking during GIF generation.")

        # Render GIFs
        render_episode_gifs(property_counts, args.output_dir, config, robot)

        print("\nGIF generation complete!")
        return

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
    print("(Belief-Based Agent)")
    print("=" * 60)

    prop_counts_only = sorted([k for k in summary.keys() if isinstance(k, int)])

    if config.allow_queries:
        # Check if we have information gain data
        has_ig_data = any(summary[p].get('avg_query_information_gain', 0) > 0 or
                          summary[p].get('avg_observation_information_gain', 0) > 0
                          for p in prop_counts_only)
        
        if has_ig_data:
            print(f"\n{'Props':<8} {'Eval Mean':<12} {'95% CI':<20} "
                  f"{'Avg Queries':<15} {'Avg Query IG':<15} {'Avg Obs IG':<15}")
            print("-" * 110)
            for num_props in prop_counts_only:
                s = summary[num_props]
                ci_str = f"[{s['ci_95_lower']:.2f}, {s['ci_95_upper']:.2f}]"
                print(f"{num_props:<8} {s['eval_mean']:<12.2f} {ci_str:<20} "
                      f"{s.get('avg_queries', 0):<15.2f} "
                      f"{s.get('avg_query_information_gain', 0):<15.4f} "
                      f"{s.get('avg_observation_information_gain', 0):<15.4f}")
            
            # Print information gain summary
            print("\n" + "-" * 60)
            print("INFORMATION GAIN SUMMARY")
            print("-" * 60)
            all_avg_query_ig = [summary[p].get('avg_query_information_gain', 0) for p in prop_counts_only]
            all_avg_obs_ig = [summary[p].get('avg_observation_information_gain', 0) for p in prop_counts_only]
            print(f"Average query IG per update (across property counts): {np.mean(all_avg_query_ig):.4f} nats")
            print(f"Average observation IG per update (across property counts): {np.mean(all_avg_obs_ig):.4f} nats")
            print(f"Query mode: {config.query_mode}")
        else:
            print(f"\n{'Props':<8} {'Eval Mean':<12} {'95% CI':<20} {'Avg Queries':<15}")
            print("-" * 60)
            for num_props in prop_counts_only:
                s = summary[num_props]
                ci_str = f"[{s['ci_95_lower']:.2f}, {s['ci_95_upper']:.2f}]"
                print(f"{num_props:<8} {s['eval_mean']:<12.2f} {ci_str:<20} {s.get('avg_queries', 0):<15.2f}")

        # Print additional query statistics
        print("\n" + "-" * 60)
        print("QUERY STATISTICS PER PROPERTY COUNT")
        print("-" * 60)
        total_queries = sum(s.get('avg_queries', 0) for s in [summary[p] for p in prop_counts_only])
        avg_overall = total_queries / len(prop_counts_only) if prop_counts_only else 0
        print(f"Average queries per episode (across all property counts): {avg_overall:.2f}")
        print(f"Query budget per episode: {config.query_budget}")
    else:
        print(f"\n{'Props':<8} {'Eval Mean':<12} {'95% CI':<20} {'n':<8}")
        print("-" * 52)
        for num_props in prop_counts_only:
            s = summary[num_props]
            ci_str = f"[{s['ci_95_lower']:.2f}, {s['ci_95_upper']:.2f}]"
            print(f"{num_props:<8} {s['eval_mean']:<12.2f} {ci_str:<20} {s['n_samples']:<8}")

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
        'num_property_values': config.num_property_values,
        'num_eval_episodes': config.num_eval_episodes,
        'allow_queries': config.allow_queries,
        'query_budget': config.query_budget,
        'query_mode': config.query_mode,
        'llm_model': config.llm_model,
        'num_seeds': args.num_seeds,
        'base_seed': args.seed,
        'action_confidence_threshold': config.action_confidence_threshold,
        'plackett_luce_learning_rate': config.plackett_luce_learning_rate,
        'plackett_luce_gradient_steps': config.plackett_luce_gradient_steps,
        'linear_gaussian_noise_variance': config.linear_gaussian_noise_variance,
        'eig_num_mc_samples': config.eig_num_mc_samples,
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
            render_episode_gifs(property_counts, args.output_dir, config, trained_robot)
        except Exception as e:
            print(f"Warning: Could not create plots: {e}")

    print("\nExperiment complete!")


if __name__ == "__main__":
    main()