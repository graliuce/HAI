#!/usr/bin/env python3
"""
Main script to run the gridworld multi-agent experiment.

This experiment investigates how a robot can learn reward specifications by
observing human behavior in a multi-agent gridworld environment.

The robot uses a hierarchical DQN:
- High-level policy: Selects which property to target (learned via Q-learning)
- Low-level policy: Navigates to nearest object with that property (A* pathfinding)

The robot is trained with variable property counts (1-5) each episode,
then evaluated separately on each property count to measure generalization.

With --allow-queries flag, the robot can query the human at test time
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
    create_belief_based_agent,
)
from gridworld.agents.query_augmented_robot import QueryAugmentedRobotAgent
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
    parser.add_argument(
        "--additive-valuation", action="store_true",
        help="Use additive valuation reward mode: each property value has a Gaussian reward, "
             "object reward is sum of its property value rewards"
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
        "--query-threshold", type=float, default=0.9,
        help="Competitive options threshold for querying - query when num_competitive > threshold (default: 0.9)"
    )
    parser.add_argument(
        "--blend-factor", type=float, default=0.5,
        help="Blend factor for beliefs vs Q-values (default: 0.5)"
    )
    parser.add_argument(
        "--llm-model", type=str, default="gpt-5-chat-latest",
        help="LLM model for queries (default: gpt-5-chat-latest)"
    )
    parser.add_argument(
        "--api-key", type=str, default=None,
        help="OpenAI API key (or set OPENAI_API_KEY env var)"
    )

    # Belief-based agent parameters
    parser.add_argument(
        "--use-belief-based-agent", action="store_true",
        help="Use belief-based agent with Bayesian preference modeling instead of DQN"
    )
    parser.add_argument(
        "--participation-ratio-threshold", type=float, default=3.0,
        help="Trigger query when participation ratio exceeds this (default: 3.0)"
    )
    parser.add_argument(
        "--plackett-luce-learning-rate", type=float, default=0.1,
        help="Learning rate for Plackett-Luce belief updates (default: 0.1)"
    )
    parser.add_argument(
        "--plackett-luce-gradient-steps", type=int, default=5,
        help="Number of gradient steps per Plackett-Luce update (default: 5)"
    )
    parser.add_argument(
        "--plackett-luce-info-gain", type=float, default=0.5,
        help="Covariance reduction rate per observation (0.3=gradual, 0.5=moderate, 0.8=aggressive) (default: 0.5)"
    )
    parser.add_argument(
        "--linear-gaussian-noise-variance", type=float, default=1.0,
        help="Noise variance for linear-Gaussian updates from queries (default: 1.0)"
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
        "--model-dir", type=str, default="models",
        help="Directory to save/load trained models (default: models)"
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Disable plotting"
    )
    parser.add_argument(
        "--gen-gifs", action="store_true",
        help="Only generate GIFs (skip training and full evaluation)"
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


def print_entropy_analysis(results: list, property_counts: list, config: ExperimentConfig):
    """Print detailed competitive options analysis for query triggering."""
    if not config.allow_queries:
        return
    
    print("\n" + "=" * 80)
    print("COMPETITIVE OPTIONS ANALYSIS (Query Triggering Mechanism)")
    print("=" * 80)
    print(f"Query Threshold: {config.query_threshold:.1f}")
    print("\nQuery triggers when num_competitive > threshold")
    print("(num_competitive = count of Q-values within 20% of range from max)\n")
    
    # Collect stats across seeds
    stats_by_prop = {}
    for num_props in property_counts:
        competitive_counts = []
        decision_points = []
        triggered_counts = []
        trigger_rates = []
        
        for result in results:
            if num_props in result.eval_entropy_stats_per_property:
                stats = result.eval_entropy_stats_per_property[num_props]
                competitive_counts.append(stats['mean_num_competitive'])
                decision_points.append(stats['num_decision_points'])
                triggered_counts.append(stats['num_triggered'])
                trigger_rates.append(stats['trigger_rate'])
        
        if competitive_counts:
            stats_by_prop[num_props] = {
                'mean_competitive': np.mean(competitive_counts),
                'std_competitive': np.std(competitive_counts),
                'mean_decision_points': np.mean(decision_points),
                'mean_triggered': np.mean(triggered_counts),
                'mean_trigger_rate': np.mean(trigger_rates)
            }
    
    # Print table header
    print(f"{'Props':<8} {'Mean Competitive':<18} {'Decision Pts':<15} {'Triggered':<15} {'Trigger Rate':<15}")
    print("-" * 85)
    
    for num_props in sorted(stats_by_prop.keys()):
        s = stats_by_prop[num_props]
        print(f"{num_props:<8} "
              f"{s['mean_competitive']:.2f} ± {s['std_competitive']:.2f}       "
              f"{s['mean_decision_points']:<15.1f} "
              f"{s['mean_triggered']:<15.1f} "
              f"{s['mean_trigger_rate']:<15.2%}")
    
    # Print sample Q-values from first seed
    if results and property_counts:
        print("\n" + "-" * 80)
        print("SAMPLE Q-VALUES AT DECISION POINTS (First Seed, First Episode)")
        print("-" * 80)
        
        for num_props in property_counts:
            if num_props in results[0].eval_entropy_stats_per_property:
                stats = results[0].eval_entropy_stats_per_property[num_props]
                sample_decisions = stats.get('sample_decision_points', [])
                
                if sample_decisions:
                    print(f"\n{num_props} Properties:")
                    print(f"  {'Decision':<10} {'Competitive':<15} {'Triggered?':<12} {'Num Actions':<15}")
                    print("  " + "-" * 75)
                    
                    for i, entry in enumerate(sample_decisions[:3]):  # Show first 3
                        q_vals = entry.get('q_values', [])
                        actions = entry.get('valid_actions', [])
                        num_competitive = entry.get('num_competitive', 0)
                        triggered = "YES" if entry.get('should_query', False) else "no"
                        num_actions = len(actions)
                        
                        print(f"  {i+1:<10} {num_competitive:<15}          "
                              f"{triggered:<12} {num_actions}")
                        
                        # Print Q-values and threshold info
                        threshold = entry.get('competitive_threshold', 0)
                        print(f"    {'Action':<10} {'Q-value':<15} {'Competitive?'}")
                        for a, q in zip(actions, q_vals):
                            is_competitive = "YES" if q > threshold else "no"
                            print(f"    {a:<10} {q:<15.4f} {is_competitive}")


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
    if args.allow_queries and not api_key:
        print("WARNING: --allow-queries requires OpenAI API key.")
        print("Set via --api-key or OPENAI_API_KEY environment variable.")
        print("Continuing without queries...")
        args.allow_queries = False

    # Create config
    config = ExperimentConfig(
        grid_size=args.grid_size,
        num_objects=args.num_objects,
        reward_ratio=args.reward_ratio,
        num_rewarding_properties=args.num_rewarding_properties,
        num_property_values=args.num_property_values,
        additive_valuation=args.additive_valuation,
        num_train_episodes=args.train_episodes,
        num_eval_episodes=args.eval_episodes,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        target_update_freq=args.target_update_freq,
        learning_starts=args.learning_starts,
        hidden_dims=hidden_dims,
        allow_queries=args.allow_queries,
        query_budget=args.query_budget,
        query_threshold=args.query_threshold,
        blend_factor=args.blend_factor,
        llm_model=args.llm_model,
        use_belief_based_agent=args.use_belief_based_agent,
        participation_ratio_threshold=args.participation_ratio_threshold,
        plackett_luce_learning_rate=args.plackett_luce_learning_rate,
        plackett_luce_gradient_steps=args.plackett_luce_gradient_steps,
        plackett_luce_info_gain=args.plackett_luce_info_gain,
        linear_gaussian_noise_variance=args.linear_gaussian_noise_variance,
        seed=args.seed
    )

    query_status = "ENABLED" if config.allow_queries else "DISABLED"
    agent_type = "BELIEF-BASED" if config.use_belief_based_agent else "DQN"

    print("=" * 60)
    print("Gridworld Variable Property Training Experiment")
    print("=" * 60)
    print("\nConfiguration:")
    print(f"  Grid size: {config.grid_size}x{config.grid_size}")
    print(f"  Number of objects: {config.num_objects}")
    print(f"  Reward ratio: {config.reward_ratio}")
    print(f"  Rewarding properties (K): {config.num_rewarding_properties}")
    print(f"  Property values per category: {config.num_property_values}")
    print(f"  Additive valuation mode: {'ENABLED' if config.additive_valuation else 'DISABLED'}")
    print(f"  Training episodes: {config.num_train_episodes}")
    print(f"  Evaluation episodes: {config.num_eval_episodes}")
    print(f"  Property counts to test: {property_counts}")
    print(f"  Number of seeds: {args.num_seeds}")
    print(f"  Base seed: {args.seed}")
    print(f"  Model directory: {args.model_dir}")
    print(f"  Output directory: {args.output_dir}")
    print(f"\n  === AGENT TYPE: {agent_type} ===")
    if config.use_belief_based_agent:
        print(f"  Participation ratio threshold: {config.participation_ratio_threshold}")
        print(f"  Plackett-Luce learning rate: {config.plackett_luce_learning_rate}")
        print(f"  Plackett-Luce gradient steps: {config.plackett_luce_gradient_steps}")
        print(f"  Plackett-Luce info gain: {config.plackett_luce_info_gain}")
        print(f"  Linear-Gaussian noise variance: {config.linear_gaussian_noise_variance}")
    else:
        print(f"  Learning rate: {config.learning_rate}")
        print(f"  Hidden dims: {config.hidden_dims}")
    print(f"\n  === QUERY SETTINGS ===")
    print(f"  Queries: {query_status}")
    if config.allow_queries:
        print(f"  Query budget: {config.query_budget}")
        if not config.use_belief_based_agent:
            print(f"  Query threshold: {config.query_threshold}")
            print(f"  Blend factor: {config.blend_factor}")
        print(f"  LLM model: {config.llm_model}")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)

    # Handle --gen-gifs mode (just generate GIFs, skip full experiment)
    if args.gen_gifs:
        print("\n" + "=" * 60)
        print("GIF Generation Mode")
        print("=" * 60)

        from gridworld.environment import GridWorld
        from gridworld.experiment import create_robot_agent, get_model_path
        from gridworld.llm_interface import LLMInterface

        # Create environment
        env = GridWorld(
            grid_size=config.grid_size,
            num_objects=config.num_objects,
            reward_ratio=config.reward_ratio,
            num_rewarding_properties=config.num_rewarding_properties,
            num_distinct_properties=5,
            num_property_values=config.num_property_values,
            additive_valuation=config.additive_valuation,
            seed=config.seed
        )

        # Create robot based on agent type
        if config.use_belief_based_agent:
            print("Setting up belief-based robot for GIF generation...")
            robot = create_belief_based_agent(config, env, api_key, verbose=True)
            print("Verbose mode enabled for belief tracking during GIF generation.")
        else:
            base_robot = create_robot_agent(config, env, verbose=False)

            # Load trained model
            model_path = get_model_path(config, args.model_dir)
            if os.path.exists(model_path):
                print(f"Loading model from: {model_path}")
                base_robot.load(model_path)
            else:
                print(f"WARNING: No trained model found at {model_path}")
                print("GIFs will use untrained agent behavior.")

            # Wrap in QueryAugmentedRobotAgent if queries are enabled
            if config.allow_queries:
                print("Setting up query-augmented robot for GIF generation...")
                llm = LLMInterface(model=config.llm_model, api_key=api_key)
                robot = QueryAugmentedRobotAgent(
                    base_agent=base_robot,
                    llm_interface=llm,
                    query_budget=config.query_budget,
                    blend_factor=config.blend_factor,
                    query_threshold=config.query_threshold,
                    verbose=True  # Enable verbose mode for detailed logging
                )
                print("Verbose mode enabled for query logging during GIF generation.")
            else:
                robot = base_robot

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
        api_key=api_key,
        model_dir=args.model_dir
    )

    # Summarize results
    summary = summarize_variable_results(results, property_counts)

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    if config.use_belief_based_agent:
        print("(Belief-Based Agent)")
    else:
        print("(DQN Agent)")
    print("=" * 60)

    prop_counts_only = sorted([k for k in summary.keys() if isinstance(k, int)])

    if config.allow_queries or config.use_belief_based_agent:
        print(f"\n{'Props':<8} {'Eval Mean':<12} {'Eval Std':<12} {'Avg Queries':<15}")
        print("-" * 50)
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

    # Print competitive options analysis if queries are enabled (DQN only)
    if config.allow_queries and not config.use_belief_based_agent:
        print_entropy_analysis(results, property_counts, config)

    # Save summary
    summary_path = os.path.join(args.output_dir, 'summary.json')
    config_dict = {
        'grid_size': config.grid_size,
        'num_objects': config.num_objects,
        'reward_ratio': config.reward_ratio,
        'num_rewarding_properties': config.num_rewarding_properties,
        'num_property_values': config.num_property_values,
        'additive_valuation': config.additive_valuation,
        'num_train_episodes': config.num_train_episodes,
        'num_eval_episodes': config.num_eval_episodes,
        'allow_queries': config.allow_queries,
        'query_budget': config.query_budget,
        'llm_model': config.llm_model,
        'num_seeds': args.num_seeds,
        'base_seed': args.seed,
        'use_belief_based_agent': config.use_belief_based_agent,
    }
    # Add agent-specific parameters
    if config.use_belief_based_agent:
        config_dict.update({
            'participation_ratio_threshold': config.participation_ratio_threshold,
            'plackett_luce_learning_rate': config.plackett_luce_learning_rate,
            'plackett_luce_gradient_steps': config.plackett_luce_gradient_steps,
            'plackett_luce_info_gain': config.plackett_luce_info_gain,
            'linear_gaussian_noise_variance': config.linear_gaussian_noise_variance,
        })
    else:
        config_dict.update({
            'learning_rate': config.learning_rate,
            'hidden_dims': config.hidden_dims,
            'query_threshold': config.query_threshold,
            'blend_factor': config.blend_factor,
        })

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
            render_episode_gifs(property_counts, args.output_dir, config, trained_robot)
        except Exception as e:
            print(f"Warning: Could not create plots: {e}")

    print("\nExperiment complete!")


if __name__ == "__main__":
    main()
