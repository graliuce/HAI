"""Experiment runner for the gridworld multi-agent task."""

from typing import Dict, List, Optional, Tuple
import numpy as np
import os
from dataclasses import dataclass, field

from .environment import GridWorld
from .agents.human import HumanAgent
from .agents.belief_based_robot import BeliefBasedRobotAgent
from .objects import PROPERTY_CATEGORIES
from tqdm import tqdm


def _print_episode_header(episode_num, human_obs, env):
    """Print a clear header at the start of each episode."""
    print("\n")
    print("=" * 100)
    print(f"EPISODE {episode_num if episode_num is not None else '?'} START")
    print("=" * 100)

    # Print human's true rewards
    print(f"\nHuman's True Rewarding Properties: {sorted(human_obs['reward_properties'])}")

    # Print the property value rewards
    if human_obs.get('property_value_rewards'):
        print("\nProperty Value Rewards:")
        prop_val_rewards = human_obs['property_value_rewards']
        # Sort by absolute reward value
        sorted_props = sorted(prop_val_rewards.items(), key=lambda x: abs(x[1]), reverse=True)
        for prop_val, reward in sorted_props:
            print(f"  {prop_val}: {reward:+.3f}")

    # Print environment info
    print(f"\nEnvironment Configuration:")
    print(f"  Grid Size: {env.grid_size}x{env.grid_size}")
    print(f"  Number of Objects: {len(env.objects)}")
    print(f"  Active Categories: {env.active_categories}")
    print(f"  Number of Distinct Properties: {env.num_distinct_properties}")

    print("=" * 100 + "\n")


@dataclass
class ExperimentConfig:
    """Configuration for the experiment."""

    # Environment parameters
    grid_size: int = 10
    num_objects: int = 20
    num_property_values: int = 5

    # Evaluation parameters
    num_eval_episodes: int = 10
    max_steps_per_episode: int = 100

    # Query parameters (for test time)
    allow_queries: bool = False
    query_budget: int = 5
    llm_model: str = "gpt-4o-mini"
    query_mode: str = "sampled_actions"  # Options: "sampled_actions", "state", "beliefs", "preference", "eig"

    # Belief-based agent parameters
    action_confidence_threshold: float = 0.6  # Query when confidence < this (0-1, higher = query more often)
    plackett_luce_learning_rate: float = 0.1
    plackett_luce_gradient_steps: int = 5
    linear_gaussian_noise_variance: float = 1.0

    # EIG-specific parameters
    eig_num_mc_samples: int = 100

    # Random seed
    seed: Optional[int] = None


@dataclass
class EpisodeResult:
    """Result of a single episode."""

    robot_reward: float = 0.0
    robot_collected: int = 0
    human_collected: int = 0
    rewarding_collected_by_robot: int = 0
    total_steps: int = 0
    reward_properties: set = field(default_factory=set)
    queries_used: int = 0
    query_history: List[Dict] = field(default_factory=list)
    entropy_history: List[Dict] = field(default_factory=list)
    # Information gain tracking
    information_gain_history: List[Dict] = field(default_factory=list)
    average_information_gain: float = 0.0
    total_information_gain: float = 0.0


@dataclass
class VariableExperimentResult:
    """Result of a variable property experiment."""

    eval_results_per_property: Dict[int, List[float]] = field(default_factory=dict)
    eval_means_per_property: Dict[int, float] = field(default_factory=dict)
    eval_stds_per_property: Dict[int, float] = field(default_factory=dict)
    # Query statistics
    eval_queries_per_property: Dict[int, List[int]] = field(default_factory=dict)
    eval_avg_queries_per_property: Dict[int, float] = field(default_factory=dict)
    # Entropy statistics
    eval_entropy_stats_per_property: Dict[int, Dict] = field(default_factory=dict)
    # Information gain statistics
    eval_avg_ig_per_property: Dict[int, float] = field(default_factory=dict)
    eval_total_ig_per_property: Dict[int, float] = field(default_factory=dict)
    eval_ig_per_query_per_property: Dict[int, List[float]] = field(default_factory=dict)


def run_episode(
    env: GridWorld,
    human: HumanAgent,
    robot: BeliefBasedRobotAgent,
    episode_num: Optional[int] = None
) -> EpisodeResult:
    """
    Run a single episode.

    Args:
        env: The GridWorld environment
        human: The human agent
        robot: The belief-based robot agent
        episode_num: Episode number for verbose output

    Returns:
        Episode result with statistics
    """
    result = EpisodeResult()

    # Reset environment and agents
    observation = env.reset()
    human_obs = env.get_human_observation()
    human.reset(
        human_obs['reward_properties'],
        object_rewards=human_obs.get('object_rewards')
    )
    result.reward_properties = human_obs['reward_properties']

    # Print episode start header if verbose
    if robot.verbose:
        _print_episode_header(episode_num, human_obs, env)

    # Reset robot - pass active categories so it initializes the correct dimensionality belief state
    robot.reset(active_categories=observation.get('active_categories'))

    # Set up human responder for queries
    robot.set_human_responder(
        human_obs['reward_properties'],
        property_value_rewards=human_obs.get('property_value_rewards')
    )

    # For verbose output, set reward properties on agent
    if hasattr(robot, 'set_reward_properties_for_verbose'):
        robot.set_reward_properties_for_verbose(human_obs['reward_properties'])

    total_reward = 0.0

    while not env.done:
        # Get actions
        human_action = human.get_action(human_obs)
        robot_action = robot.get_action(observation, training=False)

        # Execute human action first (move human)
        env.human_position = env._apply_action(env.human_position, human_action)

        # Step environment with robot action
        next_observation, reward, done, info = env.step(robot_action)

        total_reward += reward
        observation = next_observation
        human_obs = env.get_human_observation()

    # Collect statistics
    result.robot_reward = total_reward
    result.robot_collected = len(env.robot_collected)
    result.human_collected = len(env.human_collected)
    result.total_steps = env.step_count

    # Count rewarding objects collected by robot
    for obj in env.robot_collected:
        if env.get_object_reward(obj) > 0:
            result.rewarding_collected_by_robot += 1

    # Query statistics
    stats = robot.get_query_stats()
    result.queries_used = stats['queries_used']
    result.query_history = stats['query_history']
    result.entropy_history = stats.get('entropy_history', [])
    
    # Information gain statistics
    result.information_gain_history = stats.get('information_gain_history', [])
    result.average_information_gain = stats.get('average_information_gain', 0.0)
    result.total_information_gain = stats.get('total_information_gain', 0.0)

    # Print episode summary
    if episode_num is not None:
        print(f"\n{'='*100}")
        print(f"EPISODE {episode_num} COMPLETE")
        print(f"{'='*100}")
        print(f"Final Robot Reward: {total_reward:.2f}")
        print(f"Robot Collected: {result.robot_collected} objects ({result.rewarding_collected_by_robot} rewarding)")
        print(f"Human Collected: {result.human_collected} objects")
        print(f"Total Steps: {result.total_steps}")
        if result.queries_used > 0:
            print(f"Queries Used: {result.queries_used}")
            if result.average_information_gain > 0:
                print(f"Average Information Gain per Query: {result.average_information_gain:.4f} nats")
        print(f"{'='*100}\n")

    return result


def create_belief_based_agent(
    config: ExperimentConfig,
    env: GridWorld,
    api_key: Optional[str] = None,
    verbose: bool = False
) -> BeliefBasedRobotAgent:
    """
    Create a belief-based robot agent.

    Args:
        config: Experiment configuration
        env: GridWorld environment (for getting active categories)
        api_key: OpenAI API key for queries (optional)
        verbose: Whether to print debug info

    Returns:
        BeliefBasedRobotAgent
    """
    from .llm_interface import LLMInterface

    llm_interface = None
    if config.allow_queries and api_key:
        try:
            llm_interface = LLMInterface(api_key=api_key, model=config.llm_model)
        except Exception as e:
            print(f"Warning: Could not initialize LLM interface: {e}")
            print("Running without queries.")

    # Use all 5 categories to handle variable property counts
    all_categories = PROPERTY_CATEGORIES[:5]

    return BeliefBasedRobotAgent(
        grid_size=config.grid_size,
        num_objects=config.num_objects,
        active_categories=all_categories,
        num_property_values=config.num_property_values,
        llm_interface=llm_interface,
        query_budget=config.query_budget,
        query_mode=config.query_mode,
        action_confidence_threshold=config.action_confidence_threshold,
        plackett_luce_learning_rate=config.plackett_luce_learning_rate,
        plackett_luce_gradient_steps=config.plackett_luce_gradient_steps,
        linear_gaussian_noise_variance=config.linear_gaussian_noise_variance,
        eig_num_mc_samples=config.eig_num_mc_samples,
        verbose=verbose,
        seed=config.seed
    )


def run_evaluation_per_property_count(
    config: ExperimentConfig,
    robot: BeliefBasedRobotAgent,
    num_distinct_properties: int,
    num_episodes: int
) -> List[EpisodeResult]:
    """
    Run evaluation for a specific property count.
    """
    results = []

    for ep in range(num_episodes):
        env = GridWorld(
            grid_size=config.grid_size,
            num_objects=config.num_objects,
            num_distinct_properties=num_distinct_properties,
            num_property_values=config.num_property_values,
            seed=config.seed + 10000 + ep
        )

        human = HumanAgent()
        result = run_episode(env, human, robot, episode_num=ep + 1)
        results.append(result)

    return results


def run_variable_property_experiment(
    config: ExperimentConfig,
    property_counts: List[int] = None,
    num_seeds: int = 1,
    verbose: bool = True,
    api_key: Optional[str] = None
) -> Tuple[List[VariableExperimentResult], BeliefBasedRobotAgent]:
    """
    Run experiment with per-property evaluation.

    Args:
        config: Experiment configuration
        property_counts: List of property counts to use
        num_seeds: Number of random seeds
        verbose: Whether to print progress
        api_key: OpenAI API key for queries (optional)

    Returns:
        Tuple of (results list, last robot agent)
    """
    if property_counts is None:
        property_counts = [1, 2, 3, 4, 5]

    all_results: List[VariableExperimentResult] = []
    trained_robot = None

    base_seed = config.seed if config.seed is not None else 42

    for seed_idx in tqdm(range(num_seeds)):
        seed = base_seed + seed_idx

        if verbose:
            print(f"\n{'#'*60}")
            print(f"Seed {seed_idx + 1}/{num_seeds} (seed={seed})")
            print(f"{'#'*60}")

        current_config = ExperimentConfig(
            grid_size=config.grid_size,
            num_objects=config.num_objects,
            num_property_values=config.num_property_values,
            num_eval_episodes=config.num_eval_episodes,
            max_steps_per_episode=config.max_steps_per_episode,
            allow_queries=config.allow_queries,
            query_budget=config.query_budget,
            llm_model=config.llm_model,
            query_mode=config.query_mode,
            action_confidence_threshold=config.action_confidence_threshold,
            plackett_luce_learning_rate=config.plackett_luce_learning_rate,
            plackett_luce_gradient_steps=config.plackett_luce_gradient_steps,
            linear_gaussian_noise_variance=config.linear_gaussian_noise_variance,
            eig_num_mc_samples=config.eig_num_mc_samples,
            seed=seed
        )

        env = GridWorld(
            grid_size=current_config.grid_size,
            num_objects=current_config.num_objects,
            num_distinct_properties=5,
            num_property_values=current_config.num_property_values,
            seed=seed
        )

        if verbose:
            print("\nUsing Belief-Based Agent")
            print(f"  Action confidence threshold: {current_config.action_confidence_threshold}")
            print(f"  Plackett-Luce learning rate: {current_config.plackett_luce_learning_rate}")
            print(f"  Linear-Gaussian noise variance: {current_config.linear_gaussian_noise_variance}")
            if current_config.allow_queries:
                print(f"  Query budget: {current_config.query_budget}")
                print(f"  Query mode: {current_config.query_mode}")
                if current_config.query_mode == "eig":
                    print(f"  EIG MC samples: {current_config.eig_num_mc_samples}")
                    print(f"  EIG features per option: One per active property category")

        eval_robot = create_belief_based_agent(
            current_config, env, api_key, verbose=True
        )

        # Evaluate
        result = VariableExperimentResult()

        if verbose:
            print("\nEvaluating on each property count...")

        for num_props in property_counts:
            eval_results = run_evaluation_per_property_count(
                current_config,
                eval_robot,
                num_props,
                current_config.num_eval_episodes
            )

            eval_rewards = [r.robot_reward for r in eval_results]
            eval_queries = [r.queries_used for r in eval_results]

            result.eval_results_per_property[num_props] = eval_rewards
            result.eval_means_per_property[num_props] = np.mean(eval_rewards)
            result.eval_stds_per_property[num_props] = np.std(eval_rewards)
            result.eval_queries_per_property[num_props] = eval_queries
            result.eval_avg_queries_per_property[num_props] = np.mean(eval_queries)

            # Collect information gain statistics
            if current_config.allow_queries:
                all_ig_values = []
                total_ig_sum = 0.0
                
                for r in eval_results:
                    if r.information_gain_history:
                        for ig_entry in r.information_gain_history:
                            all_ig_values.append(ig_entry['information_gain'])
                        total_ig_sum += r.total_information_gain
                
                if all_ig_values:
                    result.eval_avg_ig_per_property[num_props] = np.mean(all_ig_values)
                    result.eval_total_ig_per_property[num_props] = total_ig_sum / len(eval_results)
                    result.eval_ig_per_query_per_property[num_props] = all_ig_values
                else:
                    result.eval_avg_ig_per_property[num_props] = 0.0
                    result.eval_total_ig_per_property[num_props] = 0.0
                    result.eval_ig_per_query_per_property[num_props] = []

                # Collect entropy/uncertainty statistics
                all_entropy_history = []
                for r in eval_results:
                    all_entropy_history.extend(r.entropy_history)

                if all_entropy_history:
                    num_decision_points = len(all_entropy_history)
                    num_triggered = sum(1 for h in all_entropy_history if h['should_query'])

                    # Belief-based agent format
                    if 'participation_ratio' in all_entropy_history[0]:
                        pr_values = [h['participation_ratio'] for h in all_entropy_history]
                        result.eval_entropy_stats_per_property[num_props] = {
                            'mean_participation_ratio': np.mean(pr_values),
                            'std_participation_ratio': np.std(pr_values),
                            'num_decision_points': num_decision_points,
                            'num_triggered': num_triggered,
                            'trigger_rate': num_triggered / num_decision_points if num_decision_points > 0 else 0,
                            'sample_decision_points': all_entropy_history[:5]
                        }

            if verbose:
                query_info = f", avg queries={result.eval_avg_queries_per_property[num_props]:.2f}" if current_config.allow_queries else ""
                ig_info = ""
                if current_config.allow_queries and num_props in result.eval_avg_ig_per_property:
                    ig_info = f", avg IG={result.eval_avg_ig_per_property[num_props]:.4f}"
                print(f"  {num_props} properties: "
                      f"mean={result.eval_means_per_property[num_props]:.2f}, "
                      f"std={result.eval_stds_per_property[num_props]:.2f}"
                      f"{query_info}{ig_info}")

        all_results.append(result)
        trained_robot = eval_robot

    return all_results, trained_robot


def summarize_variable_results(
    results: List[VariableExperimentResult],
    property_counts: List[int]
) -> Dict[int, Dict[str, float]]:
    """Summarize variable property experiment results."""
    summary = {}

    for num_props in property_counts:
        # Collect all individual episode returns across all seeds
        all_eval_returns = []
        for r in results:
            all_eval_returns.extend(r.eval_results_per_property[num_props])
        
        # Collect means per seed (for backwards compatibility)
        eval_means = [r.eval_means_per_property[num_props] for r in results]
        avg_queries = [r.eval_avg_queries_per_property.get(num_props, 0) for r in results]
        
        # Aggregate information gain statistics
        avg_ig_values = [r.eval_avg_ig_per_property.get(num_props, 0) for r in results if num_props in r.eval_avg_ig_per_property]
        total_ig_values = [r.eval_total_ig_per_property.get(num_props, 0) for r in results if num_props in r.eval_total_ig_per_property]

        # Compute 95% confidence interval across all episodes and seeds
        mean_return = np.mean(all_eval_returns)
        std_return = np.std(all_eval_returns, ddof=1)  # Sample std
        n_samples = len(all_eval_returns)
        
        if n_samples > 1:
            # Use t-distribution for 95% CI (t-critical values approximation)
            # For small n, use exact t-values; for large n, converges to 1.96
            if n_samples <= 30:
                # t-critical values for common small sample sizes at 95% CI
                t_critical_values = {
                    2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571,
                    7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262, 11: 2.228,
                    12: 2.201, 13: 2.179, 14: 2.160, 15: 2.145, 16: 2.131,
                    17: 2.120, 18: 2.110, 19: 2.101, 20: 2.093, 21: 2.086,
                    22: 2.080, 23: 2.074, 24: 2.069, 25: 2.064, 26: 2.060,
                    27: 2.056, 28: 2.052, 29: 2.048, 30: 2.045
                }
                t_crit = t_critical_values.get(n_samples, 2.045)
            else:
                # For large samples, use normal approximation
                t_crit = 1.96
            
            sem = std_return / np.sqrt(n_samples)
            ci_95_error = t_crit * sem
            ci_95_lower = mean_return - ci_95_error
            ci_95_upper = mean_return + ci_95_error
        else:
            ci_95_lower = mean_return
            ci_95_upper = mean_return
            ci_95_error = 0.0

        summary[num_props] = {
            'eval_mean': mean_return,
            'eval_std': std_return,
            'eval_sem': std_return / np.sqrt(n_samples) if n_samples > 1 else 0.0,
            'ci_95_lower': ci_95_lower,
            'ci_95_upper': ci_95_upper,
            'ci_95_error': ci_95_error,
            'n_samples': n_samples,
            'avg_queries': np.mean(avg_queries),
            'avg_information_gain': np.mean(avg_ig_values) if avg_ig_values else 0.0,
            'avg_total_information_gain': np.mean(total_ig_values) if total_ig_values else 0.0,
        }

    return summary


def render_episode_gif(
    num_distinct_properties: int,
    config: ExperimentConfig,
    output_path: str,
    max_steps: int = 50,
    fps: int = 4,
    trained_robot: Optional[BeliefBasedRobotAgent] = None
) -> float:
    """Render and save a GIF of an entire episode.
    
    Returns:
        Robot's total reward for the episode
    """
    import imageio

    if trained_robot is None:
        raise ValueError("Missing trained robot agent.")

    robot = trained_robot

    np.random.seed(None)

    vis_env = GridWorld(
        grid_size=config.grid_size,
        num_objects=config.num_objects,
        num_distinct_properties=num_distinct_properties,
        num_property_values=config.num_property_values,
        seed=np.random.randint(0, 1000000)
    )

    human = HumanAgent()

    observation = vis_env.reset()
    human_obs = vis_env.get_human_observation()
    human.reset(
        human_obs['reward_properties'],
        object_rewards=human_obs.get('object_rewards')
    )
    robot.reset()

    # Set up human responder for query-augmented agent
    robot.set_human_responder(
        human_obs['reward_properties'],
        property_value_rewards=human_obs.get('property_value_rewards')
    )

    frames = []
    frames.append(vis_env.render_to_array())

    robot_total_reward = 0.0
    step = 0
    while not vis_env.done and step < max_steps:
        human_action = human.get_action(human_obs)
        robot_action = robot.get_action(observation, training=False)

        vis_env.human_position = vis_env._apply_action(vis_env.human_position, human_action)
        observation, robot_reward, _, _ = vis_env.step(robot_action)
        robot_total_reward += robot_reward
        human_obs = vis_env.get_human_observation()

        frames.append(vis_env.render_to_array())
        step += 1

    imageio.mimsave(output_path, frames, fps=fps, loop=0)
    return robot_total_reward