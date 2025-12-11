"""Experiment runner for the gridworld multi-agent task."""

from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass, field

from .environment import GridWorld
from .agents.human import HumanAgent
from .agents.robot import RobotAgent
from tqdm import tqdm


@dataclass
class ExperimentConfig:
    """Configuration for the experiment."""

    # Environment parameters
    grid_size: int = 10
    num_objects: int = 20
    reward_ratio: float = 0.3
    num_rewarding_properties: int = 2  # K

    # Training parameters
    num_train_episodes: int = 1000
    num_eval_episodes: int = 10
    max_steps_per_episode: int = 100

    # Robot learning parameters
    learning_rate: float = 0.1
    discount_factor: float = 0.95
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995

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


@dataclass
class ExperimentResult:
    """Result of an experiment run."""

    num_distinct_properties: int
    train_rewards: List[float] = field(default_factory=list)
    eval_rewards: List[float] = field(default_factory=list)
    eval_mean: float = 0.0
    eval_std: float = 0.0
    train_mean: float = 0.0


def run_episode(
    env: GridWorld,
    human: HumanAgent,
    robot: RobotAgent,
    training: bool = True
) -> EpisodeResult:
    """
    Run a single episode.

    Args:
        env: The GridWorld environment
        human: The human agent
        robot: The robot agent
        training: Whether to update robot's Q-values

    Returns:
        Episode result with statistics
    """
    result = EpisodeResult()

    # Reset environment and agents
    observation = env.reset()
    human_obs = env.get_human_observation()
    human.reset(human_obs['reward_properties'])
    result.reward_properties = human_obs['reward_properties']

    robot.reset()

    total_reward = 0.0
    state = env.get_state_for_robot()

    while not env.done:
        # Get actions
        human_action = human.get_action(human_obs)
        robot_action = robot.get_action(observation, state, training=training)

        # Execute human action first (move human)
        env.human_position = env._apply_action(env.human_position, human_action)

        # Step environment with robot action
        next_observation, reward, done, info = env.step(robot_action)
        next_state = env.get_state_for_robot()

        # Update robot if training
        if training:
            robot.update(state, robot_action, reward, next_state, done)

        total_reward += reward
        state = next_state
        observation = next_observation
        human_obs = env.get_human_observation()

    # Collect statistics
    result.robot_reward = total_reward
    result.robot_collected = len(env.robot_collected)
    result.human_collected = len(env.human_collected)
    result.total_steps = env.step_count

    # Count rewarding objects collected by robot
    for obj in env.robot_collected:
        if obj.has_any_property(result.reward_properties):
            result.rewarding_collected_by_robot += 1

    return result


def run_training(
    env: GridWorld,
    human: HumanAgent,
    robot: RobotAgent,
    num_episodes: int,
    verbose: bool = False
) -> List[float]:
    """
    Run training for multiple episodes.

    Args:
        env: The GridWorld environment
        human: The human agent
        robot: The robot agent
        num_episodes: Number of training episodes
        verbose: Whether to print progress

    Returns:
        List of episode rewards
    """
    rewards = []

    for episode in range(num_episodes):
        result = run_episode(env, human, robot, training=True)
        rewards.append(result.robot_reward)

        # Decay exploration rate
        robot.decay_epsilon()

        if verbose and (episode + 1) % 100 == 0:
            recent_avg = np.mean(rewards[-100:])
            print(f"Episode {episode + 1}/{num_episodes}, "
                  f"Recent avg reward: {recent_avg:.2f}, "
                  f"Epsilon: {robot.epsilon:.3f}")

    return rewards


def run_evaluation(
    env: GridWorld,
    human: HumanAgent,
    robot: RobotAgent,
    num_episodes: int
) -> List[EpisodeResult]:
    """
    Run evaluation for multiple episodes.

    Args:
        env: The GridWorld environment
        human: The human agent
        robot: The robot agent
        num_episodes: Number of evaluation episodes

    Returns:
        List of episode results
    """
    results = []

    # Store original epsilon and set to 0 for evaluation
    original_epsilon = robot.epsilon
    robot.epsilon = 0.0

    for _ in range(num_episodes):
        result = run_episode(env, human, robot, training=False)
        results.append(result)

    # Restore epsilon
    robot.epsilon = original_epsilon

    return results


def run_experiment(
    num_distinct_properties: int,
    config: ExperimentConfig,
    verbose: bool = False
) -> ExperimentResult:
    """
    Run a complete experiment with specified number of distinct properties.

    Args:
        num_distinct_properties: Number of property categories to vary (1-5)
        config: Experiment configuration
        verbose: Whether to print progress

    Returns:
        Experiment result with training and evaluation statistics
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Running experiment with {num_distinct_properties} distinct properties")
        print(f"{'='*60}")

    # Create environment
    env = GridWorld(
        grid_size=config.grid_size,
        num_objects=config.num_objects,
        reward_ratio=config.reward_ratio,
        num_rewarding_properties=config.num_rewarding_properties,
        num_distinct_properties=num_distinct_properties,
        seed=config.seed
    )

    # Create agents
    human = HumanAgent()
    robot = RobotAgent(
        num_actions=env.NUM_ACTIONS,
        learning_rate=config.learning_rate,
        discount_factor=config.discount_factor,
        epsilon_start=config.epsilon_start,
        epsilon_end=config.epsilon_end,
        epsilon_decay=config.epsilon_decay,
        seed=config.seed
    )

    # Run training
    if verbose:
        print("\nTraining...")
    train_rewards = run_training(
        env, human, robot,
        num_episodes=config.num_train_episodes,
        verbose=verbose
    )

    # Run evaluation
    if verbose:
        print("\nEvaluating...")
    eval_results = run_evaluation(
        env, human, robot,
        num_episodes=config.num_eval_episodes
    )

    eval_rewards = [r.robot_reward for r in eval_results]

    # Compile results
    result = ExperimentResult(
        num_distinct_properties=num_distinct_properties,
        train_rewards=train_rewards,
        eval_rewards=eval_rewards,
        eval_mean=np.mean(eval_rewards),
        eval_std=np.std(eval_rewards),
        train_mean=np.mean(train_rewards[-100:])  # Last 100 episodes
    )

    if verbose:
        print(f"\nResults for {num_distinct_properties} distinct properties:")
        print(f"  Training mean (last 100): {result.train_mean:.2f}")
        print(f"  Evaluation mean: {result.eval_mean:.2f} ± {result.eval_std:.2f}")

    return result


def run_property_variation_experiment(
    config: ExperimentConfig,
    property_counts: List[int] = None,
    num_seeds: int = 5,
    verbose: bool = True
) -> Dict[int, List[ExperimentResult]]:
    """
    Run experiments varying the number of distinct properties.

    This is the main experiment to show that increasing distinct properties
    decreases the robot's ability to infer reward properties.

    Args:
        config: Base experiment configuration
        property_counts: List of distinct property counts to test (default: [1,2,3,4,5])
        num_seeds: Number of random seeds to average over
        verbose: Whether to print progress

    Returns:
        Dictionary mapping property count to list of results (one per seed)
    """
    if property_counts is None:
        property_counts = [1, 2, 3, 4, 5]

    all_results: Dict[int, List[ExperimentResult]] = {
        count: [] for count in property_counts
    }

    base_seed = config.seed if config.seed is not None else 42

    for seed_idx in tqdm(range(num_seeds)):
        seed = base_seed + seed_idx

        if verbose:
            print(f"\n{'#'*60}")
            print(f"Seed {seed_idx + 1}/{num_seeds} (seed={seed})")
            print(f"{'#'*60}")

        for num_props in property_counts:
            # Update config with current seed
            current_config = ExperimentConfig(
                grid_size=config.grid_size,
                num_objects=config.num_objects,
                reward_ratio=config.reward_ratio,
                num_rewarding_properties=config.num_rewarding_properties,
                num_train_episodes=config.num_train_episodes,
                num_eval_episodes=config.num_eval_episodes,
                max_steps_per_episode=config.max_steps_per_episode,
                learning_rate=config.learning_rate,
                discount_factor=config.discount_factor,
                epsilon_start=config.epsilon_start,
                epsilon_end=config.epsilon_end,
                epsilon_decay=config.epsilon_decay,
                seed=seed
            )

            result = run_experiment(
                num_distinct_properties=num_props,
                config=current_config,
                verbose=verbose
            )

            all_results[num_props].append(result)

    return all_results


def render_episode_gif(
    num_distinct_properties: int,
    config: ExperimentConfig,
    output_path: str,
    max_steps: int = 50,
    fps: int = 4
) -> None:
    """
    Render and save a GIF of an entire episode for visualization.

    Args:
        num_distinct_properties: Number of property categories to vary (1-5)
        config: Experiment configuration
        output_path: Path to save the GIF
        max_steps: Maximum number of steps to render
        fps: Frames per second for the GIF
    """
    import imageio

    # Create environment with a fixed seed for reproducibility
    env = GridWorld(
        grid_size=config.grid_size,
        num_objects=config.num_objects,
        reward_ratio=config.reward_ratio,
        num_rewarding_properties=config.num_rewarding_properties,
        num_distinct_properties=num_distinct_properties,
        seed=config.seed
    )

    # Create agents
    human = HumanAgent()
    robot = RobotAgent(
        num_actions=env.NUM_ACTIONS,
        learning_rate=config.learning_rate,
        discount_factor=config.discount_factor,
        epsilon_start=0.0,  # No exploration for visualization
        epsilon_end=0.0,
        epsilon_decay=1.0,
        seed=config.seed
    )

    # Reset environment
    observation = env.reset()
    human_obs = env.get_human_observation()
    human.reset(human_obs['reward_properties'])
    robot.reset()

    # Collect frames
    frames = []

    # Capture initial frame
    frames.append(env.render_to_array())

    state = env.get_state_for_robot()
    step = 0

    while not env.done and step < max_steps:
        # Get actions
        human_action = human.get_action(human_obs)
        robot_action = robot.get_action(observation, state, training=False)

        # Execute human action first
        env.human_position = env._apply_action(env.human_position, human_action)

        # Step environment with robot action
        observation, _, _, _ = env.step(robot_action)
        human_obs = env.get_human_observation()
        state = env.get_state_for_robot()

        # Capture frame
        frames.append(env.render_to_array())
        step += 1

    # Save as GIF
    imageio.mimsave(output_path, frames, fps=fps, loop=0)


def summarize_results(
    results: Dict[int, List[ExperimentResult]]
) -> Dict[int, Dict[str, float]]:
    """
    Summarize experiment results across seeds.

    Args:
        results: Dictionary from run_property_variation_experiment

    Returns:
        Summary statistics for each property count
    """
    summary = {}

    for num_props, result_list in results.items():
        eval_means = [r.eval_mean for r in result_list]
        train_means = [r.train_mean for r in result_list]

        summary[num_props] = {
            'eval_mean': np.mean(eval_means),
            'eval_std': np.std(eval_means),
            'eval_sem': np.std(eval_means) / np.sqrt(len(eval_means)),
            'train_mean': np.mean(train_means),
            'train_std': np.std(train_means),
        }

    return summary
