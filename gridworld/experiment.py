"""Experiment runner for the gridworld multi-agent task."""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass, field

from .environment import GridWorld
from .agents.human import HumanAgent
from .agents.dqn_robot import DQNRobotAgent
from .agents.hierarchical_dqn_robot import HierarchicalDQNRobotAgent
from .objects import PROPERTY_CATEGORIES
from tqdm import tqdm

# Type alias for robot agents
RobotAgent = Union[DQNRobotAgent, HierarchicalDQNRobotAgent]


@dataclass
class ExperimentConfig:
    """Configuration for the experiment."""

    # Environment parameters
    grid_size: int = 10
    num_objects: int = 20
    reward_ratio: float = 0.4
    num_rewarding_properties: int = 2  # K

    # Training parameters
    num_train_episodes: int = 1000
    num_eval_episodes: int = 10
    max_steps_per_episode: int = 100

    # DQN learning parameters (stable-baselines3 style)
    learning_rate: float = 1e-4
    discount_factor: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    exploration_fraction: float = 0.1  # Fraction of total timesteps for epsilon decay

    # DQN-specific parameters
    buffer_size: int = 100000
    batch_size: int = 32
    target_update_freq: int = 10000
    train_freq: int = 4
    gradient_steps: int = 1
    learning_starts: int = 1000
    hidden_dims: List[int] = field(default_factory=lambda: [64, 64])

    # Hierarchical policy parameters
    use_hierarchical: bool = False
    high_level_interval: int = 3  # Steps between high-level decisions (only when goal is null)

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
class VariableExperimentResult:
    """Result of a variable property training experiment."""

    train_rewards: List[float] = field(default_factory=list)
    eval_results_per_property: Dict[int, List[float]] = field(default_factory=dict)
    eval_means_per_property: Dict[int, float] = field(default_factory=dict)
    eval_stds_per_property: Dict[int, float] = field(default_factory=dict)
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
        robot: The DQN robot agent
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

    while not env.done:
        # Get actions
        human_action = human.get_action(human_obs)
        robot_action = robot.get_action(observation, training=training)

        # Execute human action first (move human)
        env.human_position = env._apply_action(env.human_position, human_action)

        # Step environment with robot action
        next_observation, reward, done, info = env.step(robot_action)

        # Update robot if training
        if training:
            robot.update(
                action=robot_action,
                reward=reward,
                done=done,
                observation=observation,
                next_observation=next_observation
            )

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
        if obj.has_any_property(result.reward_properties):
            result.rewarding_collected_by_robot += 1

    return result


def create_variable_robot_agent(
    config: ExperimentConfig,
    env: GridWorld,
) -> RobotAgent:
    """
    Create a robot agent that can handle all property counts.

    This agent is trained with all 5 property categories so it can
    generalize across different numbers of active properties.

    Args:
        config: Experiment configuration
        env: The GridWorld environment

    Returns:
        Robot agent (flat DQN or hierarchical) with all categories active
    """
    # Use ALL property categories so the agent can handle any property count
    all_categories = PROPERTY_CATEGORIES[:5]
    # Calculate total timesteps for exploration schedule
    total_timesteps = config.num_train_episodes * config.max_steps_per_episode

    if config.use_hierarchical:
        return HierarchicalDQNRobotAgent(
            num_actions=env.NUM_ACTIONS,
            high_level_interval=config.high_level_interval,
            learning_rate=config.learning_rate,
            discount_factor=config.discount_factor,
            epsilon_start=config.epsilon_start,
            epsilon_end=config.epsilon_end,
            exploration_fraction=config.exploration_fraction,
            total_timesteps=total_timesteps,
            buffer_size=config.buffer_size,
            batch_size=config.batch_size,
            target_update_freq=config.target_update_freq,
            train_freq=config.train_freq,
            gradient_steps=config.gradient_steps,
            learning_starts=config.learning_starts,
            hidden_dims=config.hidden_dims,
            grid_size=config.grid_size,
            num_objects=config.num_objects,
            active_categories=all_categories,
            seed=config.seed
        )
    else:
        return DQNRobotAgent(
            num_actions=env.NUM_ACTIONS,
            learning_rate=config.learning_rate,
            discount_factor=config.discount_factor,
            epsilon_start=config.epsilon_start,
            epsilon_end=config.epsilon_end,
            exploration_fraction=config.exploration_fraction,
            total_timesteps=total_timesteps,
            buffer_size=config.buffer_size,
            batch_size=config.batch_size,
            target_update_freq=config.target_update_freq,
            train_freq=config.train_freq,
            gradient_steps=config.gradient_steps,
            learning_starts=config.learning_starts,
            hidden_dims=config.hidden_dims,
            grid_size=config.grid_size,
            active_categories=all_categories,
            seed=config.seed
        )


def run_variable_training(
    config: ExperimentConfig,
    robot: RobotAgent,
    property_counts: List[int],
    verbose: bool = False
) -> List[float]:
    """
    Run training where property count varies each episode.

    Each training episode randomly samples a property count from
    the provided list, creating a curriculum that exposes the agent
    to all difficulty levels.

    Args:
        config: Experiment configuration
        robot: The DQN robot agent (must be initialized with all categories)
        property_counts: List of property counts to sample from (e.g., [1,2,3,4,5])
        verbose: Whether to print progress

    Returns:
        List of episode rewards
    """
    rewards = []
    rng = np.random.RandomState(config.seed)

    for episode in tqdm(range(config.num_train_episodes)):
        # Randomly sample number of distinct properties for this episode
        num_props = rng.choice(property_counts)

        # Create environment with sampled property count
        env = GridWorld(
            grid_size=config.grid_size,
            num_objects=config.num_objects,
            reward_ratio=config.reward_ratio,
            num_rewarding_properties=config.num_rewarding_properties,
            num_distinct_properties=num_props,
            seed=config.seed + episode  # Vary seed per episode
        )

        human = HumanAgent()

        # Run episode
        result = run_episode(env, human, robot, training=True)
        rewards.append(result.robot_reward)

        # Decay exploration rate
        robot.decay_epsilon()

        if verbose and (episode + 1) % 100 == 0:
            recent_avg = np.mean(rewards[-100:])
            avg_loss = robot.get_average_loss(100)
            print(f"Episode {episode + 1}/{config.num_train_episodes}, "
                  f"Recent avg reward: {recent_avg:.2f}, "
                  f"Epsilon: {robot.epsilon:.3f}, "
                  f"Loss: {avg_loss:.4f}")

    return rewards


def run_evaluation_per_property_count(
    config: ExperimentConfig,
    robot: RobotAgent,
    num_distinct_properties: int,
    num_episodes: int
) -> List[EpisodeResult]:
    """
    Run evaluation for a specific property count.

    Args:
        config: Experiment configuration
        robot: The trained DQN robot agent
        num_distinct_properties: Number of property categories to use
        num_episodes: Number of evaluation episodes

    Returns:
        List of episode results
    """
    results = []

    # Store original epsilon and set to 0 for evaluation
    original_epsilon = robot.epsilon
    robot.epsilon = 0.0

    for ep in range(num_episodes):
        # Create environment with specific property count
        env = GridWorld(
            grid_size=config.grid_size,
            num_objects=config.num_objects,
            reward_ratio=config.reward_ratio,
            num_rewarding_properties=config.num_rewarding_properties,
            num_distinct_properties=num_distinct_properties,
            seed=config.seed + 10000 + ep  # Different seeds for eval
        )

        human = HumanAgent()
        result = run_episode(env, human, robot, training=False)
        results.append(result)

    # Restore epsilon
    robot.epsilon = original_epsilon

    return results


def run_variable_property_experiment(
    config: ExperimentConfig,
    property_counts: List[int] = None,
    num_seeds: int = 1,
    verbose: bool = True
) -> Tuple[List[VariableExperimentResult], RobotAgent]:
    """
    Run experiment with variable property training and per-property evaluation.

    Training: Agent is trained with randomly varying property counts each episode.
    Evaluation: Agent is evaluated separately on each specific property count.

    This tests whether an agent trained on variable complexity can generalize
    well to different difficulty levels.

    Args:
        config: Base experiment configuration
        property_counts: List of property counts to use (default: [1,2,3,4,5])
        num_seeds: Number of random seeds to average over
        verbose: Whether to print progress

    Returns:
        Tuple of (results list, last trained robot)
        - results: List of VariableExperimentResult (one per seed)
        - robot: The last trained robot agent (for visualization)
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
            exploration_fraction=config.exploration_fraction,
            buffer_size=config.buffer_size,
            batch_size=config.batch_size,
            target_update_freq=config.target_update_freq,
            train_freq=config.train_freq,
            gradient_steps=config.gradient_steps,
            learning_starts=config.learning_starts,
            hidden_dims=config.hidden_dims,
            use_hierarchical=config.use_hierarchical,
            high_level_interval=config.high_level_interval,
            seed=seed
        )

        # Create environment (just for robot initialization)
        env = GridWorld(
            grid_size=current_config.grid_size,
            num_objects=current_config.num_objects,
            reward_ratio=current_config.reward_ratio,
            num_rewarding_properties=current_config.num_rewarding_properties,
            num_distinct_properties=5,  # Max for initialization
            seed=seed
        )

        # Create robot with ALL property categories
        robot = create_variable_robot_agent(current_config, env)

        if verbose:
            print("\nTraining with variable property counts...")

        # Train with variable property counts
        train_rewards = run_variable_training(
            current_config,
            robot,
            property_counts,
            verbose=verbose
        )

        # Evaluate on each property count separately
        result = VariableExperimentResult(
            train_rewards=train_rewards,
            train_mean=np.mean(train_rewards[-100:])
        )

        if verbose:
            print("\nEvaluating on each property count...")

        for num_props in property_counts:
            eval_results = run_evaluation_per_property_count(
                current_config,
                robot,
                num_props,
                current_config.num_eval_episodes
            )

            eval_rewards = [r.robot_reward for r in eval_results]
            result.eval_results_per_property[num_props] = eval_rewards
            result.eval_means_per_property[num_props] = np.mean(eval_rewards)
            result.eval_stds_per_property[num_props] = np.std(eval_rewards)

            if verbose:
                print(f"  {num_props} properties: "
                      f"mean={result.eval_means_per_property[num_props]:.2f}, "
                      f"std={result.eval_stds_per_property[num_props]:.2f}")

        all_results.append(result)
        trained_robot = robot

    return all_results, trained_robot


def summarize_variable_results(
    results: List[VariableExperimentResult],
    property_counts: List[int]
) -> Dict[int, Dict[str, float]]:
    """
    Summarize variable property experiment results.

    Args:
        results: List of VariableExperimentResult from each seed
        property_counts: List of property counts that were tested

    Returns:
        Summary statistics for each property count
    """
    summary = {}

    for num_props in property_counts:
        # Gather eval means across all seeds
        eval_means = [r.eval_means_per_property[num_props] for r in results]

        summary[num_props] = {
            'eval_mean': np.mean(eval_means),
            'eval_std': np.std(eval_means),
            'eval_sem': np.std(eval_means) / np.sqrt(len(eval_means)) if len(eval_means) > 1 else 0.0,
        }

    # Also compute overall training mean
    train_means = [r.train_mean for r in results]
    summary['training'] = {
        'train_mean': np.mean(train_means),
        'train_std': np.std(train_means),
    }

    return summary


def render_episode_gif(
    num_distinct_properties: int,
    config: ExperimentConfig,
    output_path: str,
    max_steps: int = 50,
    fps: int = 4,
    trained_robot: Optional[RobotAgent] = None
) -> None:
    """
    Render and save a GIF of an entire episode for visualization.

    Args:
        num_distinct_properties: Number of property categories to vary (1-5)
        config: Experiment configuration
        output_path: Path to save the GIF
        max_steps: Maximum number of steps to render
        fps: Frames per second for the GIF
        trained_robot: Pre-trained robot agent (required)
    """
    import imageio

    if trained_robot is None:
        raise ValueError("Missing trained robot agent.")

    robot = trained_robot

    # Seed numpy's RNG with a truly random value for different episodes each run
    np.random.seed(None)

    # Create environment for visualization with a random seed
    vis_env = GridWorld(
        grid_size=config.grid_size,
        num_objects=config.num_objects,
        reward_ratio=config.reward_ratio,
        num_rewarding_properties=config.num_rewarding_properties,
        num_distinct_properties=num_distinct_properties,
        seed=np.random.randint(0, 1000000)
    )

    # Create human agent for visualization
    human = HumanAgent()

    # Set robot to evaluation mode (no exploration)
    original_epsilon = robot.epsilon
    robot.epsilon = 0.0

    # Reset environment
    observation = vis_env.reset()
    human_obs = vis_env.get_human_observation()
    human.reset(human_obs['reward_properties'])
    robot.reset()

    # Collect frames
    frames = []

    # Capture initial frame
    frames.append(vis_env.render_to_array())

    step = 0

    while not vis_env.done and step < max_steps:
        # Get actions
        human_action = human.get_action(human_obs)
        robot_action = robot.get_action(observation, training=False)

        # Execute human action first
        vis_env.human_position = vis_env._apply_action(vis_env.human_position, human_action)

        # Step environment with robot action
        observation, _, _, _ = vis_env.step(robot_action)
        human_obs = vis_env.get_human_observation()

        # Capture frame
        frames.append(vis_env.render_to_array())
        step += 1

    # Restore epsilon
    robot.epsilon = original_epsilon

    # Save as GIF
    imageio.mimsave(output_path, frames, fps=fps, loop=0)
