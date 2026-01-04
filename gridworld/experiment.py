"""Experiment runner for the gridworld multi-agent task."""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass, field

from .environment import GridWorld
from .agents.human import HumanAgent
from .agents.dqn_robot import DQNRobotAgent
from .agents.hierarchical_dqn_robot import HierarchicalDQNRobotAgent
from .agents.query_augmented_robot import QueryAugmentedRobotAgent
from .objects import PROPERTY_CATEGORIES
from tqdm import tqdm

# Type alias for robot agents
RobotAgent = Union[DQNRobotAgent, HierarchicalDQNRobotAgent, QueryAugmentedRobotAgent]


@dataclass
class ExperimentConfig:
    """Configuration for the experiment."""

    # Environment parameters
    grid_size: int = 10
    num_objects: int = 20
    reward_ratio: float = 0.4
    num_rewarding_properties: int = 2
    num_property_values: int = 5

    # Training parameters
    num_train_episodes: int = 1000
    num_eval_episodes: int = 10
    max_steps_per_episode: int = 100

    # DQN learning parameters
    learning_rate: float = 1e-4
    discount_factor: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    exploration_fraction: float = 0.1

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
    high_level_interval: int = 3

    # Query parameters (for test time)
    allow_queries: bool = False
    query_budget: int = 5
    confidence_threshold: float = 3.0
    blend_factor: float = 0.5
    llm_model: str = "gpt-4o-mini"

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


@dataclass
class VariableExperimentResult:
    """Result of a variable property training experiment."""

    train_rewards: List[float] = field(default_factory=list)
    eval_results_per_property: Dict[int, List[float]] = field(default_factory=dict)
    eval_means_per_property: Dict[int, float] = field(default_factory=dict)
    eval_stds_per_property: Dict[int, float] = field(default_factory=dict)
    train_mean: float = 0.0
    # Query statistics
    eval_queries_per_property: Dict[int, List[int]] = field(default_factory=dict)
    eval_avg_queries_per_property: Dict[int, float] = field(default_factory=dict)


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
        robot: The robot agent (DQN, Hierarchical, or QueryAugmented)
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

    # If using query-augmented agent, set up human responder
    if isinstance(robot, QueryAugmentedRobotAgent) and not training:
        robot.set_human_responder(human_obs['reward_properties'])

    total_reward = 0.0

    while not env.done:
        # Get actions
        human_action = human.get_action(human_obs)
        robot_action = robot.get_action(observation, training=training)

        # Execute human action first (move human)
        env.human_position = env._apply_action(env.human_position, human_action)

        # Step environment with robot action
        next_observation, reward, done, info = env.step(robot_action)

        # Update robot if training (and not query-augmented, which doesn't train)
        if training and hasattr(robot, 'update'):
            # For QueryAugmentedRobotAgent, we update the base agent
            if isinstance(robot, QueryAugmentedRobotAgent):
                robot.base_agent.update(
                    action=robot_action,
                    reward=reward,
                    done=done,
                    observation=observation,
                    next_observation=next_observation
                )
            else:
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

    # Query statistics
    if isinstance(robot, QueryAugmentedRobotAgent):
        stats = robot.get_query_stats()
        result.queries_used = stats['queries_used']
        result.query_history = stats['query_history']

    return result


def create_variable_robot_agent(
    config: ExperimentConfig,
    env: GridWorld,
) -> Union[DQNRobotAgent, HierarchicalDQNRobotAgent]:
    """
    Create a robot agent that can handle all property counts.
    """
    all_categories = PROPERTY_CATEGORIES[:5]
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


def create_query_augmented_agent(
    base_agent: HierarchicalDQNRobotAgent,
    config: ExperimentConfig,
    api_key: Optional[str] = None
) -> QueryAugmentedRobotAgent:
    """
    Create a query-augmented agent wrapping the base agent.
    
    Args:
        base_agent: Pre-trained hierarchical DQN agent
        config: Experiment configuration
        api_key: OpenAI API key (optional, can use env var)
    
    Returns:
        QueryAugmentedRobotAgent
    """
    from .llm_interface import LLMInterface
    
    llm_interface = None
    if config.allow_queries and api_key:
        try:
            llm_interface = LLMInterface(api_key=api_key, model=config.llm_model)
        except Exception as e:
            print(f"Warning: Could not initialize LLM interface: {e}")
            print("Running without queries.")
    
    return QueryAugmentedRobotAgent(
        base_agent=base_agent,
        llm_interface=llm_interface,
        query_budget=config.query_budget,
        blend_factor=config.blend_factor,
        confidence_threshold=config.confidence_threshold,
        verbose=False
    )


def run_variable_training(
    config: ExperimentConfig,
    robot: Union[DQNRobotAgent, HierarchicalDQNRobotAgent],
    property_counts: List[int],
    verbose: bool = False
) -> List[float]:
    """
    Run training where property count varies each episode.
    """
    rewards = []
    rng = np.random.RandomState(config.seed)

    for episode in tqdm(range(config.num_train_episodes)):
        num_props = rng.choice(property_counts)

        env = GridWorld(
            grid_size=config.grid_size,
            num_objects=config.num_objects,
            reward_ratio=config.reward_ratio,
            num_rewarding_properties=config.num_rewarding_properties,
            num_distinct_properties=num_props,
            num_property_values=config.num_property_values,
            seed=config.seed + episode
        )

        human = HumanAgent()
        result = run_episode(env, human, robot, training=True)
        rewards.append(result.robot_reward)

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
    """
    results = []

    # Store original epsilon and set to 0 for evaluation
    if hasattr(robot, 'epsilon'):
        original_epsilon = robot.epsilon
        robot.epsilon = 0.0
    elif hasattr(robot, 'base_agent'):
        original_epsilon = robot.base_agent.epsilon
        robot.base_agent.epsilon = 0.0
    else:
        original_epsilon = None

    for ep in range(num_episodes):
        env = GridWorld(
            grid_size=config.grid_size,
            num_objects=config.num_objects,
            reward_ratio=config.reward_ratio,
            num_rewarding_properties=config.num_rewarding_properties,
            num_distinct_properties=num_distinct_properties,
            num_property_values=config.num_property_values,
            seed=config.seed + 10000 + ep
        )

        human = HumanAgent()
        result = run_episode(env, human, robot, training=False)
        results.append(result)

    # Restore epsilon
    if original_epsilon is not None:
        if hasattr(robot, 'epsilon'):
            robot.epsilon = original_epsilon
        elif hasattr(robot, 'base_agent'):
            robot.base_agent.epsilon = original_epsilon

    return results


def run_variable_property_experiment(
    config: ExperimentConfig,
    property_counts: List[int] = None,
    num_seeds: int = 1,
    verbose: bool = True,
    api_key: Optional[str] = None
) -> Tuple[List[VariableExperimentResult], RobotAgent]:
    """
    Run experiment with variable property training and per-property evaluation.
    
    Args:
        config: Experiment configuration
        property_counts: List of property counts to use
        num_seeds: Number of random seeds
        verbose: Whether to print progress
        api_key: OpenAI API key for queries (optional)
    
    Returns:
        Tuple of (results list, last trained robot)
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
            reward_ratio=config.reward_ratio,
            num_rewarding_properties=config.num_rewarding_properties,
            num_property_values=config.num_property_values,
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
            allow_queries=config.allow_queries,
            query_budget=config.query_budget,
            confidence_threshold=config.confidence_threshold,
            blend_factor=config.blend_factor,
            llm_model=config.llm_model,
            seed=seed
        )

        env = GridWorld(
            grid_size=current_config.grid_size,
            num_objects=current_config.num_objects,
            reward_ratio=current_config.reward_ratio,
            num_rewarding_properties=current_config.num_rewarding_properties,
            num_distinct_properties=5,
            num_property_values=current_config.num_property_values,
            seed=seed
        )

        # Create base robot (must be hierarchical for queries)
        if current_config.allow_queries:
            # Force hierarchical for query support
            current_config.use_hierarchical = True
        
        base_robot = create_variable_robot_agent(current_config, env)

        if verbose:
            print("\nTraining with variable property counts...")

        train_rewards = run_variable_training(
            current_config,
            base_robot,
            property_counts,
            verbose=verbose
        )

        # Wrap with query augmentation if enabled
        if current_config.allow_queries and isinstance(base_robot, HierarchicalDQNRobotAgent):
            eval_robot = create_query_augmented_agent(base_robot, current_config, api_key)
            if verbose:
                print(f"\nQuery augmentation enabled:")
                print(f"  Budget: {current_config.query_budget}")
                print(f"  Confidence threshold: {current_config.confidence_threshold}")
                print(f"  Blend factor: {current_config.blend_factor}")
        else:
            eval_robot = base_robot

        # Evaluate
        result = VariableExperimentResult(
            train_rewards=train_rewards,
            train_mean=np.mean(train_rewards[-100:])
        )

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

            if verbose:
                query_info = f", avg queries={result.eval_avg_queries_per_property[num_props]:.2f}" if current_config.allow_queries else ""
                print(f"  {num_props} properties: "
                      f"mean={result.eval_means_per_property[num_props]:.2f}, "
                      f"std={result.eval_stds_per_property[num_props]:.2f}"
                      f"{query_info}")

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
        eval_means = [r.eval_means_per_property[num_props] for r in results]
        avg_queries = [r.eval_avg_queries_per_property.get(num_props, 0) for r in results]

        summary[num_props] = {
            'eval_mean': np.mean(eval_means),
            'eval_std': np.std(eval_means),
            'eval_sem': np.std(eval_means) / np.sqrt(len(eval_means)) if len(eval_means) > 1 else 0.0,
            'avg_queries': np.mean(avg_queries),
        }

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
    """Render and save a GIF of an entire episode."""
    import imageio

    if trained_robot is None:
        raise ValueError("Missing trained robot agent.")

    # Get base robot for rendering
    if isinstance(trained_robot, QueryAugmentedRobotAgent):
        robot = trained_robot
    else:
        robot = trained_robot

    np.random.seed(None)

    vis_env = GridWorld(
        grid_size=config.grid_size,
        num_objects=config.num_objects,
        reward_ratio=config.reward_ratio,
        num_rewarding_properties=config.num_rewarding_properties,
        num_distinct_properties=num_distinct_properties,
        num_property_values=config.num_property_values,
        seed=np.random.randint(0, 1000000)
    )

    human = HumanAgent()

    # Set evaluation mode
    if hasattr(robot, 'epsilon'):
        original_epsilon = robot.epsilon
        robot.epsilon = 0.0
    elif hasattr(robot, 'base_agent'):
        original_epsilon = robot.base_agent.epsilon
        robot.base_agent.epsilon = 0.0
    else:
        original_epsilon = None

    observation = vis_env.reset()
    human_obs = vis_env.get_human_observation()
    human.reset(human_obs['reward_properties'])
    robot.reset()

    # Set up human responder for query-augmented agent
    if isinstance(robot, QueryAugmentedRobotAgent):
        robot.set_human_responder(human_obs['reward_properties'])

    frames = []
    frames.append(vis_env.render_to_array())

    step = 0
    while not vis_env.done and step < max_steps:
        human_action = human.get_action(human_obs)
        robot_action = robot.get_action(observation, training=False)

        vis_env.human_position = vis_env._apply_action(vis_env.human_position, human_action)
        observation, _, _, _ = vis_env.step(robot_action)
        human_obs = vis_env.get_human_observation()

        frames.append(vis_env.render_to_array())
        step += 1

    # Restore epsilon
    if original_epsilon is not None:
        if hasattr(robot, 'epsilon'):
            robot.epsilon = original_epsilon
        elif hasattr(robot, 'base_agent'):
            robot.base_agent.epsilon = original_epsilon

    imageio.mimsave(output_path, frames, fps=fps, loop=0)
