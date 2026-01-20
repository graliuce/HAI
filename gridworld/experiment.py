"""Experiment runner for the gridworld multi-agent task."""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import os
from dataclasses import dataclass, field

from .environment import GridWorld
from .agents.human import HumanAgent
from .agents.hierarchical_dqn_robot import HierarchicalDQNRobotAgent
from .agents.query_augmented_robot import QueryAugmentedRobotAgent
from .agents.belief_based_robot import BeliefBasedRobotAgent
from .objects import PROPERTY_CATEGORIES
from tqdm import tqdm

# Type alias for robot agents
RobotAgent = Union[HierarchicalDQNRobotAgent, QueryAugmentedRobotAgent, BeliefBasedRobotAgent]


@dataclass
class ExperimentConfig:
    """Configuration for the experiment."""

    # Environment parameters
    grid_size: int = 10
    num_objects: int = 20
    reward_ratio: float = 0.4
    num_rewarding_properties: int = 2
    num_property_values: int = 5
    additive_valuation: bool = False  # Use additive valuation reward mode

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

    # Query parameters (for test time)
    allow_queries: bool = False
    query_budget: int = 5
    query_threshold: float = 0.8
    blend_factor: float = 0.5
    llm_model: str = "gpt-4o-mini"

    # Belief-based agent parameters
    use_belief_based_agent: bool = False
    participation_ratio_threshold: float = 3.0  # Deprecated, use action_confidence_threshold
    action_confidence_threshold: float = 0.6  # Query when confidence < this (0-1, higher = query more often)
    plackett_luce_learning_rate: float = 0.1
    plackett_luce_gradient_steps: int = 5
    plackett_luce_info_gain: float = 0.5
    linear_gaussian_noise_variance: float = 1.0

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
    # Entropy statistics
    eval_entropy_stats_per_property: Dict[int, Dict] = field(default_factory=dict)


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
    human.reset(
        human_obs['reward_properties'],
        additive_valuation=human_obs.get('additive_valuation', False),
        object_rewards=human_obs.get('object_rewards')
    )
    result.reward_properties = human_obs['reward_properties']

    # Reset robot - for belief-based agent, pass active categories so it
    # initializes the correct dimensionality belief state
    if isinstance(robot, BeliefBasedRobotAgent):
        robot.reset(active_categories=observation.get('active_categories'))
    else:
        robot.reset()

    # If using query-augmented agent or belief-based agent, set up human responder
    if isinstance(robot, QueryAugmentedRobotAgent) and not training:
        robot.set_human_responder(
            human_obs['reward_properties'],
            property_value_rewards=human_obs.get('property_value_rewards')
        )
    elif isinstance(robot, BeliefBasedRobotAgent) and not training:
        robot.set_human_responder(
            human_obs['reward_properties'],
            property_value_rewards=human_obs.get('property_value_rewards')
        )

    # For verbose output during evaluation, set reward properties on base agent
    if not training and hasattr(robot, 'set_reward_properties_for_verbose'):
        robot.set_reward_properties_for_verbose(human_obs['reward_properties'])
    elif not training and isinstance(robot, QueryAugmentedRobotAgent) and hasattr(robot.base_agent, 'set_reward_properties_for_verbose'):
        robot.base_agent.set_reward_properties_for_verbose(human_obs['reward_properties'])

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
        result.entropy_history = stats.get('entropy_history', [])
    elif isinstance(robot, BeliefBasedRobotAgent):
        stats = robot.get_query_stats()
        result.queries_used = stats['queries_used']
        result.query_history = stats['query_history']
        result.entropy_history = stats.get('entropy_history', [])

    return result


def create_robot_agent(
    config: ExperimentConfig,
    env: GridWorld,
    verbose: bool = False
) -> HierarchicalDQNRobotAgent:
    """Create a hierarchical robot agent that can handle all property counts."""
    all_categories = PROPERTY_CATEGORIES[:5]
    total_timesteps = config.num_train_episodes * config.max_steps_per_episode

    return HierarchicalDQNRobotAgent(
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
        num_objects=config.num_objects,
        active_categories=all_categories,
        seed=config.seed,
        verbose=verbose
    )


def get_model_path(config: ExperimentConfig, model_dir: str) -> str:
    """
    Generate a model filename based on the experiment configuration.

    Args:
        config: Experiment configuration
        model_dir: Directory to save the model

    Returns:
        Full path to the model file
    """
    model_name = (
        f"model_hierarchical_"
        f"grid{config.grid_size}_"
        f"obj{config.num_objects}_"
        f"k{config.num_rewarding_properties}_"
        f"vals{config.num_property_values}_"
        f"eps{config.num_train_episodes}_"
        f"lr{config.learning_rate:.0e}_"
        f"seed{config.seed}.pt"
    )
    return os.path.join(model_dir, model_name)


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
        participation_ratio_threshold=config.participation_ratio_threshold,
        action_confidence_threshold=config.action_confidence_threshold,
        plackett_luce_learning_rate=config.plackett_luce_learning_rate,
        plackett_luce_gradient_steps=config.plackett_luce_gradient_steps,
        plackett_luce_info_gain=config.plackett_luce_info_gain,
        linear_gaussian_noise_variance=config.linear_gaussian_noise_variance,
        verbose=verbose,
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
        query_threshold=config.query_threshold,
        verbose=True
    )


def run_training(
    config: ExperimentConfig,
    robot: HierarchicalDQNRobotAgent,
    property_counts: List[int],
    verbose: bool = False
) -> List[float]:
    """Run training where property count varies each episode."""
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
            additive_valuation=config.additive_valuation,
            seed=config.seed + episode
        )

        human = HumanAgent()
        result = run_episode(env, human, robot, training=True)
        rewards.append(result.robot_reward)

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
            additive_valuation=config.additive_valuation,
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
    api_key: Optional[str] = None,
    model_dir: Optional[str] = None
) -> Tuple[List[VariableExperimentResult], RobotAgent]:
    """
    Run experiment with variable property training and per-property evaluation.
    
    Args:
        config: Experiment configuration
        property_counts: List of property counts to use
        num_seeds: Number of random seeds
        verbose: Whether to print progress
        api_key: OpenAI API key for queries (optional)
        model_dir: Directory to save/load models (if None, models won't be saved/loaded)
    
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
            additive_valuation=config.additive_valuation,
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
            allow_queries=config.allow_queries,
            query_budget=config.query_budget,
            query_threshold=config.query_threshold,
            blend_factor=config.blend_factor,
            llm_model=config.llm_model,
            use_belief_based_agent=config.use_belief_based_agent,
            participation_ratio_threshold=config.participation_ratio_threshold,
            action_confidence_threshold=config.action_confidence_threshold,
            plackett_luce_learning_rate=config.plackett_luce_learning_rate,
            plackett_luce_gradient_steps=config.plackett_luce_gradient_steps,
            plackett_luce_info_gain=config.plackett_luce_info_gain,
            linear_gaussian_noise_variance=config.linear_gaussian_noise_variance,
            seed=seed
        )

        env = GridWorld(
            grid_size=current_config.grid_size,
            num_objects=current_config.num_objects,
            reward_ratio=current_config.reward_ratio,
            num_rewarding_properties=current_config.num_rewarding_properties,
            num_distinct_properties=5,
            num_property_values=current_config.num_property_values,
            additive_valuation=current_config.additive_valuation,
            seed=seed
        )

        # Use belief-based agent if configured (no training needed)
        if current_config.use_belief_based_agent:
            if verbose:
                print("\nUsing Belief-Based Agent (no training phase)")
                print(f"  Action confidence threshold: {current_config.action_confidence_threshold}")
                print(f"  Plackett-Luce learning rate: {current_config.plackett_luce_learning_rate}")
                print(f"  Plackett-Luce info gain: {current_config.plackett_luce_info_gain}")
                print(f"  Linear-Gaussian noise variance: {current_config.linear_gaussian_noise_variance}")
                if current_config.allow_queries:
                    print(f"  Query budget: {current_config.query_budget}")

            eval_robot = create_belief_based_agent(
                current_config, env, api_key, verbose=True
            )
            # No training for belief-based agent - dummy rewards
            train_rewards = [0.0] * current_config.num_train_episodes

        else:
            # DQN-based agent with training
            base_robot = create_robot_agent(current_config, env, verbose=True)

            # Check if model exists and should be loaded
            model_exists = False
            model_path = None
            if model_dir is not None:
                os.makedirs(model_dir, exist_ok=True)
                model_path = get_model_path(current_config, model_dir)
                model_exists = os.path.exists(model_path)

            if model_exists:
                if verbose:
                    print(f"\nLoading existing model from: {model_path}")
                base_robot.load(model_path)
                # Create dummy training rewards for compatibility
                train_rewards = [0.0] * current_config.num_train_episodes
            else:
                if verbose:
                    if model_path:
                        print(f"\nNo existing model found at: {model_path}")
                    print("Training with variable property counts...")

                train_rewards = run_training(
                    current_config,
                    base_robot,
                    property_counts,
                    verbose=verbose
                )

                # Save the trained model
                if model_path is not None:
                    if verbose:
                        print(f"\nSaving trained model to: {model_path}")
                    base_robot.save(model_path)

            # Wrap with query augmentation if enabled
            if current_config.allow_queries:
                eval_robot = create_query_augmented_agent(base_robot, current_config, api_key)
                if verbose:
                    print(f"\nQuery augmentation enabled:")
                    print(f"  Budget: {current_config.query_budget}")
                    print(f"  Query threshold: {current_config.query_threshold}")
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
            
            # Collect entropy/uncertainty statistics
            if current_config.allow_queries or current_config.use_belief_based_agent:
                all_entropy_history = []
                for r in eval_results:
                    all_entropy_history.extend(r.entropy_history)

                if all_entropy_history:
                    num_decision_points = len(all_entropy_history)
                    num_triggered = sum(1 for h in all_entropy_history if h['should_query'])

                    # Handle both formats: QueryAugmented uses 'num_competitive',
                    # BeliefBased uses 'participation_ratio'
                    if 'num_competitive' in all_entropy_history[0]:
                        # Query-augmented agent format
                        num_competitive_values = [h['num_competitive'] for h in all_entropy_history]
                        result.eval_entropy_stats_per_property[num_props] = {
                            'mean_num_competitive': np.mean(num_competitive_values),
                            'std_num_competitive': np.std(num_competitive_values),
                            'num_decision_points': num_decision_points,
                            'num_triggered': num_triggered,
                            'trigger_rate': num_triggered / num_decision_points if num_decision_points > 0 else 0,
                            'sample_decision_points': all_entropy_history[:5]
                        }
                    elif 'participation_ratio' in all_entropy_history[0]:
                        # Belief-based agent format
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
        additive_valuation=config.additive_valuation,
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
    human.reset(
        human_obs['reward_properties'],
        additive_valuation=human_obs.get('additive_valuation', False),
        object_rewards=human_obs.get('object_rewards')
    )
    robot.reset()

    # Set up human responder for query-augmented agent
    if isinstance(robot, QueryAugmentedRobotAgent):
        robot.set_human_responder(
            human_obs['reward_properties'],
            property_value_rewards=human_obs.get('property_value_rewards')
        )
        # Enable verbose mode for detailed logging during GIF generation
        if config.allow_queries:
            robot.verbose = True

    frames = []
    
    # Track query information for display
    current_query_info = None
    prev_query_count = 0
    
    frames.append(vis_env.render_to_array(query_info=current_query_info))

    step = 0
    while not vis_env.done and step < max_steps:
        human_action = human.get_action(human_obs)
        robot_action = robot.get_action(observation, training=False)
        
        # If a new query was made, update the display info
        if isinstance(robot, QueryAugmentedRobotAgent) and robot.queries_used > prev_query_count:
            prev_query_count = robot.queries_used
            if robot.query_history:
                last_query = robot.query_history[-1]
                current_query_info = {
                    'weights': last_query['weights'],
                    'query_num': robot.queries_used
                }

        vis_env.human_position = vis_env._apply_action(vis_env.human_position, human_action)
        observation, _, _, _ = vis_env.step(robot_action)
        human_obs = vis_env.get_human_observation()

        frames.append(vis_env.render_to_array(query_info=current_query_info))
        step += 1

    # Restore epsilon
    if original_epsilon is not None:
        if hasattr(robot, 'epsilon'):
            robot.epsilon = original_epsilon
        elif hasattr(robot, 'base_agent'):
            robot.base_agent.epsilon = original_epsilon

    imageio.mimsave(output_path, frames, fps=fps, loop=0)
