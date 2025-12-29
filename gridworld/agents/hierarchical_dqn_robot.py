"""Hierarchical DQN Robot agent with goal-setting high-level and goal-conditioned low-level policies."""

from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple
import random
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ..objects import PROPERTY_CATEGORIES, PROPERTY_VALUES


class HierarchicalReplayBuffer:
    """Experience replay buffer for hierarchical DQN training."""

    def __init__(self, capacity: int = 100000):
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        goal: np.ndarray = None
    ):
        """Add a transition to the buffer."""
        self.buffer.append((state, action, reward, next_state, done, goal))

    def sample(self, batch_size: int) -> Tuple:
        """Sample a batch of transitions."""
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones, goals = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
            np.array(goals) if goals[0] is not None else None
        )

    def __len__(self):
        return len(self.buffer)


class HighLevelQNetwork(nn.Module):
    """Neural network for high-level goal-selection Q-function."""

    def __init__(
        self,
        input_dim: int,
        num_goals: int,
        hidden_dims: List[int] = None
    ):
        """
        Initialize high-level Q-network.

        Args:
            input_dim: Dimension of state features
            num_goals: Number of possible goals (e.g., K nearest objects)
            hidden_dims: List of hidden layer dimensions
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 128]

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_goals))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x)


class LowLevelQNetwork(nn.Module):
    """Neural network for low-level goal-conditioned action Q-function."""

    def __init__(
        self,
        state_dim: int,
        goal_dim: int,
        num_actions: int,
        hidden_dims: List[int] = None
    ):
        """
        Initialize low-level goal-conditioned Q-network.

        Args:
            state_dim: Dimension of state features
            goal_dim: Dimension of goal features
            num_actions: Number of possible actions (5: up, down, left, right, stay)
            hidden_dims: List of hidden layer dimensions
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 128]

        input_dim = state_dim + goal_dim
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_actions))

        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """Forward pass with state and goal concatenation."""
        x = torch.cat([state, goal], dim=-1)
        return self.network(x)


class HierarchicalDQNRobotAgent:
    """
    A hierarchical robot agent with:
    - High-level policy: Selects which object to collect (goal)
    - Low-level policy: Navigates to the selected goal object

    The high-level policy operates at a slower timescale (every H steps or when goal is reached).
    The low-level policy operates every step, conditioned on the current goal.
    """

    def __init__(
        self,
        num_actions: int = 5,
        num_goal_candidates: int = 5,
        high_level_interval: int = 5,
        learning_rate: float = 1e-4,
        discount_factor: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        exploration_fraction: float = 0.1,
        total_timesteps: int = 100000,
        buffer_size: int = 100000,
        batch_size: int = 32,
        target_update_freq: int = 10000,
        train_freq: int = 4,
        gradient_steps: int = 1,
        learning_starts: int = 1000,
        hidden_dims: List[int] = None,
        grid_size: int = 10,
        active_categories: List[str] = None,
        seed: Optional[int] = None,
        device: str = None
    ):
        """
        Initialize the hierarchical DQN robot agent.

        Args:
            num_actions: Number of low-level actions (5: up, down, left, right, stay)
            num_goal_candidates: Number of candidate goals (K nearest objects)
            high_level_interval: Steps between high-level decisions (H)
            learning_rate: Learning rate for optimizers
            discount_factor: Discount factor (gamma)
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            exploration_fraction: Fraction of training for epsilon decay
            total_timesteps: Total timesteps for exploration schedule
            buffer_size: Size of replay buffers
            batch_size: Batch size for training
            target_update_freq: How often to update target networks
            train_freq: How often to train
            gradient_steps: Gradient updates per train call
            learning_starts: Steps before training starts
            hidden_dims: Hidden layer dimensions
            grid_size: Size of the grid
            active_categories: List of active property categories
            seed: Random seed
            device: Device to use ('cpu' or 'cuda')
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for DQN. Install with: pip install torch")

        self.num_actions = num_actions
        self.num_goal_candidates = num_goal_candidates
        self.high_level_interval = high_level_interval
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.exploration_fraction = exploration_fraction
        self.total_timesteps = total_timesteps
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.train_freq = train_freq
        self.gradient_steps = gradient_steps
        self.learning_starts = learning_starts
        self.grid_size = grid_size
        self.hidden_dims = hidden_dims or [128, 128]

        # Current exploration rates (separate for high and low level)
        self.epsilon_high = epsilon_start
        self.epsilon_low = epsilon_start

        # Set active categories
        self.active_categories = active_categories or PROPERTY_CATEGORIES[:2]

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Random generators
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)

        # Build property value indices for encoding
        self._build_property_indices()

        # Calculate input dimensions
        self.state_dim = self._calculate_state_dim()
        self.goal_dim = self._calculate_goal_dim()
        self.high_level_input_dim = self._calculate_high_level_input_dim()

        # Initialize high-level networks (goal selection)
        self.high_level_q_network = HighLevelQNetwork(
            self.high_level_input_dim, num_goal_candidates, self.hidden_dims
        ).to(self.device)

        self.high_level_target_network = HighLevelQNetwork(
            self.high_level_input_dim, num_goal_candidates, self.hidden_dims
        ).to(self.device)
        self.high_level_target_network.load_state_dict(self.high_level_q_network.state_dict())
        self.high_level_target_network.eval()

        # Initialize low-level networks (goal-conditioned navigation)
        self.low_level_q_network = LowLevelQNetwork(
            self.state_dim, self.goal_dim, num_actions, self.hidden_dims
        ).to(self.device)

        self.low_level_target_network = LowLevelQNetwork(
            self.state_dim, self.goal_dim, num_actions, self.hidden_dims
        ).to(self.device)
        self.low_level_target_network.load_state_dict(self.low_level_q_network.state_dict())
        self.low_level_target_network.eval()

        # Optimizers
        self.high_level_optimizer = optim.Adam(
            self.high_level_q_network.parameters(), lr=learning_rate
        )
        self.low_level_optimizer = optim.Adam(
            self.low_level_q_network.parameters(), lr=learning_rate
        )

        # Replay buffers (separate for high and low level)
        self.high_level_buffer = HierarchicalReplayBuffer(buffer_size)
        self.low_level_buffer = HierarchicalReplayBuffer(buffer_size)

        # Track inferred reward properties (same as flat DQN)
        self.inferred_properties: Dict[str, float] = defaultdict(float)
        self.property_counts: Dict[str, int] = defaultdict(int)

        # Hierarchical state tracking
        self.current_goal_idx: Optional[int] = None
        self.current_goal_features: Optional[np.ndarray] = None
        self.current_goal_object_id: Optional[int] = None
        self.steps_since_goal_selection: int = 0
        self.high_level_state: Optional[np.ndarray] = None
        self.cumulative_low_level_reward: float = 0.0

        # Training statistics
        self.total_reward = 0.0
        self.steps = 0
        self.total_steps = 0
        self.high_level_train_losses = []
        self.low_level_train_losses = []
        self._n_updates = 0

    def _build_property_indices(self):
        """Build indices for one-hot encoding property values."""
        self.property_to_idx = {}
        idx = 0
        for category in PROPERTY_CATEGORIES:
            for value in PROPERTY_VALUES[category]:
                self.property_to_idx[value] = idx
                idx += 1
        self.total_property_values = idx

    def _calculate_state_dim(self) -> int:
        """
        Calculate dimension of state features for low-level policy.

        Features:
        - Robot position (2)
        - Human position (2)
        - Relative position of human (2)
        - Human collected properties (one-hot)
        - Inferred property scores
        """
        pos_features = 6
        collected_features = self.total_property_values
        props_dim = sum(len(PROPERTY_VALUES[cat]) for cat in self.active_categories)
        inferred_features = props_dim
        return pos_features + collected_features + inferred_features

    def _calculate_goal_dim(self) -> int:
        """
        Calculate dimension of goal features.

        Goal features:
        - Relative position to goal object (2)
        - Distance to goal (1)
        - Goal object properties (one-hot for active categories)
        """
        props_dim = sum(len(PROPERTY_VALUES[cat]) for cat in self.active_categories)
        return 3 + props_dim  # rel_pos (2) + distance (1) + properties

    def _calculate_high_level_input_dim(self) -> int:
        """
        Calculate dimension of high-level policy input.

        Features:
        - State features (same as low-level state)
        - For each of K candidate objects:
            - Relative position (2)
            - Distance (1)
            - Properties (one-hot)
        """
        props_dim = sum(len(PROPERTY_VALUES[cat]) for cat in self.active_categories)
        per_object_features = 3 + props_dim
        return self.state_dim + self.num_goal_candidates * per_object_features

    def _encode_state(self, observation: dict) -> np.ndarray:
        """
        Encode observation into state features for low-level policy.
        """
        features = []

        # Robot position (normalized)
        robot_pos = observation['robot_position']
        features.append(robot_pos[0] / self.grid_size)
        features.append(robot_pos[1] / self.grid_size)

        # Human position (normalized)
        human_pos = observation['human_position']
        features.append(human_pos[0] / self.grid_size)
        features.append(human_pos[1] / self.grid_size)

        # Relative position of human to robot
        rel_x = (human_pos[0] - robot_pos[0]) / self.grid_size
        rel_y = (human_pos[1] - robot_pos[1]) / self.grid_size
        features.append(rel_x)
        features.append(rel_y)

        # Human collected properties (one-hot)
        collected_props = np.zeros(self.total_property_values)
        for collected in observation.get('human_collected', []):
            for prop_value in collected['properties'].values():
                if prop_value in self.property_to_idx:
                    collected_props[self.property_to_idx[prop_value]] = 1.0
        features.extend(collected_props.tolist())

        # Inferred property scores (normalized)
        max_score = max(self.inferred_properties.values()) if self.inferred_properties else 1.0
        max_score = max(max_score, 1.0)

        for cat in self.active_categories:
            for value in PROPERTY_VALUES[cat]:
                score = self.inferred_properties.get(value, 0.0)
                features.append(score / max_score)

        return np.array(features, dtype=np.float32)

    def _encode_goal(self, observation: dict, goal_object_data: dict) -> np.ndarray:
        """
        Encode goal object into goal features.

        Args:
            observation: Current observation
            goal_object_data: Data for the goal object (position, properties)

        Returns:
            Goal feature vector
        """
        features = []
        robot_pos = observation['robot_position']
        goal_pos = goal_object_data['position']

        # Relative position (normalized)
        rel_x = (goal_pos[0] - robot_pos[0]) / self.grid_size
        rel_y = (goal_pos[1] - robot_pos[1]) / self.grid_size
        features.append(rel_x)
        features.append(rel_y)

        # Distance (normalized)
        dist = np.sqrt((goal_pos[0] - robot_pos[0])**2 + (goal_pos[1] - robot_pos[1])**2)
        max_dist = np.sqrt(2) * self.grid_size
        features.append(dist / max_dist)

        # Object properties (one-hot for active categories)
        props = goal_object_data['properties']
        for cat in self.active_categories:
            for value in PROPERTY_VALUES[cat]:
                if props.get(cat) == value:
                    features.append(1.0)
                else:
                    features.append(0.0)

        return np.array(features, dtype=np.float32)

    def _get_goal_candidates(self, observation: dict) -> List[Tuple[int, dict]]:
        """
        Get K nearest objects as goal candidates.

        Returns:
            List of (object_id, object_data) tuples for nearest objects
        """
        robot_pos = observation['robot_position']
        objects = observation.get('objects', {})

        if not objects:
            return []

        # Calculate distances and sort
        object_list = []
        for obj_id, obj_data in objects.items():
            pos = obj_data['position']
            dist = np.sqrt((pos[0] - robot_pos[0])**2 + (pos[1] - robot_pos[1])**2)
            object_list.append((dist, obj_id, obj_data))

        object_list.sort(key=lambda x: x[0])

        # Return K nearest
        candidates = []
        for i in range(min(self.num_goal_candidates, len(object_list))):
            _, obj_id, obj_data = object_list[i]
            candidates.append((obj_id, obj_data))

        return candidates

    def _encode_high_level_input(self, observation: dict) -> np.ndarray:
        """
        Encode observation for high-level policy.
        Includes state features and features for each goal candidate.
        """
        # Base state features
        state_features = self._encode_state(observation)

        # Get goal candidates
        candidates = self._get_goal_candidates(observation)

        # Encode each candidate
        candidate_features = []
        props_dim = sum(len(PROPERTY_VALUES[cat]) for cat in self.active_categories)
        per_object_dim = 3 + props_dim

        for i in range(self.num_goal_candidates):
            if i < len(candidates):
                _, obj_data = candidates[i]
                goal_features = self._encode_goal(observation, obj_data)
                candidate_features.extend(goal_features.tolist())
            else:
                # Pad with zeros if fewer candidates
                candidate_features.extend([0.0] * per_object_dim)

        return np.concatenate([state_features, np.array(candidate_features, dtype=np.float32)])

    def _get_exploration_rate(self) -> float:
        """Compute exploration rate using linear schedule."""
        progress = self.total_steps / self.total_timesteps
        exploration_progress = min(1.0, progress / self.exploration_fraction)
        return self.epsilon_start + exploration_progress * (self.epsilon_end - self.epsilon_start)

    def reset(self):
        """Reset episode-specific state."""
        self.inferred_properties = defaultdict(float)
        self.property_counts = defaultdict(int)
        self.total_reward = 0.0
        self.steps = 0

        # Reset hierarchical state
        self.current_goal_idx = None
        self.current_goal_features = None
        self.current_goal_object_id = None
        self.steps_since_goal_selection = 0
        self.high_level_state = None
        self.cumulative_low_level_reward = 0.0

    def reset_learning(self):
        """Fully reset the agent including learned parameters."""
        # Reinitialize high-level networks
        self.high_level_q_network = HighLevelQNetwork(
            self.high_level_input_dim, self.num_goal_candidates, self.hidden_dims
        ).to(self.device)
        self.high_level_target_network = HighLevelQNetwork(
            self.high_level_input_dim, self.num_goal_candidates, self.hidden_dims
        ).to(self.device)
        self.high_level_target_network.load_state_dict(self.high_level_q_network.state_dict())
        self.high_level_target_network.eval()

        # Reinitialize low-level networks
        self.low_level_q_network = LowLevelQNetwork(
            self.state_dim, self.goal_dim, self.num_actions, self.hidden_dims
        ).to(self.device)
        self.low_level_target_network = LowLevelQNetwork(
            self.state_dim, self.goal_dim, self.num_actions, self.hidden_dims
        ).to(self.device)
        self.low_level_target_network.load_state_dict(self.low_level_q_network.state_dict())
        self.low_level_target_network.eval()

        # Reinitialize optimizers
        self.high_level_optimizer = optim.Adam(
            self.high_level_q_network.parameters(), lr=self.learning_rate
        )
        self.low_level_optimizer = optim.Adam(
            self.low_level_q_network.parameters(), lr=self.learning_rate
        )

        # Reset buffers
        self.high_level_buffer = HierarchicalReplayBuffer(self.high_level_buffer.buffer.maxlen)
        self.low_level_buffer = HierarchicalReplayBuffer(self.low_level_buffer.buffer.maxlen)

        self.epsilon_high = self.epsilon_start
        self.epsilon_low = self.epsilon_start
        self.total_steps = 0
        self._n_updates = 0
        self.high_level_train_losses = []
        self.low_level_train_losses = []
        self.reset()

    def _should_select_new_goal(self, observation: dict) -> bool:
        """
        Determine if high-level policy should select a new goal.

        Returns True if:
        - No current goal
        - High-level interval has passed
        - Current goal object no longer exists (collected or gone)
        """
        if self.current_goal_idx is None:
            return True

        if self.steps_since_goal_selection >= self.high_level_interval:
            return True

        # Check if current goal object still exists
        if self.current_goal_object_id is not None:
            objects = observation.get('objects', {})
            if self.current_goal_object_id not in objects:
                return True

        return False

    def _select_goal(self, observation: dict, training: bool = True) -> int:
        """
        High-level policy: Select which object to collect.

        Args:
            observation: Current observation
            training: Whether in training mode

        Returns:
            Goal index (0 to num_goal_candidates-1)
        """
        high_level_input = self._encode_high_level_input(observation)
        candidates = self._get_goal_candidates(observation)
        num_valid = len(candidates)

        if num_valid == 0:
            return 0  # No objects available

        # Epsilon-greedy for high-level
        if training and self.rng.random() < self.epsilon_high:
            goal_idx = self.rng.randint(0, num_valid - 1)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(high_level_input).unsqueeze(0).to(self.device)
                q_values = self.high_level_q_network(state_tensor)
                # Mask invalid goals
                q_values_np = q_values.cpu().numpy()[0]
                q_values_np[num_valid:] = -float('inf')
                goal_idx = np.argmax(q_values_np)

        return goal_idx

    def get_action(self, observation: dict, training: bool = True) -> int:
        """
        Get action using hierarchical policy.

        Args:
            observation: Current observation
            training: Whether in training mode

        Returns:
            Action (0=up, 1=down, 2=left, 3=right, 4=stay)
        """
        # Update property inference
        self._update_inference(observation)

        # Check if we need to select a new goal
        if self._should_select_new_goal(observation):
            # Store previous high-level state for learning
            if training and self.high_level_state is not None and self.current_goal_idx is not None:
                # Store high-level transition
                new_high_level_state = self._encode_high_level_input(observation)
                self.high_level_buffer.push(
                    self.high_level_state,
                    self.current_goal_idx,
                    self.cumulative_low_level_reward,
                    new_high_level_state,
                    False,  # Not terminal for high-level
                    None
                )

            # Select new goal
            self.current_goal_idx = self._select_goal(observation, training)
            self.high_level_state = self._encode_high_level_input(observation)
            self.steps_since_goal_selection = 0
            self.cumulative_low_level_reward = 0.0

            # Update goal features
            candidates = self._get_goal_candidates(observation)
            if self.current_goal_idx < len(candidates):
                obj_id, obj_data = candidates[self.current_goal_idx]
                self.current_goal_object_id = obj_id
                self.current_goal_features = self._encode_goal(observation, obj_data)
            else:
                self.current_goal_object_id = None
                self.current_goal_features = np.zeros(self.goal_dim, dtype=np.float32)

        # Update goal features (position may have changed even if same goal)
        if self.current_goal_object_id is not None:
            objects = observation.get('objects', {})
            if self.current_goal_object_id in objects:
                obj_data = objects[self.current_goal_object_id]
                self.current_goal_features = self._encode_goal(observation, obj_data)

        # Low-level policy: Get action conditioned on goal
        state_features = self._encode_state(observation)

        # Epsilon-greedy for low-level
        if training and self.rng.random() < self.epsilon_low:
            action = self.rng.randint(0, self.num_actions - 1)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state_features).unsqueeze(0).to(self.device)
                goal_tensor = torch.FloatTensor(self.current_goal_features).unsqueeze(0).to(self.device)
                q_values = self.low_level_q_network(state_tensor, goal_tensor)
                action = q_values.argmax(dim=1).item()

        return action

    def _update_inference(self, observation: dict):
        """Update inferred reward properties based on human's collections."""
        human_collected = observation.get('human_collected', [])

        for collected in human_collected:
            props = collected['properties']
            obj_id = collected['id']

            if obj_id not in self.property_counts:
                for prop_value in props.values():
                    self.inferred_properties[prop_value] += 1.0
                self.property_counts[obj_id] = 1

    def _compute_intrinsic_reward(
        self,
        observation: dict,
        next_observation: dict,
        extrinsic_reward: float
    ) -> float:
        """
        Compute intrinsic reward for low-level policy.

        The low-level policy gets rewarded for:
        - Getting closer to the goal object
        - Reaching the goal object
        """
        if self.current_goal_object_id is None:
            return extrinsic_reward

        robot_pos = observation['robot_position']
        next_robot_pos = next_observation['robot_position']
        objects = observation.get('objects', {})
        next_objects = next_observation.get('objects', {})

        # Check if goal was collected
        if self.current_goal_object_id in objects and self.current_goal_object_id not in next_objects:
            # Goal was collected (by us or human)
            if next_robot_pos == objects[self.current_goal_object_id]['position']:
                # We collected it - give bonus
                return extrinsic_reward + 0.5
            else:
                # Human collected it
                return extrinsic_reward

        # Goal still exists - reward for getting closer
        if self.current_goal_object_id in objects:
            goal_pos = objects[self.current_goal_object_id]['position']
            prev_dist = np.sqrt((robot_pos[0] - goal_pos[0])**2 + (robot_pos[1] - goal_pos[1])**2)
            curr_dist = np.sqrt((next_robot_pos[0] - goal_pos[0])**2 + (next_robot_pos[1] - goal_pos[1])**2)

            # Small reward for getting closer
            distance_reward = (prev_dist - curr_dist) / self.grid_size * 0.1
            return extrinsic_reward + distance_reward

        return extrinsic_reward

    def update(
        self,
        action: int,
        reward: float,
        done: bool,
        observation: dict,
        next_observation: dict
    ):
        """
        Update both high-level and low-level Q-networks.
        """
        self.total_reward += reward
        self.steps += 1
        self.total_steps += 1
        self.steps_since_goal_selection += 1

        # Update exploration rates
        epsilon = self._get_exploration_rate()
        self.epsilon_high = epsilon
        self.epsilon_low = epsilon

        # Compute intrinsic reward for low-level
        intrinsic_reward = self._compute_intrinsic_reward(observation, next_observation, reward)

        # Accumulate reward for high-level
        self.cumulative_low_level_reward += reward

        # Store low-level transition
        state_features = self._encode_state(observation)
        next_state_features = self._encode_state(next_observation)

        self.low_level_buffer.push(
            state_features,
            action,
            intrinsic_reward,
            next_state_features,
            done,
            self.current_goal_features.copy() if self.current_goal_features is not None else None
        )

        # Store high-level transition if episode is done
        if done and self.high_level_state is not None and self.current_goal_idx is not None:
            new_high_level_state = self._encode_high_level_input(next_observation)
            self.high_level_buffer.push(
                self.high_level_state,
                self.current_goal_idx,
                self.cumulative_low_level_reward,
                new_high_level_state,
                True,
                None
            )

        # Training
        if len(self.low_level_buffer) < self.learning_starts:
            return

        if self.total_steps % self.train_freq == 0:
            for _ in range(self.gradient_steps):
                self._train_low_level_step()

                # Train high-level less frequently
                if len(self.high_level_buffer) >= self.batch_size:
                    self._train_high_level_step()

        # Update target networks
        if self.total_steps % self.target_update_freq == 0:
            self.low_level_target_network.load_state_dict(self.low_level_q_network.state_dict())
            self.high_level_target_network.load_state_dict(self.high_level_q_network.state_dict())

    def _train_low_level_step(self):
        """Train low-level goal-conditioned policy."""
        states, actions, rewards, next_states, dones, goals = self.low_level_buffer.sample(
            self.batch_size
        )

        if goals is None:
            return

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        goals = torch.FloatTensor(goals).to(self.device)

        # Current Q-values
        current_q = self.low_level_q_network(states, goals).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values
        with torch.no_grad():
            next_q = self.low_level_target_network(next_states, goals).max(dim=1)[0]
            target_q = rewards + (1 - dones) * self.discount_factor * next_q

        # Loss
        loss = F.smooth_l1_loss(current_q, target_q)

        # Optimize
        self.low_level_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.low_level_q_network.parameters(), 10.0)
        self.low_level_optimizer.step()

        self.low_level_train_losses.append(loss.item())

    def _train_high_level_step(self):
        """Train high-level goal-selection policy."""
        states, actions, rewards, next_states, dones, _ = self.high_level_buffer.sample(
            self.batch_size
        )

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Current Q-values
        current_q = self.high_level_q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values
        with torch.no_grad():
            next_q = self.high_level_target_network(next_states).max(dim=1)[0]
            target_q = rewards + (1 - dones) * self.discount_factor * next_q

        # Loss
        loss = F.smooth_l1_loss(current_q, target_q)

        # Optimize
        self.high_level_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.high_level_q_network.parameters(), 10.0)
        self.high_level_optimizer.step()

        self.high_level_train_losses.append(loss.item())
        self._n_updates += 1

    def decay_epsilon(self):
        """Deprecated: Epsilon decay is handled in update()."""
        pass

    def get_inferred_properties(self) -> List[Tuple[str, float]]:
        """Get the inferred reward properties sorted by confidence."""
        return sorted(
            self.inferred_properties.items(),
            key=lambda x: x[1],
            reverse=True
        )

    def get_average_loss(self, last_n: int = 100) -> float:
        """Get average training loss over last n steps."""
        low_losses = self.low_level_train_losses[-last_n:] if self.low_level_train_losses else []
        high_losses = self.high_level_train_losses[-last_n:] if self.high_level_train_losses else []

        all_losses = low_losses + high_losses
        if not all_losses:
            return 0.0
        return sum(all_losses) / len(all_losses)

    @property
    def epsilon(self) -> float:
        """Return current epsilon (for compatibility)."""
        return self.epsilon_low

    @epsilon.setter
    def epsilon(self, value: float):
        """Set epsilon for both levels (for compatibility)."""
        self.epsilon_low = value
        self.epsilon_high = value

    def save(self, path: str):
        """Save the model to a file."""
        torch.save({
            'high_level_q_network': self.high_level_q_network.state_dict(),
            'high_level_target_network': self.high_level_target_network.state_dict(),
            'low_level_q_network': self.low_level_q_network.state_dict(),
            'low_level_target_network': self.low_level_target_network.state_dict(),
            'high_level_optimizer': self.high_level_optimizer.state_dict(),
            'low_level_optimizer': self.low_level_optimizer.state_dict(),
            'epsilon_high': self.epsilon_high,
            'epsilon_low': self.epsilon_low,
            'total_steps': self.total_steps,
            'config': {
                'num_actions': self.num_actions,
                'num_goal_candidates': self.num_goal_candidates,
                'high_level_interval': self.high_level_interval,
                'state_dim': self.state_dim,
                'goal_dim': self.goal_dim,
                'high_level_input_dim': self.high_level_input_dim,
                'hidden_dims': self.hidden_dims,
                'active_categories': self.active_categories,
                'grid_size': self.grid_size
            }
        }, path)

    def load(self, path: str):
        """Load the model from a file."""
        checkpoint = torch.load(path, map_location=self.device)
        self.high_level_q_network.load_state_dict(checkpoint['high_level_q_network'])
        self.high_level_target_network.load_state_dict(checkpoint['high_level_target_network'])
        self.low_level_q_network.load_state_dict(checkpoint['low_level_q_network'])
        self.low_level_target_network.load_state_dict(checkpoint['low_level_target_network'])
        self.high_level_optimizer.load_state_dict(checkpoint['high_level_optimizer'])
        self.low_level_optimizer.load_state_dict(checkpoint['low_level_optimizer'])
        self.epsilon_high = checkpoint['epsilon_high']
        self.epsilon_low = checkpoint['epsilon_low']
        self.total_steps = checkpoint['total_steps']
