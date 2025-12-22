"""DQN Robot agent using deep Q-learning for function approximation."""

from collections import defaultdict, deque
from typing import Dict, List, Optional, Set, Tuple
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


class ReplayBuffer:
    """Experience replay buffer for DQN training."""

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
        done: bool
    ):
        """Add a transition to the buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple:
        """Sample a batch of transitions."""
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32)
        )

    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    """Neural network for Q-function approximation."""

    def __init__(
        self,
        input_dim: int,
        num_actions: int,
        hidden_dims: List[int] = None
    ):
        """
        Initialize Q-network.

        Args:
            input_dim: Dimension of input features
            num_actions: Number of possible actions
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

        layers.append(nn.Linear(prev_dim, num_actions))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x)


class DQNRobotAgent:
    """
    A robot agent that learns via Deep Q-Network (DQN).

    Uses function approximation to handle large state spaces that
    make tabular Q-learning infeasible.
    """

    def __init__(
        self,
        num_actions: int = 5,
        learning_rate: float = 1e-4,
        discount_factor: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        buffer_size: int = 100000,
        batch_size: int = 64,
        target_update_freq: int = 100,
        learning_starts: int = 1000,
        hidden_dims: List[int] = None,
        grid_size: int = 10,
        active_categories: List[str] = None,
        seed: Optional[int] = None,
        device: str = None
    ):
        """
        Initialize the DQN robot agent.

        Args:
            num_actions: Number of possible actions (5: up, down, left, right, stay)
            learning_rate: Learning rate for the optimizer
            discount_factor: Discount factor (gamma)
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Decay rate for epsilon per episode
            buffer_size: Size of the replay buffer
            batch_size: Batch size for training
            target_update_freq: How often to update target network (in steps)
            learning_starts: Number of steps before training starts
            hidden_dims: Hidden layer dimensions for Q-network
            grid_size: Size of the grid (for normalization)
            active_categories: List of active property categories
            seed: Random seed
            device: Device to use ('cpu' or 'cuda')
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for DQN. Install with: pip install torch")

        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.learning_starts = learning_starts
        self.grid_size = grid_size
        self.hidden_dims = hidden_dims or [128, 128]

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

        # Calculate input dimension based on feature encoding
        self.input_dim = self._calculate_input_dim()

        # Initialize networks
        self.q_network = QNetwork(
            self.input_dim, num_actions, self.hidden_dims
        ).to(self.device)

        self.target_network = QNetwork(
            self.input_dim, num_actions, self.hidden_dims
        ).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)

        # Track inferred reward properties (for heuristic fallback)
        self.inferred_properties: Dict[str, float] = defaultdict(float)
        self.property_counts: Dict[str, int] = defaultdict(int)

        # Training statistics
        self.total_reward = 0.0
        self.steps = 0
        self.total_steps = 0  # Across all episodes
        self.train_losses = []

    def _build_property_indices(self):
        """Build indices for one-hot encoding property values."""
        self.property_to_idx = {}
        idx = 0
        for category in PROPERTY_CATEGORIES:
            for value in PROPERTY_VALUES[category]:
                self.property_to_idx[value] = idx
                idx += 1
        self.total_property_values = idx

    def _calculate_input_dim(self) -> int:
        """
        Calculate the dimension of the input feature vector.

        Features:
        - Robot position (2 normalized values)
        - Human position (2 normalized values)
        - Relative position of human (2 normalized values)
        - Human collected properties (one-hot, size = total property values)
        - For each of 3 nearest objects:
            - Relative position (2 normalized values)
            - Distance (1 normalized value)
            - Property one-hot encoding (size = values in active categories)
        - Inferred property scores (size = values in active categories)
        """
        # Position features
        pos_features = 6  # robot (2) + human (2) + relative (2)

        # Human collected properties (all possible property values)
        collected_features = self.total_property_values

        # Calculate features per object based on active categories
        props_per_object = sum(
            len(PROPERTY_VALUES[cat]) for cat in self.active_categories
        )

        # Per-object features: relative pos (2) + distance (1) + properties
        per_object_features = 3 + props_per_object

        # 3 nearest objects
        object_features = 3 * per_object_features

        # Inferred property scores for active categories
        inferred_features = props_per_object

        return pos_features + collected_features + object_features + inferred_features

    def _encode_observation(self, observation: dict) -> np.ndarray:
        """
        Encode observation dictionary into a fixed-size feature vector.

        Args:
            observation: Observation dictionary from environment

        Returns:
            numpy array of features
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

        # Human collected properties (one-hot for all possible values)
        collected_props = np.zeros(self.total_property_values)
        for collected in observation.get('human_collected', []):
            for prop_value in collected['properties'].values():
                if prop_value in self.property_to_idx:
                    collected_props[self.property_to_idx[prop_value]] = 1.0
        features.extend(collected_props.tolist())

        # Find 3 nearest objects
        objects = observation.get('objects', {})
        object_list = []
        for obj_id, obj_data in objects.items():
            pos = obj_data['position']
            dist = np.sqrt(
                (pos[0] - robot_pos[0])**2 + (pos[1] - robot_pos[1])**2
            )
            object_list.append((dist, obj_id, obj_data))

        object_list.sort(key=lambda x: x[0])
        nearest = object_list[:3]

        # Calculate features per object
        props_per_object = sum(
            len(PROPERTY_VALUES[cat]) for cat in self.active_categories
        )
        per_object_features = 3 + props_per_object  # rel_pos (2) + dist (1) + props

        # Encode each nearest object
        for i in range(3):
            if i < len(nearest):
                dist, obj_id, obj_data = nearest[i]
                pos = obj_data['position']

                # Relative position (normalized)
                rel_x = (pos[0] - robot_pos[0]) / self.grid_size
                rel_y = (pos[1] - robot_pos[1]) / self.grid_size

                # Distance (normalized by max possible distance)
                max_dist = np.sqrt(2) * self.grid_size
                norm_dist = dist / max_dist

                features.append(rel_x)
                features.append(rel_y)
                features.append(norm_dist)

                # Object properties (one-hot for active categories only)
                props = obj_data['properties']
                for cat in self.active_categories:
                    for value in PROPERTY_VALUES[cat]:
                        if props.get(cat) == value:
                            features.append(1.0)
                        else:
                            features.append(0.0)
            else:
                # Pad with zeros if fewer than 3 objects
                features.extend([0.0] * per_object_features)

        # Inferred property scores (normalized)
        max_score = max(self.inferred_properties.values()) if self.inferred_properties else 1.0
        max_score = max(max_score, 1.0)  # Avoid division by zero

        for cat in self.active_categories:
            for value in PROPERTY_VALUES[cat]:
                score = self.inferred_properties.get(value, 0.0)
                features.append(score / max_score)

        return np.array(features, dtype=np.float32)

    def reset(self):
        """Reset episode-specific state (not learned parameters)."""
        self.inferred_properties = defaultdict(float)
        self.property_counts = defaultdict(int)
        self.total_reward = 0.0
        self.steps = 0

    def reset_learning(self):
        """Fully reset the agent including learned Q-values."""
        # Reinitialize networks
        self.q_network = QNetwork(
            self.input_dim, self.num_actions, self.hidden_dims
        ).to(self.device)
        self.target_network = QNetwork(
            self.input_dim, self.num_actions, self.hidden_dims
        ).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.replay_buffer = ReplayBuffer(self.replay_buffer.buffer.maxlen)
        self.epsilon = self.epsilon_start
        self.total_steps = 0
        self.train_losses = []
        self.reset()

    def get_action(
        self,
        observation: dict,
        state: Tuple = None,  # Unused, kept for compatibility
        training: bool = True
    ) -> int:
        """
        Get the action for the robot agent.

        Uses epsilon-greedy policy during training.

        Args:
            observation: Current observation
            state: Unused (kept for API compatibility with tabular agent)
            training: Whether we're in training mode

        Returns:
            Action (0=up, 1=down, 2=left, 3=right, 4=stay)
        """
        # Update property inference based on human's collected objects
        self._update_inference(observation)

        # Encode observation
        features = self._encode_observation(observation)

        # Epsilon-greedy action selection
        if training and self.rng.random() < self.epsilon:
            # Explore: random action
            return self.rng.randint(0, self.num_actions - 1)
        else:
            # Exploit: best action from Q-network
            with torch.no_grad():
                state_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                return q_values.argmax(dim=1).item()

    def _update_inference(self, observation: dict):
        """
        Update inferred reward properties based on human's collections.

        Objects collected by the human are likely rewarding.
        """
        human_collected = observation.get('human_collected', [])

        for collected in human_collected:
            props = collected['properties']
            obj_id = collected['id']

            # Only count each object once
            if obj_id not in self.property_counts:
                for prop_value in props.values():
                    self.inferred_properties[prop_value] += 1.0
                self.property_counts[obj_id] = 1

    def update(
        self,
        state: Tuple,
        action: int,
        reward: float,
        next_state: Tuple,
        done: bool,
        observation: dict = None,
        next_observation: dict = None
    ):
        """
        Update Q-network using experience replay.

        Args:
            state: Current state (unused, kept for compatibility)
            action: Action taken
            reward: Reward received
            next_state: Next state (unused, kept for compatibility)
            done: Whether episode is done
            observation: Current observation (for feature encoding)
            next_observation: Next observation (for feature encoding)
        """
        self.total_reward += reward
        self.steps += 1
        self.total_steps += 1

        # If observations not provided, skip (can't train without features)
        if observation is None or next_observation is None:
            return

        # Encode observations
        state_features = self._encode_observation(observation)
        next_state_features = self._encode_observation(next_observation)

        # Store transition in replay buffer
        self.replay_buffer.push(
            state_features, action, reward, next_state_features, done
        )

        # Don't train until we have enough samples
        if len(self.replay_buffer) < self.learning_starts:
            return

        # Sample batch and train
        self._train_step()

        # Update target network
        if self.total_steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def _train_step(self):
        """Perform one training step on a batch from replay buffer."""
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Compute current Q-values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute target Q-values (using target network)
        with torch.no_grad():
            next_q = self.target_network(next_states).max(dim=1)[0]
            target_q = rewards + (1 - dones) * self.discount_factor * next_q

        # Compute loss (Huber loss for stability)
        loss = F.smooth_l1_loss(current_q, target_q)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()

        self.train_losses.append(loss.item())

    def decay_epsilon(self):
        """Decay epsilon after each episode."""
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon * self.epsilon_decay
        )

    def get_inferred_properties(self) -> List[Tuple[str, float]]:
        """Get the inferred reward properties sorted by confidence."""
        return sorted(
            self.inferred_properties.items(),
            key=lambda x: x[1],
            reverse=True
        )

    def get_average_loss(self, last_n: int = 100) -> float:
        """Get average training loss over last n steps."""
        if not self.train_losses:
            return 0.0
        recent = self.train_losses[-last_n:]
        return sum(recent) / len(recent)

    def save(self, path: str):
        """Save the model to a file."""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'total_steps': self.total_steps,
            'config': {
                'num_actions': self.num_actions,
                'input_dim': self.input_dim,
                'hidden_dims': self.hidden_dims,
                'active_categories': self.active_categories,
                'grid_size': self.grid_size
            }
        }, path)

    def load(self, path: str):
        """Load the model from a file."""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.total_steps = checkpoint['total_steps']
