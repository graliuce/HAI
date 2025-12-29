"""Hierarchical robot agent with learned goal-selection and A* navigation."""

from collections import defaultdict, deque
from typing import Dict, List, Optional, Set, Tuple
import random
import heapq
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


class HighLevelQNetwork(nn.Module):
    """Neural network for high-level goal-selection Q-function."""

    def __init__(
        self,
        input_dim: int,
        num_goals: int,
        hidden_dims: List[int] = None
    ):
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
        return self.network(x)


class HierarchicalDQNRobotAgent:
    """
    A hierarchical robot agent with:
    - High-level policy (learned): Selects which object to collect (goal)
    - Low-level policy (A* pathfinding): Navigates to the selected goal object

    Only the high-level policy is trained. The low-level uses A* pathfinding
    similar to the human agent's navigation.
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
            learning_rate: Learning rate for optimizer
            discount_factor: Discount factor (gamma)
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            exploration_fraction: Fraction of training for epsilon decay
            total_timesteps: Total timesteps for exploration schedule
            buffer_size: Size of replay buffer
            batch_size: Batch size for training
            target_update_freq: How often to update target network
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

        # Current exploration rate for high-level policy
        self.epsilon = epsilon_start

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

        # Calculate input dimension for high-level policy
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

        # Optimizer
        self.optimizer = optim.Adam(
            self.high_level_q_network.parameters(), lr=learning_rate
        )

        # Replay buffer for high-level transitions
        self.replay_buffer = ReplayBuffer(buffer_size)

        # Track inferred reward properties
        self.inferred_properties: Dict[str, float] = defaultdict(float)
        self.property_counts: Dict[str, int] = defaultdict(int)

        # Hierarchical state tracking
        self.current_goal_idx: Optional[int] = None
        self.current_goal_object_id: Optional[int] = None
        self.current_goal_position: Optional[Tuple[int, int]] = None
        self.steps_since_goal_selection: int = 0
        self.high_level_state: Optional[np.ndarray] = None
        self.cumulative_reward: float = 0.0

        # A* pathfinding state
        self.current_path: List[Tuple[int, int]] = []

        # Training statistics
        self.total_reward = 0.0
        self.steps = 0
        self.total_steps = 0
        self.train_losses = []
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

    def _calculate_high_level_input_dim(self) -> int:
        """
        Calculate dimension of high-level policy input.

        Features:
        - Robot position (2)
        - Human position (2)
        - Relative position of human (2)
        - Human collected properties (one-hot)
        - Inferred property scores
        - For each of K candidate objects:
            - Relative position (2)
            - Distance (1)
            - Properties (one-hot)
        """
        pos_features = 6
        collected_features = self.total_property_values
        props_dim = sum(len(PROPERTY_VALUES[cat]) for cat in self.active_categories)
        inferred_features = props_dim
        per_object_features = 3 + props_dim
        candidate_features = self.num_goal_candidates * per_object_features

        return pos_features + collected_features + inferred_features + candidate_features

    def _encode_high_level_input(self, observation: dict) -> np.ndarray:
        """Encode observation for high-level policy."""
        features = []
        robot_pos = observation['robot_position']
        human_pos = observation['human_position']

        # Robot position (normalized)
        features.append(robot_pos[0] / self.grid_size)
        features.append(robot_pos[1] / self.grid_size)

        # Human position (normalized)
        features.append(human_pos[0] / self.grid_size)
        features.append(human_pos[1] / self.grid_size)

        # Relative position of human to robot
        features.append((human_pos[0] - robot_pos[0]) / self.grid_size)
        features.append((human_pos[1] - robot_pos[1]) / self.grid_size)

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

        # Get goal candidates and encode each
        candidates = self._get_goal_candidates(observation)
        props_dim = sum(len(PROPERTY_VALUES[cat]) for cat in self.active_categories)
        per_object_dim = 3 + props_dim

        for i in range(self.num_goal_candidates):
            if i < len(candidates):
                obj_id, obj_data = candidates[i]
                obj_pos = obj_data['position']

                # Relative position (normalized)
                features.append((obj_pos[0] - robot_pos[0]) / self.grid_size)
                features.append((obj_pos[1] - robot_pos[1]) / self.grid_size)

                # Distance (normalized)
                dist = np.sqrt((obj_pos[0] - robot_pos[0])**2 + (obj_pos[1] - robot_pos[1])**2)
                max_dist = np.sqrt(2) * self.grid_size
                features.append(dist / max_dist)

                # Object properties (one-hot for active categories)
                props = obj_data['properties']
                for cat in self.active_categories:
                    for value in PROPERTY_VALUES[cat]:
                        features.append(1.0 if props.get(cat) == value else 0.0)
            else:
                # Pad with zeros if fewer candidates
                features.extend([0.0] * per_object_dim)

        return np.array(features, dtype=np.float32)

    def _get_goal_candidates(self, observation: dict) -> List[Tuple[int, dict]]:
        """Get K nearest objects as goal candidates."""
        robot_pos = observation['robot_position']
        objects = observation.get('objects', {})

        if not objects:
            return []

        object_list = []
        for obj_id, obj_data in objects.items():
            pos = obj_data['position']
            dist = np.sqrt((pos[0] - robot_pos[0])**2 + (pos[1] - robot_pos[1])**2)
            object_list.append((dist, obj_id, obj_data))

        object_list.sort(key=lambda x: x[0])

        candidates = []
        for i in range(min(self.num_goal_candidates, len(object_list))):
            _, obj_id, obj_data = object_list[i]
            candidates.append((obj_id, obj_data))

        return candidates

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
        self.current_goal_object_id = None
        self.current_goal_position = None
        self.steps_since_goal_selection = 0
        self.high_level_state = None
        self.cumulative_reward = 0.0
        self.current_path = []

    def reset_learning(self):
        """Fully reset the agent including learned parameters."""
        self.high_level_q_network = HighLevelQNetwork(
            self.high_level_input_dim, self.num_goal_candidates, self.hidden_dims
        ).to(self.device)
        self.high_level_target_network = HighLevelQNetwork(
            self.high_level_input_dim, self.num_goal_candidates, self.hidden_dims
        ).to(self.device)
        self.high_level_target_network.load_state_dict(self.high_level_q_network.state_dict())
        self.high_level_target_network.eval()

        self.optimizer = optim.Adam(
            self.high_level_q_network.parameters(), lr=self.learning_rate
        )
        self.replay_buffer = ReplayBuffer(self.replay_buffer.buffer.maxlen)
        self.epsilon = self.epsilon_start
        self.total_steps = 0
        self._n_updates = 0
        self.train_losses = []
        self.reset()

    def _should_select_new_goal(self, observation: dict) -> bool:
        """Determine if high-level policy should select a new goal."""
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
        """High-level policy: Select which object to collect."""
        high_level_input = self._encode_high_level_input(observation)
        candidates = self._get_goal_candidates(observation)
        num_valid = len(candidates)

        if num_valid == 0:
            return 0

        # Epsilon-greedy for high-level
        if training and self.rng.random() < self.epsilon:
            goal_idx = self.rng.randint(0, num_valid - 1)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(high_level_input).unsqueeze(0).to(self.device)
                q_values = self.high_level_q_network(state_tensor)
                q_values_np = q_values.cpu().numpy()[0]
                # Mask invalid goals
                q_values_np[num_valid:] = -float('inf')
                goal_idx = np.argmax(q_values_np)

        return goal_idx

    # ==================== A* Navigation ====================

    def _astar(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        obstacles: Set[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """A* pathfinding algorithm that avoids obstacles."""
        if start == goal:
            return [start]

        counter = 0
        open_set = [(0, counter, start)]
        came_from = {}
        g_score = {start: 0}

        while open_set:
            _, _, current = heapq.heappop(open_set)

            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return list(reversed(path))

            for neighbor in self._get_neighbors(current, obstacles):
                tentative_g = g_score[current] + 1

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self._l2_distance(neighbor, goal)
                    counter += 1
                    heapq.heappush(open_set, (f_score, counter, neighbor))

        # No path found
        return [start]

    def _get_neighbors(self, pos: Tuple[int, int], obstacles: Set[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Get valid neighboring positions, excluding obstacles."""
        x, y = pos
        neighbors = []

        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                if (nx, ny) not in obstacles:
                    neighbors.append((nx, ny))

        return neighbors

    @staticmethod
    def _l2_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        return ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5

    def _get_action_to_adjacent(self, current: Tuple[int, int], adjacent: Tuple[int, int]) -> int:
        """Get action to move to an adjacent cell."""
        dx = adjacent[0] - current[0]
        dy = adjacent[1] - current[1]

        if dx == 1:
            return 3  # right
        elif dx == -1:
            return 2  # left
        elif dy == 1:
            return 1  # down
        elif dy == -1:
            return 0  # up
        else:
            return 4  # stay

    def _get_navigation_action(self, observation: dict) -> int:
        """Use A* to navigate toward the current goal."""
        if self.current_goal_position is None:
            return 4  # stay

        robot_pos = observation['robot_position']
        objects = observation.get('objects', {})

        # Build set of obstacle positions (all objects except goal)
        obstacles = set()
        for obj_id, obj_data in objects.items():
            if obj_id != self.current_goal_object_id:
                obstacles.add(obj_data['position'])

        # Check if we need to recalculate path
        if not self.current_path or self.current_path[0] != robot_pos:
            self.current_path = self._astar(robot_pos, self.current_goal_position, obstacles)

        if len(self.current_path) <= 1:
            return 4  # Already at target or no path

        # Pop current position and move to next
        self.current_path.pop(0)
        next_pos = self.current_path[0]
        return self._get_action_to_adjacent(robot_pos, next_pos)

    # ==================== Main Action Selection ====================

    def get_action(self, observation: dict, training: bool = True) -> int:
        """
        Get action using hierarchical policy.

        High-level: Select goal object (learned DQN)
        Low-level: Navigate to goal (A* pathfinding)
        """
        # Update property inference
        self._update_inference(observation)

        # Check if we need to select a new goal
        if self._should_select_new_goal(observation):
            # Store previous high-level state for learning
            if training and self.high_level_state is not None and self.current_goal_idx is not None:
                new_high_level_state = self._encode_high_level_input(observation)
                self.replay_buffer.push(
                    self.high_level_state,
                    self.current_goal_idx,
                    self.cumulative_reward,
                    new_high_level_state,
                    False
                )

            # Select new goal
            self.current_goal_idx = self._select_goal(observation, training)
            self.high_level_state = self._encode_high_level_input(observation)
            self.steps_since_goal_selection = 0
            self.cumulative_reward = 0.0
            self.current_path = []

            # Update goal object info
            candidates = self._get_goal_candidates(observation)
            if self.current_goal_idx < len(candidates):
                obj_id, obj_data = candidates[self.current_goal_idx]
                self.current_goal_object_id = obj_id
                self.current_goal_position = obj_data['position']
            else:
                self.current_goal_object_id = None
                self.current_goal_position = None

        # Update goal position (object may have been collected by human)
        if self.current_goal_object_id is not None:
            objects = observation.get('objects', {})
            if self.current_goal_object_id in objects:
                self.current_goal_position = objects[self.current_goal_object_id]['position']
            else:
                # Goal no longer exists
                self.current_goal_position = None

        # Use A* to navigate toward goal
        return self._get_navigation_action(observation)

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

    def update(
        self,
        action: int,
        reward: float,
        done: bool,
        observation: dict,
        next_observation: dict
    ):
        """Update high-level Q-network."""
        self.total_reward += reward
        self.steps += 1
        self.total_steps += 1
        self.steps_since_goal_selection += 1

        # Update exploration rate
        self.epsilon = self._get_exploration_rate()

        # Accumulate reward for high-level
        self.cumulative_reward += reward

        # Store high-level transition if episode is done
        if done and self.high_level_state is not None and self.current_goal_idx is not None:
            new_high_level_state = self._encode_high_level_input(next_observation)
            self.replay_buffer.push(
                self.high_level_state,
                self.current_goal_idx,
                self.cumulative_reward,
                new_high_level_state,
                True
            )

        # Training
        if len(self.replay_buffer) < self.learning_starts:
            return

        if self.total_steps % self.train_freq == 0:
            for _ in range(self.gradient_steps):
                self._train_step()

        # Update target network
        if self.total_steps % self.target_update_freq == 0:
            self.high_level_target_network.load_state_dict(self.high_level_q_network.state_dict())

    def _train_step(self):
        """Train high-level goal-selection policy."""
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

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
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.high_level_q_network.parameters(), 10.0)
        self.optimizer.step()

        self.train_losses.append(loss.item())
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
        if not self.train_losses:
            return 0.0
        recent = self.train_losses[-last_n:]
        return sum(recent) / len(recent)

    def save(self, path: str):
        """Save the model to a file."""
        torch.save({
            'high_level_q_network': self.high_level_q_network.state_dict(),
            'high_level_target_network': self.high_level_target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'total_steps': self.total_steps,
            'config': {
                'num_actions': self.num_actions,
                'num_goal_candidates': self.num_goal_candidates,
                'high_level_interval': self.high_level_interval,
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
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.total_steps = checkpoint['total_steps']
