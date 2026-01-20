"""Hierarchical robot agent with property-based goal-selection and A* navigation."""

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


if not TORCH_AVAILABLE:
    raise ImportError(
        "PyTorch is required for HierarchicalDQNRobotAgent. "
        "Install it with: pip install torch"
    )


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
    """Neural network for high-level property-selection Q-function."""

    def __init__(
        self,
        input_dim: int,
        num_property_actions: int,
        hidden_dims: List[int] = None
    ):
        """
        Args:
            input_dim: Dimension of state features
            num_property_actions: Number of property values to choose from
            hidden_dims: List of hidden layer dimensions
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [64, 64]

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        # Output: Q-value for each property value action
        layers.append(nn.Linear(prev_dim, num_property_actions))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class HierarchicalDQNRobotAgent:
    """
    Hierarchical robot agent with property-based goal selection.

    Architecture:
    - High-level policy (learned DQN): Selects which property value to target
    - Low-level policy (A* pathfinding): Navigates to nearest object with that property

    Learning:
    - The agent infers which properties are rewarding by observing what the human collects
    - Robot waits until human collects first object before starting to collect

    State space (compact, ~38 dims with 10 properties):
    - Robot position (2), Human position (2), Relative human position (2)
    - Can collect flag (1), Number of human collected (1)
    - Per-property features (3 each): inferred score, object count, min distance
    """

    def __init__(
        self,
        num_actions: int = 5,
        learning_rate: float = 1e-3,
        discount_factor: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        exploration_fraction: float = 0.3,
        total_timesteps: int = 100000,
        buffer_size: int = 50000,
        batch_size: int = 32,
        target_update_freq: int = 500,
        train_freq: int = 4,
        gradient_steps: int = 1,
        learning_starts: int = 500,
        hidden_dims: List[int] = None,
        grid_size: int = 10,
        num_objects: int = 20,
        active_categories: List[str] = None,
        seed: Optional[int] = None,
        device: str = None,
        verbose: bool = False
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for DQN. Install with: pip install torch")

        self.num_actions = num_actions
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
        self.num_objects = num_objects
        self.hidden_dims = hidden_dims or [64, 64]
        self.verbose = verbose

        # Set active categories
        self.active_categories = active_categories or PROPERTY_CATEGORIES[:2]

        # Build property value to action mapping
        self._build_property_action_mapping()

        # Number of property-based actions
        self.num_property_actions = len(self.property_values)

        # Current exploration rate
        self.epsilon = epsilon_start

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

        # Calculate input dimension for high-level policy
        self.high_level_input_dim = self._calculate_high_level_input_dim()

        # Initialize networks
        self.high_level_q_network = HighLevelQNetwork(
            self.high_level_input_dim, self.num_property_actions, self.hidden_dims
        ).to(self.device)

        self.high_level_target_network = HighLevelQNetwork(
            self.high_level_input_dim, self.num_property_actions, self.hidden_dims
        ).to(self.device)
        self.high_level_target_network.load_state_dict(self.high_level_q_network.state_dict())
        self.high_level_target_network.eval()

        # Optimizer
        self.optimizer = optim.Adam(
            self.high_level_q_network.parameters(), lr=learning_rate
        )

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)

        # Track inferred reward properties
        self.inferred_properties: Dict[str, float] = defaultdict(float)
        self.property_counts: Dict[str, int] = defaultdict(int)

        # Hierarchical state tracking
        self.current_target_property: Optional[str] = None
        self.current_target_action: Optional[int] = None
        self.current_goal_object_id: Optional[int] = None
        self.current_goal_position: Optional[Tuple[int, int]] = None
        self.high_level_state: Optional[np.ndarray] = None
        self.cumulative_reward: float = 0.0
        self.has_started: bool = False

        # A* pathfinding state
        self.current_path: List[Tuple[int, int]] = []

        # Training statistics
        self.total_reward = 0.0
        self.steps = 0
        self.total_steps = 0
        self.train_losses = []
        self._n_updates = 0
        
        # For verbose output during evaluation
        self.true_reward_properties: Optional[Set[str]] = None

    def _build_property_action_mapping(self):
        """Build mapping from action index to property value."""
        self.property_values = []  # List of property values (action index -> value)
        self.property_to_action = {}  # Map property value -> action index

        for category in self.active_categories:
            for value in PROPERTY_VALUES[category]:
                self.property_to_action[value] = len(self.property_values)
                self.property_values.append(value)

    def _calculate_high_level_input_dim(self) -> int:
        """
        Calculate dimension of high-level policy input.

        Compact state representation:
        - Robot position (2)
        - Human position (2)
        - Relative position of human (2)
        - Robot can collect flag (1)
        - Number of human collected (1, normalized)
        - For each property value:
            - Inferred score (1, normalized)
            - Number of available objects with this property (1, normalized)
            - Min distance to object with this property (1, normalized)
        """
        base_features = 8  # robot pos (2) + human pos (2) + relative pos (2) + can_collect (1) + num_collected (1)
        per_property_features = 3  # inferred_score + count + min_distance
        property_features = len(self.property_values) * per_property_features

        return base_features + property_features

    def _encode_high_level_input(self, observation: dict) -> np.ndarray:
        """Encode observation for high-level policy with compact representation."""
        features = []
        robot_pos = observation['robot_position']
        human_pos = observation['human_position']
        objects = observation.get('objects', {})

        # Robot position (normalized)
        features.append(robot_pos[0] / self.grid_size)
        features.append(robot_pos[1] / self.grid_size)

        # Human position (normalized)
        features.append(human_pos[0] / self.grid_size)
        features.append(human_pos[1] / self.grid_size)

        # Relative position of human to robot
        features.append((human_pos[0] - robot_pos[0]) / self.grid_size)
        features.append((human_pos[1] - robot_pos[1]) / self.grid_size)

        # Robot can collect flag
        robot_can_collect = observation.get('robot_can_collect', False)
        features.append(1.0 if robot_can_collect else 0.0)

        # Number of human collected objects (normalized)
        human_collected = observation.get('human_collected', [])
        features.append(len(human_collected) / max(1, self.num_objects))

        # Compute per-property features
        max_dist = np.sqrt(2) * self.grid_size
        max_inferred = max(self.inferred_properties.values()) if self.inferred_properties else 1.0
        max_inferred = max(max_inferred, 1.0)

        for prop_value in self.property_values:
            # Inferred score (normalized)
            inferred_score = self.inferred_properties.get(prop_value, 0.0)
            features.append(inferred_score / max_inferred)

            # Count objects with this property and find minimum distance
            count = 0
            min_dist = max_dist

            for obj_data in objects.values():
                props = obj_data['properties']
                if prop_value in props.values():
                    count += 1
                    obj_pos = obj_data['position']
                    dist = np.sqrt(
                        (obj_pos[0] - robot_pos[0])**2 +
                        (obj_pos[1] - robot_pos[1])**2
                    )
                    min_dist = min(min_dist, dist)

            # Object count (normalized)
            features.append(count / max(1, self.num_objects))

            # Min distance (normalized, inverted so closer = higher)
            if count > 0:
                features.append(1.0 - min_dist / max_dist)
            else:
                features.append(0.0)  # No objects with this property

        return np.array(features, dtype=np.float32)

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
        self.current_target_property = None
        self.current_target_action = None
        self.current_goal_object_id = None
        self.current_goal_position = None
        self.high_level_state = None
        self.cumulative_reward = 0.0
        self.has_started = False
        self.current_path = []

    def reset_learning(self):
        """Fully reset the agent including learned parameters."""
        self.high_level_q_network = HighLevelQNetwork(
            self.high_level_input_dim, self.num_property_actions, self.hidden_dims
        ).to(self.device)
        self.high_level_target_network = HighLevelQNetwork(
            self.high_level_input_dim, self.num_property_actions, self.hidden_dims
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

    def _get_valid_property_actions(self, observation: dict) -> List[int]:
        """Get list of valid property actions (properties that have available objects)."""
        objects = observation.get('objects', {})
        valid_actions = []

        for action_idx, prop_value in enumerate(self.property_values):
            # Check if any object has this property
            for obj_data in objects.values():
                if prop_value in obj_data['properties'].values():
                    valid_actions.append(action_idx)
                    break

        return valid_actions if valid_actions else [0]  # Fallback to first action

    def _should_select_new_goal(self, observation: dict) -> bool:
        """Determine if high-level policy should select a new goal."""
        if self.current_target_property is None:
            return True

        # Re-trigger if target object no longer exists
        if self.current_goal_object_id is not None:
            objects = observation.get('objects', {})
            if self.current_goal_object_id not in objects:
                return True

        return False

    def set_reward_properties_for_verbose(self, reward_properties: Set[str]):
        """Set the true reward properties for verbose output during evaluation."""
        self.true_reward_properties = reward_properties

    def _print_decision_summary(self, observation: dict, q_values: np.ndarray, 
                                 valid_actions: List[int], selected_action: int):
        """Print a summary of the high-level decision."""
        print("\n" + "="*80)
        print("HIGH-LEVEL DECISION")
        print("="*80)
        
        # Print rewarding properties if available
        if self.true_reward_properties is not None:
            rewarding_props = sorted(list(self.true_reward_properties))
            print(f"True Rewarding Properties: {rewarding_props}")
        
        # Count human collections
        human_collected = observation.get('human_collected', [])
        print(f"Human objects collected: {len(human_collected)}")
        
        # Get valid Q-values for table
        valid_q = q_values[valid_actions]
        
        # Calculate softmax probabilities
        q_max = valid_q.max()
        probs = np.exp(valid_q - q_max)
        probs = probs / (probs.sum() + 1e-8)
        
        # Calculate entropy
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        max_entropy = np.log(len(valid_actions))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        print(f"Entropy: {normalized_entropy:.4f}")
        
        # Print Q-values table
        rewarding_set = self.true_reward_properties if self.true_reward_properties is not None else set()
        print(f"\n{'Action':<15} {'Property':<15} {'Q-value':<12} {'Observed':<12} {'R?'}")
        print("-" * 65)
        for i, action_idx in enumerate(valid_actions):
            prop = self.property_values[action_idx]
            q_val = valid_q[i]
            obs_count = self.inferred_properties.get(prop, 0.0)
            is_rewarding = "✓" if prop in rewarding_set else ""
            marker = " <--" if action_idx == selected_action else ""
            print(f"{action_idx:<15} {prop:<15} {q_val:.4f}       {obs_count:<12.0f} {is_rewarding}{marker}")
        
        selected_prop = self.property_values[selected_action]
        is_selected_rewarding = "✓ (CORRECT)" if selected_prop in rewarding_set else "✗ (WRONG)" if rewarding_set else ""
        print(f"\nSelected: Action {selected_action} (property: {selected_prop}) {is_selected_rewarding}")
        print("="*80 + "\n")

    def _select_goal(self, observation: dict, training: bool = True) -> int:
        """
        High-level policy: Select which property to target.

        Returns:
            Action index corresponding to a property value
        """
        high_level_input = self._encode_high_level_input(observation)
        valid_actions = self._get_valid_property_actions(observation)

        if not valid_actions:
            return 0

        # Get Q-values
        with torch.no_grad():
            state_tensor = torch.FloatTensor(high_level_input).unsqueeze(0).to(self.device)
            q_values = self.high_level_q_network(state_tensor)
            q_values_np = q_values.cpu().numpy()[0]

        # Epsilon-greedy
        if training and self.rng.random() < self.epsilon:
            selected_action = self.rng.choice(valid_actions)
        else:
            # Mask invalid actions
            masked_q = np.full_like(q_values_np, -float('inf'))
            for a in valid_actions:
                masked_q[a] = q_values_np[a]
            selected_action = int(np.argmax(masked_q))

        # Print decision if verbose and not training
        if self.verbose and not training:
            self._print_decision_summary(observation, q_values_np, valid_actions, selected_action)

        return selected_action

    def _find_nearest_object_with_property(
        self,
        observation: dict,
        target_property: str
    ) -> Optional[Tuple[int, Tuple[int, int]]]:
        """Find the nearest object that has the target property."""
        robot_pos = observation['robot_position']
        objects = observation.get('objects', {})

        best_obj_id = None
        best_pos = None
        best_dist = float('inf')

        for obj_id, obj_data in objects.items():
            props = obj_data['properties']
            if target_property in props.values():
                obj_pos = obj_data['position']
                dist = np.sqrt(
                    (obj_pos[0] - robot_pos[0])**2 +
                    (obj_pos[1] - robot_pos[1])**2
                )
                if dist < best_dist:
                    best_dist = dist
                    best_obj_id = obj_id
                    best_pos = obj_pos

        if best_obj_id is not None:
            return (best_obj_id, best_pos)
        return None

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

        return [start]  # No path found

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

        # Recalculate path if needed
        if not self.current_path or self.current_path[0] != robot_pos:
            self.current_path = self._astar(robot_pos, self.current_goal_position, obstacles)

        if len(self.current_path) <= 1:
            return 4  # Already at target or no path

        self.current_path.pop(0)
        next_pos = self.current_path[0]
        return self._get_action_to_adjacent(robot_pos, next_pos)

    # ==================== Main Action Selection ====================

    def get_action(self, observation: dict, training: bool = True) -> int:
        """
        Get action using hierarchical policy.

        High-level: Select target property (learned DQN)
        Low-level: Navigate to nearest object with that property (A* pathfinding)
        """
        # Robot must wait until human collects first object
        robot_can_collect = observation.get('robot_can_collect', False)
        if not robot_can_collect:
            return 4  # stay

        # First time human has collected - mark as started
        if not self.has_started:
            self.has_started = True
            self._update_inference(observation)
            self.current_target_property = None

        # Update property inference
        self._update_inference(observation)

        # Check if we need to select a new goal
        if self._should_select_new_goal(observation):
            # Store previous transition for learning
            if training and self.high_level_state is not None and self.current_target_action is not None:
                new_high_level_state = self._encode_high_level_input(observation)
                self.replay_buffer.push(
                    self.high_level_state,
                    self.current_target_action,
                    self.cumulative_reward,
                    new_high_level_state,
                    False
                )

            # Select new target property
            self.current_target_action = self._select_goal(observation, training)
            self.current_target_property = self.property_values[self.current_target_action]
            self.high_level_state = self._encode_high_level_input(observation)
            self.cumulative_reward = 0.0
            self.current_path = []

            # Find nearest object with target property
            result = self._find_nearest_object_with_property(observation, self.current_target_property)
            if result:
                self.current_goal_object_id, self.current_goal_position = result
            else:
                self.current_goal_object_id = None
                self.current_goal_position = None

        # Update goal position if object still exists
        if self.current_goal_object_id is not None:
            objects = observation.get('objects', {})
            if self.current_goal_object_id in objects:
                self.current_goal_position = objects[self.current_goal_object_id]['position']
            else:
                # Object collected, find next nearest with same property
                result = self._find_nearest_object_with_property(observation, self.current_target_property)
                if result:
                    self.current_goal_object_id, self.current_goal_position = result
                    self.current_path = []
                else:
                    self.current_goal_object_id = None
                    self.current_goal_position = None

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

        # Update exploration rate
        self.epsilon = self._get_exploration_rate()

        # Accumulate reward for high-level
        self.cumulative_reward += reward

        # Store transition if episode is done
        if done and self.high_level_state is not None and self.current_target_action is not None:
            new_high_level_state = self._encode_high_level_input(next_observation)
            self.replay_buffer.push(
                self.high_level_state,
                self.current_target_action,
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
        """Train high-level property-selection policy."""
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
                'num_property_actions': self.num_property_actions,
                'high_level_input_dim': self.high_level_input_dim,
                'hidden_dims': self.hidden_dims,
                'active_categories': self.active_categories,
                'grid_size': self.grid_size,
                'num_objects': self.num_objects,
                'property_values': self.property_values
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
