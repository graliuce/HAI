"""Robot agent with Q-learning."""

from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple
import random
import numpy as np


class RobotAgent:
    """
    A robot agent that learns via Q-learning.

    The robot must infer reward-giving properties by observing
    which objects the human collects.
    """

    def __init__(
        self,
        num_actions: int = 5,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        seed: Optional[int] = None
    ):
        """
        Initialize the robot agent.

        Args:
            num_actions: Number of possible actions
            learning_rate: Learning rate (alpha) for Q-learning
            discount_factor: Discount factor (gamma)
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Decay rate for epsilon per episode
            seed: Random seed
        """
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)

        # Q-table: maps state -> action values
        self.q_table: Dict[Tuple, np.ndarray] = defaultdict(
            lambda: np.zeros(self.num_actions)
        )

        # Track inferred reward properties
        self.inferred_properties: Dict[str, float] = defaultdict(float)
        self.property_counts: Dict[str, int] = defaultdict(int)

        # Episode statistics
        self.total_reward = 0.0
        self.steps = 0

    def reset(self):
        """Reset episode-specific state (not learned parameters)."""
        self.inferred_properties = defaultdict(float)
        self.property_counts = defaultdict(int)
        self.total_reward = 0.0
        self.steps = 0

    def reset_learning(self):
        """Fully reset the agent including learned Q-values."""
        self.q_table = defaultdict(lambda: np.zeros(self.num_actions))
        self.epsilon = self.epsilon_start
        self.reset()

    def get_action(
        self,
        observation: dict,
        state: Tuple,
        training: bool = True
    ) -> int:
        """
        Get the action for the robot agent.

        Uses epsilon-greedy policy during training.

        Args:
            observation: Current observation
            state: Hashable state representation
            training: Whether we're in training mode

        Returns:
            Action (0=up, 1=down, 2=left, 3=right, 4=stay)
        """
        # Update property inference based on human's collected objects
        self._update_inference(observation)

        # Epsilon-greedy action selection
        if training and self.rng.random() < self.epsilon:
            # Explore: random action
            return self.rng.randint(0, self.num_actions - 1)
        else:
            # Exploit: best action from Q-table
            # If state unseen, use heuristic
            if state not in self.q_table:
                return self._heuristic_action(observation)

            q_values = self.q_table[state]

            # Break ties randomly
            max_q = np.max(q_values)
            best_actions = np.where(q_values == max_q)[0]
            return self.rng.choice(best_actions)

    def _heuristic_action(self, observation: dict) -> int:
        """
        Heuristic action when state is unseen.

        Move toward objects with inferred high-reward properties,
        or toward objects the human has been collecting.
        """
        robot_pos = observation['robot_position']
        objects = observation['objects']
        human_collected = observation['human_collected']

        if not objects:
            return 4  # stay

        # Score each object based on inferred properties
        scored_objects = []

        for obj_id, obj_data in objects.items():
            props = obj_data['properties']
            score = 0.0

            # Add score based on inferred properties
            for prop_value in props.values():
                score += self.inferred_properties.get(prop_value, 0.0)

            # Bonus if object has same properties as human-collected items
            if human_collected:
                for collected in human_collected:
                    collected_props = set(collected['properties'].values())
                    obj_props = set(props.values())
                    overlap = len(collected_props & obj_props)
                    score += overlap * 0.5

            pos = obj_data['position']
            dist = self._l2_distance(robot_pos, pos)

            # Combine score and distance (closer is better)
            if dist > 0:
                scored_objects.append((score / dist, pos))
            else:
                scored_objects.append((float('inf'), pos))

        if not scored_objects:
            return 4

        # Move toward best scoring object
        best_pos = max(scored_objects, key=lambda x: x[0])[1]
        return self._get_action_toward(robot_pos, best_pos)

    def _update_inference(self, observation: dict):
        """
        Update inferred reward properties based on human's collections.

        Objects collected by the human are likely rewarding.
        """
        human_collected = observation['human_collected']

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
        done: bool
    ):
        """
        Update Q-values using Q-learning.

        Q(s, a) = Q(s, a) + alpha * (r + gamma * max Q(s', a') - Q(s, a))

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        current_q = self.q_table[state][action]

        if done:
            target = reward
        else:
            next_max_q = np.max(self.q_table[next_state])
            target = reward + self.discount_factor * next_max_q

        # Q-learning update
        self.q_table[state][action] = current_q + self.learning_rate * (
            target - current_q
        )

        self.total_reward += reward
        self.steps += 1

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

    def _get_action_toward(
        self,
        current: Tuple[int, int],
        target: Tuple[int, int]
    ) -> int:
        """Get action to move toward target."""
        dx = target[0] - current[0]
        dy = target[1] - current[1]

        if dx == 0 and dy == 0:
            return 4  # stay

        if abs(dx) >= abs(dy):
            return 3 if dx > 0 else 2  # right or left
        else:
            return 1 if dy > 0 else 0  # down or up

    @staticmethod
    def _l2_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate L2 distance."""
        return ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5


class RobotAgentWithPropertyTracking(RobotAgent):
    """
    Enhanced robot agent that explicitly tracks property probabilities.

    Uses Bayesian-like updates to maintain beliefs about which properties
    are rewarding based on human behavior.
    """

    def __init__(
        self,
        active_categories: List[str],
        property_values: Dict[str, List[str]],
        **kwargs
    ):
        """
        Initialize with property tracking.

        Args:
            active_categories: List of active property categories
            property_values: Dict mapping categories to possible values
            **kwargs: Arguments for parent class
        """
        super().__init__(**kwargs)

        self.active_categories = active_categories
        self.property_values = property_values

        # Initialize uniform prior over all property values
        self.property_beliefs: Dict[str, float] = {}
        for category in active_categories:
            for value in property_values[category]:
                # Prior probability of each property being rewarding
                self.property_beliefs[value] = 0.5

        # Count of objects with each property that human collected
        self.collected_with_property: Dict[str, int] = defaultdict(int)
        # Count of objects with each property that human passed by (didn't collect)
        self.passed_with_property: Dict[str, int] = defaultdict(int)

        self.seen_object_ids: Set[int] = set()

    def reset(self):
        """Reset episode-specific state."""
        super().reset()
        # Reset beliefs to prior
        for category in self.active_categories:
            for value in self.property_values[category]:
                self.property_beliefs[value] = 0.5
        self.collected_with_property = defaultdict(int)
        self.passed_with_property = defaultdict(int)
        self.seen_object_ids = set()

    def _update_inference(self, observation: dict):
        """Update beliefs based on human's collections."""
        human_collected = observation['human_collected']

        for collected in human_collected:
            obj_id = collected['id']
            if obj_id in self.seen_object_ids:
                continue

            self.seen_object_ids.add(obj_id)
            props = collected['properties']

            for prop_value in props.values():
                self.collected_with_property[prop_value] += 1
                # Update belief using simple count-based approach
                total = (
                    self.collected_with_property[prop_value] +
                    self.passed_with_property[prop_value]
                )
                if total > 0:
                    self.property_beliefs[prop_value] = (
                        self.collected_with_property[prop_value] / total
                    )

        # Also update inferred_properties for heuristic
        self.inferred_properties = dict(self.property_beliefs)

    def get_top_properties(self, k: int = 2) -> List[str]:
        """Get the top k most likely rewarding properties."""
        sorted_props = sorted(
            self.property_beliefs.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [prop for prop, _ in sorted_props[:k]]
