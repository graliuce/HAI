"""Belief-based robot agent with explicit Bayesian preference modeling."""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import heapq
import numpy as np

from ..objects import PROPERTY_CATEGORIES, PROPERTY_VALUES
from ..llm_interface import LLMInterface, SimulatedHumanResponder


@dataclass
class GaussianBeliefState:
    """
    Maintains a multivariate Gaussian belief over feature preference weights.

    Features are instantiations of properties (e.g., red, blue, circle, square).
    The belief state is N(mean, covariance) where:
    - mean: Expected preference weight for each feature
    - covariance: Uncertainty over preference weights

    Prior: Standard multivariate Gaussian N(0, I)
    """

    mean: np.ndarray  # Shape: (num_features,)
    covariance: np.ndarray  # Shape: (num_features, num_features)
    feature_names: List[str] = field(default_factory=list)
    feature_to_idx: Dict[str, int] = field(default_factory=dict)

    @classmethod
    def create_prior(cls, feature_names: List[str]) -> 'GaussianBeliefState':
        """
        Create a standard multivariate Gaussian prior N(0, I).

        Args:
            feature_names: List of feature names (e.g., ['red', 'blue', 'circle', ...])

        Returns:
            GaussianBeliefState initialized with N(0, I)
        """
        num_features = len(feature_names)
        mean = np.zeros(num_features)
        covariance = np.eye(num_features)
        feature_to_idx = {name: i for i, name in enumerate(feature_names)}

        return cls(
            mean=mean,
            covariance=covariance,
            feature_names=list(feature_names),
            feature_to_idx=feature_to_idx
        )

    @property
    def num_features(self) -> int:
        return len(self.feature_names)

    def get_expected_weights(self) -> Dict[str, float]:
        """Get expected weight for each feature."""
        return {name: self.mean[idx] for name, idx in self.feature_to_idx.items()}

    def get_feature_variance(self, feature_name: str) -> float:
        """Get variance (uncertainty) for a specific feature."""
        idx = self.feature_to_idx[feature_name]
        return self.covariance[idx, idx]

    def compute_participation_ratio(self) -> float:
        """
        Compute the participation ratio of the covariance eigenvalues.

        PR = (sum of eigenvalues)^2 / (sum of squared eigenvalues)

        This measures the "effective dimensionality" of the uncertainty:
        - PR = 1: Uncertainty concentrated in one direction
        - PR = d: Uncertainty spread uniformly across all d dimensions

        High PR indicates high overall uncertainty, suggesting a query may be valuable.

        Returns:
            Participation ratio in range [1, num_features]
        """
        eigenvalues = np.linalg.eigvalsh(self.covariance)
        eigenvalues = np.maximum(eigenvalues, 1e-10)  # Numerical stability

        sum_eigenvalues = np.sum(eigenvalues)
        sum_squared_eigenvalues = np.sum(eigenvalues ** 2)

        if sum_squared_eigenvalues < 1e-10:
            return 1.0

        return (sum_eigenvalues ** 2) / sum_squared_eigenvalues

    def get_object_feature_vector(self, properties: Dict[str, str]) -> np.ndarray:
        """
        Convert object properties to a binary feature vector.

        Args:
            properties: Dict mapping category -> value (e.g., {'color': 'red', 'shape': 'circle'})

        Returns:
            Binary feature vector of shape (num_features,)
        """
        feature_vector = np.zeros(self.num_features)
        for prop_value in properties.values():
            if prop_value in self.feature_to_idx:
                feature_vector[self.feature_to_idx[prop_value]] = 1.0
        return feature_vector

    def get_expected_utility(self, properties: Dict[str, str]) -> float:
        """
        Compute expected utility of an object given current beliefs.

        Utility = sum of expected weights for each feature the object has.

        Args:
            properties: Object properties

        Returns:
            Expected utility (dot product of feature vector with mean weights)
        """
        feature_vector = self.get_object_feature_vector(properties)
        return np.dot(self.mean, feature_vector)

    def update_from_choice_plackett_luce(
        self,
        chosen_object: Dict[str, str],
        alternative_objects: List[Dict[str, str]],
        chosen_distance: float,
        alternative_distances: List[float],
        learning_rate: float = 0.1,
        num_gradient_steps: int = 5
    ):
        """
        Update beliefs using Plackett-Luce model for the human's choice.

        The human chooses the object with highest utility/distance ratio.
        We model this as a softmax choice model:

        P(choose i) = exp(utility_i / distance_i) / sum_j exp(utility_j / distance_j)

        where utility = w^T * features

        We use gradient ascent on the log-likelihood to update the mean,
        and a simple approximation for the covariance update.

        Args:
            chosen_object: Properties of the object the human chose
            alternative_objects: Properties of all available objects (including chosen)
            chosen_distance: Distance from human to chosen object
            alternative_distances: Distances from human to all objects
            learning_rate: Step size for gradient updates
            num_gradient_steps: Number of gradient ascent steps
        """
        # Build feature vectors for all objects
        chosen_features = self.get_object_feature_vector(chosen_object)
        all_features = [self.get_object_feature_vector(obj) for obj in alternative_objects]

        # Use max(dist, 1.0) to avoid division by zero
        chosen_dist = max(chosen_distance, 1.0)
        all_distances = [max(d, 1.0) for d in alternative_distances]

        # Gradient ascent on log-likelihood
        for _ in range(num_gradient_steps):
            # Compute utility/distance for all objects
            utilities = np.array([np.dot(self.mean, f) for f in all_features])
            weighted_utilities = utilities / np.array(all_distances)

            # Softmax probabilities
            max_wu = np.max(weighted_utilities)
            exp_wu = np.exp(weighted_utilities - max_wu)
            probs = exp_wu / (np.sum(exp_wu) + 1e-10)

            # Gradient of log P(chosen) with respect to weights
            # d log P / d w = (chosen_features / chosen_dist) - sum_j p_j * (features_j / dist_j)
            expected_features = sum(
                p * (f / d) for p, f, d in zip(probs, all_features, all_distances)
            )
            gradient = (chosen_features / chosen_dist) - expected_features

            # Update mean
            self.mean = self.mean + learning_rate * gradient

        # Update covariance using Fisher information approximation
        # The Fisher information for Plackett-Luce is related to the Hessian of log-likelihood
        # We approximate it with a rank-1 update based on the observed information

        # Compute approximate Fisher information from the choice
        # For each alternative, compute its contribution to the information matrix
        fisher_info = np.zeros((self.num_features, self.num_features))
        for p, f, d in zip(probs, all_features, all_distances):
            weighted_f = f / d
            # Contribution is p * (1-p) * outer(f/d, f/d) but simplified
            fisher_info += p * np.outer(weighted_f, weighted_f)

        # Subtract the expected outer product squared
        expected_f = sum(p * (f / d) for p, f, d in zip(probs, all_features, all_distances))
        fisher_info -= np.outer(expected_f, expected_f)

        # Scale the information gain (higher = faster uncertainty reduction)
        info_gain = 0.5  # Increased from 0.1

        # Update covariance: Sigma_new = Sigma - info_gain * Sigma @ Fisher @ Sigma
        # This is the standard Bayesian update approximation
        update = info_gain * self.covariance @ fisher_info @ self.covariance
        self.covariance = self.covariance - update

        # Ensure covariance stays positive definite with minimum eigenvalue
        eigenvalues, eigenvectors = np.linalg.eigh(self.covariance)
        eigenvalues = np.maximum(eigenvalues, 0.01)  # Minimum variance
        self.covariance = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    def update_from_linear_gaussian(
        self,
        observed_weights: Dict[str, float],
        observation_noise_variance: float = 1.0
    ):
        """
        Update beliefs using Linear-Gaussian observation model.

        The LLM provides a noisy observation of the true weights:
        y = w + epsilon, where epsilon ~ N(0, R)

        Standard Bayesian linear Gaussian update:
        mu_new = mu + K(y - mu)
        Sigma_new = (I - K) * Sigma

        where K = Sigma * (Sigma + R)^{-1}

        Args:
            observed_weights: Dict mapping feature names to observed weights
                             (from LLM, already L2-normalized)
            observation_noise_variance: Variance of observation noise
        """
        # Convert observed weights to vector (only for features we have)
        y = np.zeros(self.num_features)
        observed_mask = np.zeros(self.num_features, dtype=bool)

        for name, weight in observed_weights.items():
            if name in self.feature_to_idx:
                idx = self.feature_to_idx[name]
                y[idx] = weight
                observed_mask[idx] = True

        # For features not mentioned, we don't update (treat as no observation)
        # Build observation matrix H that selects only observed features
        num_observed = np.sum(observed_mask)
        if num_observed == 0:
            return  # No update if no features were observed

        # Full observation model for simplicity (observe all features)
        # Noise covariance (diagonal)
        R = np.eye(self.num_features) * observation_noise_variance

        # Kalman gain: K = Sigma * (Sigma + R)^{-1}
        S = self.covariance + R
        K = np.linalg.solve(S.T, self.covariance.T).T  # More stable than direct inverse

        # Update mean: mu_new = mu + K * (y - mu)
        innovation = y - self.mean
        self.mean = self.mean + K @ innovation

        # Update covariance: Sigma_new = (I - K) * Sigma
        I = np.eye(self.num_features)
        self.covariance = (I - K) @ self.covariance

        # Ensure symmetry and positive definiteness
        self.covariance = 0.5 * (self.covariance + self.covariance.T)
        eigenvalues, eigenvectors = np.linalg.eigh(self.covariance)
        eigenvalues = np.maximum(eigenvalues, 0.001)  # Minimum variance
        self.covariance = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T


class BeliefBasedRobotAgent:
    """
    Robot agent that uses explicit Bayesian belief modeling over human preferences.

    Architecture:
    - Maintains multivariate Gaussian belief over feature preference weights
    - High-level policy: Selects object with highest expected_utility / distance
    - Low-level policy: A* pathfinding to reach selected object

    Belief updates:
    - Observation: Plackett-Luce model when human collects an object
    - Query: Linear-Gaussian model from LLM-extracted weights

    Query triggering:
    - Uses participation ratio of covariance eigenvalues
    - Query when PR > threshold (high effective dimensionality = high uncertainty)
    """

    def __init__(
        self,
        grid_size: int = 10,
        num_objects: int = 20,
        active_categories: List[str] = None,
        num_property_values: int = 5,
        llm_interface: Optional[LLMInterface] = None,
        query_budget: int = 5,
        participation_ratio_threshold: float = 3.0,
        plackett_luce_learning_rate: float = 0.1,
        plackett_luce_gradient_steps: int = 5,
        linear_gaussian_noise_variance: float = 1.0,
        verbose: bool = False,
        seed: Optional[int] = None
    ):
        """
        Args:
            grid_size: Size of the grid
            num_objects: Number of objects in environment
            active_categories: Property categories that are active (vary across objects)
            num_property_values: Number of values per property category
            llm_interface: Interface to LLM for queries
            query_budget: Maximum queries per episode
            participation_ratio_threshold: Trigger query when PR > this value
            plackett_luce_learning_rate: Learning rate for Plackett-Luce updates
            plackett_luce_gradient_steps: Number of gradient steps per observation
            linear_gaussian_noise_variance: Observation noise for linear-Gaussian updates
            verbose: Print debug information
            seed: Random seed
        """
        self.grid_size = grid_size
        self.num_objects = num_objects
        self.active_categories = active_categories or PROPERTY_CATEGORIES[:2]
        self.num_property_values = num_property_values
        self.llm = llm_interface
        self.query_budget = query_budget
        self.pr_threshold = participation_ratio_threshold
        self.pl_learning_rate = plackett_luce_learning_rate
        self.pl_gradient_steps = plackett_luce_gradient_steps
        self.lg_noise_variance = linear_gaussian_noise_variance
        self.verbose = verbose

        # Random generator
        if seed is not None:
            np.random.seed(seed)
        self.rng = np.random.RandomState(seed)

        # Build feature list (all property values for active categories)
        self.feature_names = []
        for category in self.active_categories:
            for value in PROPERTY_VALUES[category][:num_property_values]:
                self.feature_names.append(value)

        # Also create property_values list for compatibility with existing code
        self.property_values = self.feature_names.copy()

        # Initialize belief state (will be reset per episode)
        self.belief: Optional[GaussianBeliefState] = None

        # Current target tracking
        self.current_target_object_id: Optional[int] = None
        self.current_target_position: Optional[Tuple[int, int]] = None
        self.current_path: List[Tuple[int, int]] = []

        # Query tracking
        self.queries_used: int = 0
        self.query_history: List[Dict] = []

        # Observation tracking
        self.observed_object_ids: Set[int] = set()
        self.has_started: bool = False

        # Human responder (set during episode for testing)
        self.human_responder: Optional[SimulatedHumanResponder] = None

        # For tracking decisions
        self.decision_history: List[Dict] = []
        self.entropy_history: List[Dict] = []

        # True reward properties (for verbose output)
        self.true_reward_properties: Optional[Set[str]] = None

    def reset(self, active_categories: List[str] = None):
        """
        Reset for a new episode.

        Args:
            active_categories: If provided, update the feature space to match
                              the environment's active categories for this episode.
        """
        # Update active categories if provided (for per-episode adaptation)
        if active_categories is not None:
            self.active_categories = active_categories
            # Rebuild feature list for the new active categories
            self.feature_names = []
            for category in self.active_categories:
                for value in PROPERTY_VALUES[category][:self.num_property_values]:
                    self.feature_names.append(value)
            self.property_values = self.feature_names.copy()

        # Initialize prior belief N(0, I) with correct dimensionality
        self.belief = GaussianBeliefState.create_prior(self.feature_names)

        if self.verbose:
            print(f"\n[Belief Reset]")
            print(f"  Active Categories: {self.active_categories}")
            print(f"  Number of Features: {len(self.feature_names)}")
            print(f"  Initial Participation Ratio: {self.belief.compute_participation_ratio():.3f}")

        # Reset target tracking
        self.current_target_object_id = None
        self.current_target_position = None
        self.current_path = []

        # Reset query tracking
        self.queries_used = 0
        self.query_history = []

        # Reset observation tracking
        self.observed_object_ids = set()
        self.has_started = False

        # Reset history
        self.decision_history = []
        self.entropy_history = []

        self.human_responder = None

    def set_human_responder(
        self,
        reward_properties: set,
        property_value_rewards: Optional[Dict[str, float]] = None
    ):
        """Set up simulated human responder with true preferences."""
        if self.llm is None:
            return
        self.human_responder = SimulatedHumanResponder(
            reward_properties,
            llm_interface=self.llm,
            verbose=self.verbose,
            property_value_rewards=property_value_rewards
        )

    def set_reward_properties_for_verbose(self, reward_properties: Set[str]):
        """Set the true reward properties for verbose output."""
        self.true_reward_properties = reward_properties

    def _compute_object_utilities(
        self,
        observation: dict
    ) -> Dict[int, Tuple[float, float, float]]:
        """
        Compute expected utility, distance, and utility/distance ratio for all objects.

        Returns:
            Dict mapping object_id -> (utility, distance, ratio)
        """
        objects = observation.get('objects', {})
        robot_pos = observation['robot_position']

        results = {}
        for obj_id, obj_data in objects.items():
            properties = obj_data['properties']
            position = obj_data['position']

            utility = self.belief.get_expected_utility(properties)
            distance = max(1.0, np.sqrt(
                (position[0] - robot_pos[0])**2 + (position[1] - robot_pos[1])**2
            ))
            ratio = utility / distance

            results[obj_id] = (utility, distance, ratio)

        return results

    def _select_target(self, observation: dict) -> Optional[int]:
        """
        Select the target object with highest utility/distance ratio.

        Returns:
            Object ID of selected target, or None if no objects
        """
        utilities = self._compute_object_utilities(observation)

        if not utilities:
            return None

        # Select object with highest ratio
        best_obj_id = max(utilities.keys(), key=lambda oid: utilities[oid][2])

        # Store decision info
        if self.verbose:
            objects = observation.get('objects', {})
            decision_info = {
                'step': observation.get('step', 0),
                'selected_object': best_obj_id,
                'utilities': {
                    oid: {
                        'properties': objects[oid]['properties'],
                        'utility': u,
                        'distance': d,
                        'ratio': r
                    }
                    for oid, (u, d, r) in utilities.items()
                }
            }
            self.decision_history.append(decision_info)

        return best_obj_id

    def _should_query(self, observation: dict) -> bool:
        """
        Decide whether to trigger a query based on participation ratio.

        Query if:
        1. We have query budget remaining
        2. Participation ratio exceeds threshold

        Returns:
            True if should query
        """
        if self.llm is None:
            return False

        if self.queries_used >= self.query_budget:
            return False

        pr = self.belief.compute_participation_ratio()
        should_query = pr > self.pr_threshold

        # Store for analysis
        self.entropy_history.append({
            'participation_ratio': pr,
            'threshold': self.pr_threshold,
            'should_query': should_query,
            'queries_used': self.queries_used
        })

        if self.verbose:
            print(f"[Query Decision] PR={pr:.3f}, threshold={self.pr_threshold}, query={should_query}")

        return should_query

    def _execute_query(self, observation: dict) -> Dict[str, float]:
        """
        Execute a query to the human and return extracted weights.

        Returns:
            Dict mapping feature names to weights
        """
        if self.llm is None or self.human_responder is None:
            return {}

        # Gather board properties
        objects = observation.get('objects', {})
        board_properties = []
        for obj_data in objects.values():
            board_properties.extend(obj_data['properties'].values())
        board_properties = list(dict.fromkeys(board_properties))

        # Gather collected properties
        human_collected = observation.get('human_collected', [])
        collected_properties = []
        for collected in human_collected:
            collected_properties.extend(collected['properties'].values())
        collected_properties = list(dict.fromkeys(collected_properties))

        # Generate query
        query = self.llm.generate_query(
            board_properties,
            collected_properties,
            self.active_categories
        )

        # Get human response
        response = self.human_responder.respond_to_query(
            query,
            board_properties,
            collected_properties
        )

        # Interpret response
        property_weights = self.llm.interpret_response(
            query,
            response,
            board_properties,
            collected_properties,
            self.feature_names,
            self.active_categories
        )

        # Normalize using L2 norm
        weights_array = np.array([property_weights.get(f, 0.0) for f in self.feature_names])
        l2_norm = np.linalg.norm(weights_array)
        if l2_norm > 1e-10:
            weights_array = weights_array / l2_norm
            property_weights = {
                f: weights_array[i] for i, f in enumerate(self.feature_names)
            }

        # Store in history
        self.query_history.append({
            'query': query,
            'response': response,
            'weights': property_weights,
            'board_properties': board_properties,
            'collected_properties': collected_properties
        })
        self.queries_used += 1

        if self.verbose:
            print(f"\n[Query {self.queries_used}]")
            print(f"  Query: {query}")
            print(f"  Response: {response}")
            print(f"  Extracted weights (L2 normalized):")
            sorted_weights = sorted(property_weights.items(), key=lambda x: abs(x[1]), reverse=True)
            for prop, weight in sorted_weights[:5]:
                print(f"    {prop}: {weight:+.4f}")

        return property_weights

    def _update_belief_from_observation(
        self,
        chosen_object: dict,
        observation: dict
    ):
        """
        Update belief using Plackett-Luce model when human collects an object.

        Args:
            chosen_object: The object the human collected (with 'properties' and 'position')
            observation: Full observation for distances to alternatives
        """
        objects = observation.get('objects', {})
        human_pos = observation['human_position']

        # Get all alternative objects (including the chosen one conceptually)
        # But since the chosen object may have been removed, we reconstruct
        all_objects_props = [chosen_object['properties']]
        all_distances = [max(1.0, np.sqrt(
            (chosen_object['position'][0] - human_pos[0])**2 +
            (chosen_object['position'][1] - human_pos[1])**2
        ))]

        for obj_id, obj_data in objects.items():
            all_objects_props.append(obj_data['properties'])
            dist = np.sqrt(
                (obj_data['position'][0] - human_pos[0])**2 +
                (obj_data['position'][1] - human_pos[1])**2
            )
            all_distances.append(max(1.0, dist))

        # Distance to chosen object
        chosen_distance = all_distances[0]

        # Update belief
        self.belief.update_from_choice_plackett_luce(
            chosen_object=chosen_object['properties'],
            alternative_objects=all_objects_props,
            chosen_distance=chosen_distance,
            alternative_distances=all_distances,
            learning_rate=self.pl_learning_rate,
            num_gradient_steps=self.pl_gradient_steps
        )

        if self.verbose:
            pr_after = self.belief.compute_participation_ratio()
            print(f"\n[Belief Update - Plackett-Luce]")
            print(f"  Human collected object with properties: {chosen_object['properties']}")
            print(f"  Distance to chosen: {chosen_distance:.2f}")
            print(f"  Number of alternatives: {len(all_objects_props) - 1}")
            print(f"  Participation Ratio after update: {pr_after:.3f}")
            print(f"  Updated mean weights (top 5):")
            weights = self.belief.get_expected_weights()
            sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
            for prop, weight in sorted_weights[:5]:
                print(f"    {prop}: {weight:+.4f}")

    def _process_observations(self, observation: dict):
        """Process new human collections and update beliefs."""
        human_collected = observation.get('human_collected', [])

        for collected in human_collected:
            obj_id = collected['id']
            if obj_id not in self.observed_object_ids:
                self.observed_object_ids.add(obj_id)

                # Reconstruct object info for belief update
                # Note: position might not be in collected, so we'll estimate
                # For now, use human position as proxy (they just collected it)
                chosen_object = {
                    'properties': collected['properties'],
                    'position': observation['human_position']
                }

                self._update_belief_from_observation(chosen_object, observation)

    # ==================== A* Navigation ====================

    def _astar(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        obstacles: Set[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """A* pathfinding algorithm."""
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

        return [start]

    def _get_neighbors(
        self,
        pos: Tuple[int, int],
        obstacles: Set[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """Get valid neighboring positions."""
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
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def _get_action_to_adjacent(
        self,
        current: Tuple[int, int],
        adjacent: Tuple[int, int]
    ) -> int:
        """Get action to move to adjacent cell."""
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
        """Navigate toward current target using A*."""
        if self.current_target_position is None:
            return 4  # stay

        robot_pos = observation['robot_position']
        objects = observation.get('objects', {})

        # Build obstacles (all objects except target)
        obstacles = set()
        for obj_id, obj_data in objects.items():
            if obj_id != self.current_target_object_id:
                obstacles.add(obj_data['position'])

        # Recalculate path if needed
        if not self.current_path or self.current_path[0] != robot_pos:
            self.current_path = self._astar(robot_pos, self.current_target_position, obstacles)

        if len(self.current_path) <= 1:
            return 4  # At target or no path

        self.current_path.pop(0)
        next_pos = self.current_path[0]
        return self._get_action_to_adjacent(robot_pos, next_pos)

    def _should_select_new_target(self, observation: dict) -> bool:
        """Check if we need to select a new target."""
        if self.current_target_object_id is None:
            return True

        objects = observation.get('objects', {})
        if self.current_target_object_id not in objects:
            return True  # Target was collected

        return False

    def _print_decision_summary(self, observation: dict, query_triggered: bool):
        """Print summary of high-level decision."""
        print("\n" + "="*80)
        print("HIGH-LEVEL DECISION (Belief-Based)")
        print("="*80)

        # Print true reward properties if available
        if self.true_reward_properties:
            print(f"True Rewarding Properties: {sorted(self.true_reward_properties)}")

        # Print belief state summary
        print(f"\nBelief State:")
        print(f"  Active Categories: {self.active_categories}")
        print(f"  Number of Features: {len(self.feature_names)}")
        print(f"  Participation Ratio: {self.belief.compute_participation_ratio():.3f} (max={len(self.feature_names)})")
        print(f"  Query Threshold: {self.pr_threshold}")
        print(f"  Queries Used: {self.queries_used}/{self.query_budget}")

        # Print expected weights
        weights = self.belief.get_expected_weights()
        sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        print(f"\n  Expected Weights (top 5):")
        for prop, weight in sorted_weights[:5]:
            marker = " *" if self.true_reward_properties and prop in self.true_reward_properties else ""
            print(f"    {prop}: {weight:+.4f}{marker}")

        # Print object utilities
        utilities = self._compute_object_utilities(observation)
        if utilities:
            print(f"\n  Object Utilities (top 5 by ratio):")
            sorted_objs = sorted(utilities.items(), key=lambda x: x[1][2], reverse=True)[:5]
            for obj_id, (u, d, r) in sorted_objs:
                marker = " <-- TARGET" if obj_id == self.current_target_object_id else ""
                print(f"    Object {obj_id}: utility={u:+.3f}, dist={d:.1f}, ratio={r:+.3f}{marker}")

        if query_triggered:
            print("\n  *** QUERY TRIGGERED ***")
            if self.query_history:
                last_query = self.query_history[-1]
                print(f"  Query: {last_query['query']}")
                print(f"  Response: {last_query['response']}")

        print("="*80 + "\n")

    def get_action(self, observation: dict, training: bool = False) -> int:
        """
        Get action using belief-based policy.

        High-level: Select object with highest expected_utility / distance
        Low-level: A* navigation to selected object

        Args:
            observation: Current observation
            training: If True, don't query

        Returns:
            Action (0=up, 1=down, 2=left, 3=right, 4=stay)
        """
        robot_can_collect = observation.get('robot_can_collect', False)
        if not robot_can_collect:
            return 4  # stay

        # First time starting
        if not self.has_started:
            self.has_started = True

        # Process any new observations (human collections)
        self._process_observations(observation)

        # Check if we need to select a new target
        if self._should_select_new_target(observation):
            query_triggered = False

            # Query decision
            if not training and self._should_query(observation):
                query_triggered = True
                weights = self._execute_query(observation)
                self.belief.update_from_linear_gaussian(
                    weights,
                    observation_noise_variance=self.lg_noise_variance
                )

            # Select new target
            target_obj_id = self._select_target(observation)
            self.current_target_object_id = target_obj_id
            self.current_path = []

            if target_obj_id is not None:
                objects = observation.get('objects', {})
                self.current_target_position = objects[target_obj_id]['position']
            else:
                self.current_target_position = None

            # Verbose output
            if self.verbose:
                self._print_decision_summary(observation, query_triggered)

        # Update target position if still exists
        if self.current_target_object_id is not None:
            objects = observation.get('objects', {})
            if self.current_target_object_id in objects:
                self.current_target_position = objects[self.current_target_object_id]['position']

        return self._get_navigation_action(observation)

    def update(self, *args, **kwargs):
        """No-op update (this agent doesn't learn during episodes)."""
        pass

    def get_query_stats(self) -> Dict:
        """Get statistics about queries and beliefs."""
        return {
            'queries_used': self.queries_used,
            'query_budget': self.query_budget,
            'query_history': self.query_history,
            'participation_ratio': self.belief.compute_participation_ratio() if self.belief else 0.0,
            'expected_weights': self.belief.get_expected_weights() if self.belief else {},
            'entropy_history': self.entropy_history,
            'decision_history': self.decision_history
        }

    def get_inferred_properties(self) -> List[Tuple[str, float]]:
        """Get inferred property weights sorted by value."""
        if self.belief is None:
            return []
        weights = self.belief.get_expected_weights()
        return sorted(weights.items(), key=lambda x: x[1], reverse=True)
