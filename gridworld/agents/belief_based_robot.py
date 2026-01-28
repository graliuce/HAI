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

    def get_utility_variance(self, properties: Dict[str, str]) -> float:
        """
        Compute variance of utility estimate for an object.

        For linear utility u = w^T x, the variance is Var(u) = x^T Σ x

        Args:
            properties: Object properties

        Returns:
            Variance of utility estimate
        """
        feature_vector = self.get_object_feature_vector(properties)
        return feature_vector @ self.covariance @ feature_vector

    def sample_weights(self, rng: np.random.RandomState = None) -> np.ndarray:
        """
        Sample a weight vector from the posterior distribution.

        Args:
            rng: Random state for sampling

        Returns:
            Sampled weight vector of shape (num_features,)
        """
        if rng is None:
            rng = np.random.RandomState()
        return rng.multivariate_normal(self.mean, self.covariance)

    def update_from_choice_plackett_luce(
        self,
        chosen_object: Dict[str, str],
        alternative_objects: List[Dict[str, str]],
        chosen_distance: float,
        alternative_distances: List[float],
        learning_rate: float = 0.1,
        num_gradient_steps: int = 50,
        convergence_tol: float = 1e-6
    ):
        """
        Update beliefs using Plackett-Luce model with Laplace approximation.

        Following the posterior update method:
        1. Find MAP estimate by maximizing log p(θ|D) via gradient ascent
        2. Compute Hessian H = Σ^{-1} + Fisher at the MAP estimate
        3. Set new covariance as H^{-1}

        Args:
            chosen_object: Properties of the object the human chose
            alternative_objects: Properties of all available objects (including chosen)
            chosen_distance: Distance from human to chosen object
            alternative_distances: Distances from human to all objects
            learning_rate: Step size for gradient updates
            num_gradient_steps: Maximum number of gradient ascent steps
            convergence_tol: Stop when gradient norm falls below this
        """
        # Build feature vectors for all objects
        chosen_features = self.get_object_feature_vector(chosen_object)
        all_features = [self.get_object_feature_vector(obj) for obj in alternative_objects]

        # Use max(dist, 1.0) to avoid division by zero
        chosen_dist = max(chosen_distance, 1.0)
        all_distances = np.array([max(d, 1.0) for d in alternative_distances])

        # Precompute weighted features (φ / c)
        weighted_features = [f / d for f, d in zip(all_features, all_distances)]
        chosen_weighted = chosen_features / chosen_dist

        # Prior precision and mean
        prior_precision = np.linalg.inv(self.covariance)
        prior_mean = self.mean.copy()

        # === Step 1: Find MAP estimate via gradient ascent ===
        theta = self.mean.copy()

        for step in range(num_gradient_steps):
            # Compute utilities and softmax probabilities
            utilities = np.array([np.dot(theta, f) for f in all_features])
            weighted_utilities = utilities / all_distances

            # Softmax probabilities (numerically stable)
            max_wu = np.max(weighted_utilities)
            exp_wu = np.exp(weighted_utilities - max_wu)
            probs = exp_wu / (np.sum(exp_wu) + 1e-10)

            # Gradient of log-likelihood
            expected_weighted = np.zeros(self.num_features)
            for p, wf in zip(probs, weighted_features):
                expected_weighted += p * wf
            grad_likelihood = chosen_weighted - expected_weighted

            # Gradient of log-prior: -Σ^{-1}(θ - μ)
            grad_prior = -prior_precision @ (theta - prior_mean)

            # Total gradient of log-posterior
            gradient = grad_likelihood + grad_prior

            # Check convergence
            if np.linalg.norm(gradient) < convergence_tol:
                break

            # Update
            theta = theta + learning_rate * gradient

        # === Step 2: Compute Hessian at MAP estimate ===
        # Recompute probabilities at final theta
        utilities = np.array([np.dot(theta, f) for f in all_features])
        weighted_utilities = utilities / all_distances
        max_wu = np.max(weighted_utilities)
        exp_wu = np.exp(weighted_utilities - max_wu)
        probs = exp_wu / (np.sum(exp_wu) + 1e-10)

        # Fisher information of likelihood
        fisher_info = np.zeros((self.num_features, self.num_features))
        for p, wf in zip(probs, weighted_features):
            fisher_info += p * np.outer(wf, wf)

        expected_weighted = np.zeros(self.num_features)
        for p, wf in zip(probs, weighted_features):
            expected_weighted += p * wf
        fisher_info -= np.outer(expected_weighted, expected_weighted)

        # Hessian of negative log-posterior = prior precision + Fisher
        H = prior_precision + fisher_info

        # === Step 3: New covariance is H^{-1} ===
        try:
            new_covariance = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            new_covariance = np.linalg.pinv(H)

        # Ensure symmetry and positive definiteness
        new_covariance = 0.5 * (new_covariance + new_covariance.T)
        eigenvalues, eigenvectors = np.linalg.eigh(new_covariance)
        min_eigenvalue = 1e-6
        eigenvalues = np.maximum(eigenvalues, min_eigenvalue)
        new_covariance = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

        # Update belief state
        self.mean = theta
        self.covariance = new_covariance

    def update_from_linear_gaussian(
        self,
        observed_weights: Dict[str, float],
        observation_noise_variance: float = 1.0
    ):
        """
        Update beliefs using Linear-Gaussian observation model.
        
        Only updates features with non-zero observed weights.
        Unmentioned features remain unchanged (no observation).
        
        For observed features, the update is:
        y_i = w_i + epsilon, where epsilon ~ N(0, R_i)
        
        Using standard Kalman filter equations for partial observations.
        """
        # Get indices and values of observed features (non-zero weights only)
        observed_indices = []
        observed_values = []
        
        for name, weight in observed_weights.items():
            if name in self.feature_to_idx and abs(weight) > 1e-10:
                observed_indices.append(self.feature_to_idx[name])
                observed_values.append(weight)
        
        if not observed_indices:
            return  # No update if nothing observed
        
        # Build observation matrix H (selects only observed features)
        num_observed = len(observed_indices)
        H = np.zeros((num_observed, self.num_features))
        for i, idx in enumerate(observed_indices):
            H[i, idx] = 1.0
        
        y = np.array(observed_values)  # Observations
        
        # Observation noise covariance (diagonal)
        R = np.eye(num_observed) * observation_noise_variance
        
        # Kalman filter update
        # Innovation: y - H*mu
        innovation = y - H @ self.mean
        
        # Innovation covariance: S = H*Sigma*H^T + R
        S = H @ self.covariance @ H.T + R
        
        # Kalman gain: K = Sigma*H^T*S^{-1}
        K = self.covariance @ H.T @ np.linalg.solve(S, np.eye(num_observed))
        
        # Update mean: mu_new = mu + K*innovation
        self.mean = self.mean + K @ innovation
        
        # Update covariance: Sigma_new = (I - K*H)*Sigma
        I = np.eye(self.num_features)
        self.covariance = (I - K @ H) @ self.covariance
        
        # Ensure symmetry and positive definiteness
        self.covariance = 0.5 * (self.covariance + self.covariance.T)
        eigenvalues, eigenvectors = np.linalg.eigh(self.covariance)
        eigenvalues = np.maximum(eigenvalues, 0.001)
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
        action_confidence_threshold: float = 0.6,
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
            participation_ratio_threshold: (DEPRECATED) Old trigger using participation ratio
            action_confidence_threshold: Trigger query when action confidence < this value.
                Action confidence measures how consistently posterior samples agree on the
                best object to target. Range [0, 1]:
                - 1.0: All samples agree on best object (high confidence, no query needed)
                - 0.5: Half the samples disagree (low confidence, should query)
                - Recommended: 0.6-0.8 (query when less than 60-80% agreement)
            plackett_luce_learning_rate: Learning rate for Plackett-Luce mean updates
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
        self.action_confidence_threshold = action_confidence_threshold
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

    def _compute_action_confidence(
        self,
        observation: dict,
        num_samples: int = 50
    ) -> Tuple[float, int, Dict]:
        """
        Compute action confidence using Thompson sampling.

        Samples weight vectors from the posterior, finds the best object for each
        sample, and measures how often samples agree on the best choice.

        Args:
            observation: Current observation with objects
            num_samples: Number of posterior samples

        Returns:
            Tuple of:
            - confidence: Agreement rate in [0, 1] (1 = all samples agree)
            - best_obj_id: Most commonly chosen object
            - details: Dict with sampling details
        """
        objects = observation.get('objects', {})
        robot_pos = observation['robot_position']

        if not objects:
            return 1.0, None, {}

        # Precompute feature vectors and distances
        obj_ids = list(objects.keys())
        feature_vectors = []
        distances = []
        for obj_id in obj_ids:
            obj_data = objects[obj_id]
            feature_vectors.append(
                self.belief.get_object_feature_vector(obj_data['properties'])
            )
            pos = obj_data['position']
            dist = max(1.0, np.sqrt(
                (pos[0] - robot_pos[0])**2 + (pos[1] - robot_pos[1])**2
            ))
            distances.append(dist)

        feature_matrix = np.array(feature_vectors)  # (num_objects, num_features)
        distances = np.array(distances)  # (num_objects,)

        # Sample from posterior and count votes
        vote_counts = defaultdict(int)

        for _ in range(num_samples):
            # Sample weights from posterior
            sampled_weights = self.belief.sample_weights(self.rng)

            # Compute utility/distance for all objects with sampled weights
            utilities = feature_matrix @ sampled_weights  # (num_objects,)
            ratios = utilities / distances

            # Find best object
            best_idx = np.argmax(ratios)
            best_obj_id = obj_ids[best_idx]
            vote_counts[best_obj_id] += 1

        # Find most voted object and compute confidence
        most_voted_obj = max(vote_counts.keys(), key=lambda x: vote_counts[x])
        confidence = vote_counts[most_voted_obj] / num_samples

        details = {
            'vote_counts': dict(vote_counts),
            'num_samples': num_samples,
            'num_objects': len(obj_ids)
        }

        return confidence, most_voted_obj, details

    def _compute_feature_votes_from_thompson(
        self,
        observation: dict,
        thompson_details: dict
    ) -> Dict[str, float]:
        """
        Compute normalized feature vote distribution from Thompson sampling results.
        
        For each feature, count how many votes went to objects containing that feature,
        then normalize by total votes.
        
        Args:
            observation: Current observation with objects
            thompson_details: Details dict from _compute_action_confidence containing vote_counts
            
        Returns:
            Dict mapping feature names to normalized vote frequencies
        """
        vote_counts = thompson_details.get('vote_counts', {})
        if not vote_counts:
            # Return uniform distribution if no votes
            return {feature: 0.0 for feature in self.feature_names}
        
        objects = observation.get('objects', {})
        total_votes = thompson_details.get('num_samples', sum(vote_counts.values()))
        
        # Count votes for each feature
        feature_votes = defaultdict(float)
        
        for obj_id, votes in vote_counts.items():
            if obj_id in objects:
                # Get all features (property values) of this object
                obj_properties = objects[obj_id]['properties']
                for prop_value in obj_properties.values():
                    if prop_value in self.belief.feature_to_idx:
                        feature_votes[prop_value] += votes
        
        # Normalize by total votes
        normalized_votes = {}
        for feature in self.feature_names:
            normalized_votes[feature] = feature_votes.get(feature, 0.0) / total_votes
        
        return normalized_votes

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
        Decide whether to trigger a query based on action confidence.

        Action confidence uses Thompson sampling: sample weight vectors from the
        posterior, find the best object for each sample, and measure agreement.
        Low agreement means uncertainty about which object to target → query.

        Query if:
        1. We have query budget remaining
        2. Action confidence < threshold (uncertain about best action)

        Returns:
            True if should query
        """
        if self.llm is None:
            return False

        if self.queries_used >= self.query_budget:
            return False

        # Compute action confidence using Thompson sampling
        confidence, best_obj, details = self._compute_action_confidence(observation)
        should_query = confidence < self.action_confidence_threshold

        # Also compute PR for logging (but don't use for decision)
        pr = self.belief.compute_participation_ratio()

        # Store for analysis
        self.entropy_history.append({
            'participation_ratio': pr,
            'action_confidence': confidence,
            'threshold': self.action_confidence_threshold,
            'should_query': should_query,
            'queries_used': self.queries_used,
            'vote_details': details
        })

        if self.verbose:
            print(f"[Query Decision] Action Confidence={confidence:.3f} "
                  f"(threshold={self.action_confidence_threshold}), "
                  f"PR={pr:.3f}, query={should_query}")
            if details.get('vote_counts'):
                print(f"  Vote distribution: {details['vote_counts']}")

        return should_query

    def _execute_query(self, observation: dict) -> Tuple[Dict[str, float], str, str]:
        """
        Execute a query to the human and return extracted weights.

        Returns:
            Tuple of (weights dict, query string, response string)
        """
        if self.llm is None or self.human_responder is None:
            return {}, "", ""

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

        # Compute Thompson sampling vote distribution for features
        # Run Thompson sampling to see which objects are selected
        confidence, best_obj, details = self._compute_action_confidence(observation, num_samples=100)
        
        # Extract feature vote counts from Thompson sampling
        thompson_votes = self._compute_feature_votes_from_thompson(observation, details)

        # Print Thompson sampling vote distribution if verbose
        if self.verbose:
            print("\n[Thompson Sampling Vote Distribution]")
            print(f"{'Feature':<15} | {'Normalized Value':<20}")
            print("-" * 38)
            # Sort by normalized value (descending) for better readability
            sorted_votes = sorted(thompson_votes.items(), key=lambda x: x[1], reverse=True)
            for feature, norm_value in sorted_votes:
                print(f"{feature:<15} | {norm_value:<20.3f}")
            print()

        # Generate query
        query = self.llm.generate_query(
            board_properties,
            thompson_votes,
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

        return property_weights, query, response

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

        # Store previous belief state for comparison
        if self.verbose:
            prev_mean = self.belief.mean.copy()
            prev_variances = np.array([self.belief.covariance[i, i] for i in range(self.belief.num_features)])

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
            self._print_belief_update_table(
                prev_mean, 
                prev_variances, 
                chosen_object, 
                chosen_distance,
                len(all_objects_props) - 1
            )

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

    def _print_belief_update_table(
        self, 
        prev_mean: np.ndarray,
        prev_variances: np.ndarray,
        chosen_object: dict,
        chosen_distance: float,
        num_alternatives: int
    ):
        """Print a detailed table of belief updates from human observation."""
        print("\n" + "=" * 120)
        print("BELIEF UPDATE: Human Collected Object")
        print("=" * 120)
        
        print(f"\nObject Properties: {chosen_object['properties']}")
        print(f"Distance to Object: {chosen_distance:.2f}")
        print(f"Number of Alternative Objects: {num_alternatives}")
        
        # Get current mean and variances
        curr_mean = self.belief.mean
        curr_variances = np.array([self.belief.covariance[i, i] for i in range(self.belief.num_features)])
        
        # Calculate deltas
        mean_deltas = curr_mean - prev_mean
        variance_deltas = curr_variances - prev_variances
        
        print(f"\nBelief Updates (All Features):")
        print("-" * 120)
        print(f"{'Feature':<15} {'Prev Mean':>12} {'New Mean':>12} {'Δ Mean':>12} {'Prev Var':>12} {'New Var':>12} {'Δ Var':>12}")
        print("-" * 120)
        
        # Create list of (feature, abs_mean_delta) for sorting
        feature_deltas = []
        for i, feature in enumerate(self.feature_names):
            feature_deltas.append((i, feature, abs(mean_deltas[i])))
        
        # Sort by absolute mean delta (largest changes first), then show all
        feature_deltas.sort(key=lambda x: x[2], reverse=True)
        
        for i, feature, _ in feature_deltas:
            marker = " *" if self.true_reward_properties and feature in self.true_reward_properties else ""
            print(f"{feature:<15} {prev_mean[i]:>+12.4f} {curr_mean[i]:>+12.4f} {mean_deltas[i]:>+12.4f} "
                  f"{prev_variances[i]:>12.4f} {curr_variances[i]:>12.4f} {variance_deltas[i]:>+12.4f}{marker}")
        
        print("-" * 120)
        print("* indicates true rewarding property")
        print("=" * 120 + "\n")

    def _print_query_update_table(
        self,
        prev_mean: np.ndarray,
        prev_variances: np.ndarray,
        query: str,
        response: str,
        weights: Dict[str, float]
    ):
        """Print a detailed table of belief updates from query response."""
        print("\n" + "=" * 120)
        print("BELIEF UPDATE: Query Response")
        print("=" * 120)
        
        print(f"\nQuery: {query}")
        print(f"Response: {response}")
        
        # Print LLM-extracted weights for all features
        print(f"\nLLM-Extracted Weights (L2 Normalized):")
        print("-" * 80)
        print(f"{'Feature':<15} {'LLM Weight':>15} {'True Reward?':<15}")
        print("-" * 80)
        
        # Sort by absolute weight value (largest first)
        sorted_weights = sorted(weights.items(), key=lambda x: abs(x[1]), reverse=True)
        for feature, weight in sorted_weights:
            is_true = "YES *" if self.true_reward_properties and feature in self.true_reward_properties else ""
            print(f"{feature:<15} {weight:>+15.6f} {is_true:<15}")
        
        # Show features not mentioned by LLM (weight = 0)
        mentioned_features = set(weights.keys())
        unmentioned = [f for f in self.feature_names if f not in mentioned_features]
        if unmentioned:
            print(f"\n  Features not mentioned (weight = 0): {', '.join(unmentioned)}")
        
        print("-" * 80)
        
        # Get current mean and variances
        curr_mean = self.belief.mean
        curr_variances = np.array([self.belief.covariance[i, i] for i in range(self.belief.num_features)])
        
        # Calculate deltas
        mean_deltas = curr_mean - prev_mean
        variance_deltas = curr_variances - prev_variances
        
        print(f"\nBelief Updates After Incorporating Query (All Features):")
        print("-" * 120)
        print(f"{'Feature':<15} {'Prev Mean':>12} {'New Mean':>12} {'Δ Mean':>12} {'Prev Var':>12} {'New Var':>12} {'Δ Var':>12}")
        print("-" * 120)
        
        # Create list of (feature, abs_mean_delta) for sorting
        feature_deltas = []
        for i, feature in enumerate(self.feature_names):
            feature_deltas.append((i, feature, abs(mean_deltas[i])))
        
        # Sort by absolute mean delta (largest changes first), then show all
        feature_deltas.sort(key=lambda x: x[2], reverse=True)
        
        for i, feature, _ in feature_deltas:
            marker = " *" if self.true_reward_properties and feature in self.true_reward_properties else ""
            print(f"{feature:<15} {prev_mean[i]:>+12.4f} {curr_mean[i]:>+12.4f} {mean_deltas[i]:>+12.4f} "
                  f"{prev_variances[i]:>12.4f} {curr_variances[i]:>12.4f} {variance_deltas[i]:>+12.4f}{marker}")
        
        print("-" * 120)
        print("* indicates true rewarding property")
        print("=" * 120 + "\n")

    def _print_decision_summary(self, observation: dict, query_triggered: bool):
        """Print summary of high-level decision."""
        print("\n" + "="*120)
        print("HIGH-LEVEL DECISION POINT (Robot Choosing Object)")
        print("="*120)

        # Compute action confidence for display
        confidence, best_obj, details = self._compute_action_confidence(observation)

        # Print belief state summary
        print(f"\nCurrent Belief State:")
        print(f"  Active Categories: {self.active_categories}")
        print(f"  Number of Features: {len(self.feature_names)}")
        print(f"  Action Confidence: {confidence:.3f} (threshold={self.action_confidence_threshold})")
        if details.get('vote_counts'):
            print(f"    Vote distribution across objects: {details['vote_counts']}")
        print(f"  Queries Used: {self.queries_used}/{self.query_budget}")

        # Print expected weights in a table
        weights = self.belief.get_expected_weights()
        sorted_weights = sorted(weights.items(), key=lambda x: abs(x[1]), reverse=True)
        print(f"\n  Expected Feature Weights:")
        print(f"  {'-'*60}")
        print(f"  {'Feature':<15} {'Weight':>12} {'Variance':>12}")
        print(f"  {'-'*60}")
        for prop, weight in sorted_weights:
            variance = self.belief.get_feature_variance(prop)
            marker = " *" if self.true_reward_properties and prop in self.true_reward_properties else ""
            print(f"  {prop:<15} {weight:>+12.4f} {variance:>12.4f}{marker}")
        print(f"  {'-'*60}")
        print(f"  * indicates true rewarding property")

        # Print object utilities in a detailed table
        utilities = self._compute_object_utilities(observation)
        if utilities:
            objects = observation.get('objects', {})
            print(f"\n  All Object Utilities and Decision:")
            print(f"  {'-'*120}")
            print(f"  {'Obj ID':<8} {'Properties':<40} {'Utility':>12} {'Distance':>10} {'Ratio':>12} {'Selected':<10}")
            print(f"  {'-'*120}")
            sorted_objs = sorted(utilities.items(), key=lambda x: x[1][2], reverse=True)
            for obj_id, (u, d, r) in sorted_objs:
                props_str = ', '.join([f"{k}:{v}" for k, v in objects[obj_id]['properties'].items()])
                selected = ">>> YES" if obj_id == self.current_target_object_id else ""
                print(f"  {obj_id:<8} {props_str:<40} {u:>+12.4f} {d:>10.2f} {r:>+12.4f} {selected:<10}")
            print(f"  {'-'*120}")
            
            # Highlight the chosen object
            if self.current_target_object_id is not None and self.current_target_object_id in utilities:
                chosen_u, chosen_d, chosen_r = utilities[self.current_target_object_id]
                chosen_props = objects[self.current_target_object_id]['properties']
                print(f"\n  CHOSEN OBJECT: {self.current_target_object_id}")
                print(f"    Properties: {chosen_props}")
                print(f"    Estimated Utility: {chosen_u:+.4f}")
                print(f"    Distance: {chosen_d:.2f}")
                print(f"    Estimated Utility/Distance: {chosen_r:+.4f}")

        print("="*120 + "\n")

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
                
                # Store previous belief state for comparison
                if self.verbose:
                    prev_mean = self.belief.mean.copy()
                    prev_variances = np.array([self.belief.covariance[i, i] for i in range(self.belief.num_features)])
                
                weights, query, response = self._execute_query(observation)
                self.belief.update_from_linear_gaussian(
                    weights,
                    observation_noise_variance=self.lg_noise_variance
                )
                
                # Print query update table
                if self.verbose:
                    self._print_query_update_table(prev_mean, prev_variances, query, response, weights)

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
