"""Belief-based robot agent with explicit Bayesian preference modeling."""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import heapq
import numpy as np
from itertools import combinations, product

from ..objects import PROPERTY_CATEGORIES, PROPERTY_VALUES
from ..llm_interface import LLMInterface, SimulatedHumanResponder


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function."""
    return np.where(x >= 0, 
                    1 / (1 + np.exp(-x)), 
                    np.exp(x) / (1 + np.exp(x)))


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

    def compute_differential_entropy(self) -> float:
        """
        Compute the differential entropy of the Gaussian belief.
        
        H[θ] = 0.5 * (d * ln(2πe) + ln(det(Σ)))
        
        Returns:
            Differential entropy in nats
        """
        d = self.num_features
        sign, logdet = np.linalg.slogdet(self.covariance)
        if sign <= 0:
            # Use pseudo-determinant for numerical stability
            eigenvalues = np.linalg.eigvalsh(self.covariance)
            eigenvalues = np.maximum(eigenvalues, 1e-10)
            logdet = np.sum(np.log(eigenvalues))
        
        return 0.5 * (d * np.log(2 * np.pi * np.e) + logdet)

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

    def update_from_pairwise_choice_bradley_terry(
        self,
        chosen_features: np.ndarray,
        rejected_features: np.ndarray,
        learning_rate: float = 0.1,
        num_gradient_steps: int = 50,
        convergence_tol: float = 1e-6
    ):
        """
        Update beliefs from a pairwise preference observation using Bradley-Terry model.
        
        p(chosen | chosen, rejected, θ) = σ(θ · (chosen - rejected))
        
        Uses Laplace approximation (MAP + Hessian) for posterior update.
        
        Args:
            chosen_features: Feature vector of chosen option
            rejected_features: Feature vector of rejected option
            learning_rate: Step size for gradient updates
            num_gradient_steps: Maximum number of gradient ascent steps
            convergence_tol: Stop when gradient norm falls below this
        """
        # Difference vector
        diff = chosen_features - rejected_features
        
        # Prior precision and mean
        prior_precision = np.linalg.inv(self.covariance)
        prior_mean = self.mean.copy()
        
        # === Step 1: Find MAP estimate via gradient ascent ===
        theta = self.mean.copy()
        
        for step in range(num_gradient_steps):
            # Bradley-Terry probability
            logit = np.dot(theta, diff)
            prob = sigmoid(np.array([logit]))[0]
            
            # Gradient of log-likelihood: (1 - p(chosen)) * diff
            grad_likelihood = (1 - prob) * diff
            
            # Gradient of log-prior: -Σ^{-1}(θ - μ)
            grad_prior = -prior_precision @ (theta - prior_mean)
            
            # Total gradient
            gradient = grad_likelihood + grad_prior
            
            if np.linalg.norm(gradient) < convergence_tol:
                break
            
            theta = theta + learning_rate * gradient
        
        # === Step 2: Compute Hessian at MAP ===
        logit = np.dot(theta, diff)
        prob = sigmoid(np.array([logit]))[0]
        
        # Hessian of negative log-likelihood: p(1-p) * diff * diff^T
        fisher_info = prob * (1 - prob) * np.outer(diff, diff)
        
        # Hessian of negative log-posterior
        H = prior_precision + fisher_info
        
        # === Step 3: New covariance is H^{-1} ===
        try:
            new_covariance = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            new_covariance = np.linalg.pinv(H)
        
        # Ensure symmetry and positive definiteness
        new_covariance = 0.5 * (new_covariance + new_covariance.T)
        eigenvalues, eigenvectors = np.linalg.eigh(new_covariance)
        eigenvalues = np.maximum(eigenvalues, 1e-6)
        new_covariance = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        
        self.mean = theta
        self.covariance = new_covariance


class EIGQuerySelector:
    """
    Selects optimal pairwise comparison queries using Expected Information Gain (EIG).
    
    Based on the BOED framework from Handa et al. (2024):
    - Uses Bradley-Terry model for pairwise preferences
    - Estimates EIG via Monte Carlo sampling from the posterior
    - Queries are pairs of binary feature vectors
    """
    
    def __init__(
        self,
        num_features: int,
        features_per_option: int = 2,
        num_mc_samples: int = 100,
        num_candidate_queries: int = 500,
        rng: np.random.RandomState = None
    ):
        """
        Args:
            num_features: Total number of features
            features_per_option: Number of features active per option (K in the paper)
            num_mc_samples: Number of Monte Carlo samples for EIG estimation
            num_candidate_queries: Number of candidate queries to evaluate
            rng: Random state
        """
        self.num_features = num_features
        self.features_per_option = features_per_option
        self.num_mc_samples = num_mc_samples
        self.num_candidate_queries = num_candidate_queries
        self.rng = rng if rng is not None else np.random.RandomState()
    
    def _generate_candidate_queries(
        self,
        available_feature_indices: List[int] = None
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate candidate pairwise comparison queries.
        
        Each query is a pair (option_a, option_b) where each option is a binary
        feature vector with exactly `features_per_option` active features.
        
        Args:
            available_feature_indices: If provided, only use these feature indices
            
        Returns:
            List of (option_a, option_b) tuples
        """
        if available_feature_indices is None:
            available_feature_indices = list(range(self.num_features))
        
        candidates = []
        
        # If we have few enough features, enumerate all combinations
        n_available = len(available_feature_indices)
        k = min(self.features_per_option, n_available)
        
        # Generate all possible options with k active features
        all_options = list(combinations(available_feature_indices, k))
        
        if len(all_options) * (len(all_options) - 1) // 2 <= self.num_candidate_queries:
            # Enumerate all pairs
            for i, opt_a_indices in enumerate(all_options):
                for opt_b_indices in all_options[i+1:]:
                    # Create binary vectors
                    opt_a = np.zeros(self.num_features)
                    opt_b = np.zeros(self.num_features)
                    for idx in opt_a_indices:
                        opt_a[idx] = 1.0
                    for idx in opt_b_indices:
                        opt_b[idx] = 1.0
                    candidates.append((opt_a, opt_b))
        else:
            # Sample random queries
            for _ in range(self.num_candidate_queries):
                # Sample indices for option A
                opt_a_indices = self.rng.choice(
                    available_feature_indices, size=k, replace=False
                )
                # Sample indices for option B (different from A)
                remaining = [i for i in available_feature_indices if i not in opt_a_indices]
                if len(remaining) >= k:
                    opt_b_indices = self.rng.choice(remaining, size=k, replace=False)
                else:
                    # Allow some overlap if not enough remaining
                    opt_b_indices = self.rng.choice(
                        available_feature_indices, size=k, replace=False
                    )
                
                opt_a = np.zeros(self.num_features)
                opt_b = np.zeros(self.num_features)
                for idx in opt_a_indices:
                    opt_a[idx] = 1.0
                for idx in opt_b_indices:
                    opt_b[idx] = 1.0
                
                candidates.append((opt_a, opt_b))
        
        return candidates
    
    def _compute_bradley_terry_prob(
        self,
        theta: np.ndarray,
        option_a: np.ndarray,
        option_b: np.ndarray
    ) -> float:
        """
        Compute Bradley-Terry probability p(choose A | A, B, θ).
        
        p(A | A, B, θ) = σ(θ · (A - B))
        """
        diff = option_a - option_b
        logit = np.dot(theta, diff)
        return sigmoid(np.array([logit]))[0]
    
    def _estimate_eig(
        self,
        belief: GaussianBeliefState,
        option_a: np.ndarray,
        option_b: np.ndarray
    ) -> float:
        """
        Estimate Expected Information Gain for a pairwise query using Monte Carlo.
        
        EIG(q) = E_{p(y|q)}[H[θ] - H[θ|y,q]]
               = H[θ] - E_{p(y|q)}[H[θ|y,q]]
        
        For Bradley-Terry model with Gaussian posterior approximation:
        1. Sample θ from current posterior
        2. For each θ, compute p(y=A|θ) and p(y=B|θ)
        3. Estimate marginal p(y=A) and p(y=B) by averaging
        4. Compute expected posterior entropy via Laplace approximation
        
        Args:
            belief: Current belief state
            option_a: Feature vector for option A
            option_b: Feature vector for option B
            
        Returns:
            Estimated EIG in nats
        """
        # Current entropy H[θ]
        prior_entropy = belief.compute_differential_entropy()
        
        # Sample theta values from posterior
        theta_samples = []
        for _ in range(self.num_mc_samples):
            theta_samples.append(belief.sample_weights(self.rng))
        theta_samples = np.array(theta_samples)
        
        # Compute p(y=A|θ) for each sample
        probs_a = np.array([
            self._compute_bradley_terry_prob(theta, option_a, option_b)
            for theta in theta_samples
        ])
        probs_b = 1 - probs_a
        
        # Marginal probabilities (average over samples)
        p_a = np.mean(probs_a)
        p_b = np.mean(probs_b)
        
        # For numerical stability
        p_a = np.clip(p_a, 1e-10, 1 - 1e-10)
        p_b = np.clip(p_b, 1e-10, 1 - 1e-10)
        
        # Estimate posterior entropy for each outcome via Laplace approximation
        # H[θ|y,q] ≈ 0.5 * (d * ln(2πe) + ln(det(Σ_post)))
        
        diff = option_a - option_b
        
        # For y=A: The posterior precision increases by Fisher info
        # Fisher = p(1-p) * diff * diff^T, where p = σ(θ·diff)
        # We estimate this at the posterior mean
        
        def estimate_posterior_entropy(choice_a: bool) -> float:
            """Estimate posterior entropy after observing choice."""
            # Compute expected Fisher information at posterior mean
            logit = np.dot(belief.mean, diff)
            prob = sigmoid(np.array([logit]))[0]
            
            # Fisher information
            fisher_info = prob * (1 - prob) * np.outer(diff, diff)
            
            # Posterior precision = prior precision + Fisher
            try:
                prior_precision = np.linalg.inv(belief.covariance)
            except np.linalg.LinAlgError:
                prior_precision = np.linalg.pinv(belief.covariance)
            
            posterior_precision = prior_precision + fisher_info
            
            # Posterior covariance
            try:
                posterior_cov = np.linalg.inv(posterior_precision)
            except np.linalg.LinAlgError:
                posterior_cov = np.linalg.pinv(posterior_precision)
            
            # Ensure positive definiteness
            eigenvalues = np.linalg.eigvalsh(posterior_cov)
            eigenvalues = np.maximum(eigenvalues, 1e-10)
            logdet = np.sum(np.log(eigenvalues))
            
            d = belief.num_features
            return 0.5 * (d * np.log(2 * np.pi * np.e) + logdet)
        
        # The posterior entropy is approximately the same regardless of outcome
        # (since we're using Laplace approximation at the same point)
        posterior_entropy = estimate_posterior_entropy(True)
        
        # EIG = H[θ] - E[H[θ|y]]
        # Since posterior entropy is approximately the same for both outcomes:
        eig = prior_entropy - posterior_entropy
        
        return max(0, eig)  # EIG should be non-negative
    
    def select_optimal_query(
        self,
        belief: GaussianBeliefState,
        available_feature_indices: List[int] = None,
        verbose: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Select the pairwise query with highest Expected Information Gain.
        
        Args:
            belief: Current belief state
            available_feature_indices: If provided, only use these features
            verbose: Print debug info
            
        Returns:
            Tuple of (option_a, option_b, eig_value)
        """
        candidates = self._generate_candidate_queries(available_feature_indices)
        
        if not candidates:
            # Fallback: random query
            opt_a = np.zeros(self.num_features)
            opt_b = np.zeros(self.num_features)
            indices = list(range(self.num_features))
            k = min(self.features_per_option, len(indices))
            for idx in self.rng.choice(indices, size=k, replace=False):
                opt_a[idx] = 1.0
            remaining = [i for i in indices if opt_a[i] == 0]
            if len(remaining) >= k:
                for idx in self.rng.choice(remaining, size=k, replace=False):
                    opt_b[idx] = 1.0
            else:
                for idx in self.rng.choice(indices, size=k, replace=False):
                    opt_b[idx] = 1.0
            return opt_a, opt_b, 0.0
        
        best_query = None
        best_eig = -float('inf')
        
        for opt_a, opt_b in candidates:
            eig = self._estimate_eig(belief, opt_a, opt_b)
            if eig > best_eig:
                best_eig = eig
                best_query = (opt_a, opt_b)
        
        if verbose:
            print(f"[EIG Query Selection] Best EIG: {best_eig:.4f}")
            print(f"  Option A features: {np.where(best_query[0] == 1)[0]}")
            print(f"  Option B features: {np.where(best_query[1] == 1)[0]}")
        
        return best_query[0], best_query[1], best_eig
    
    def compute_realized_information_gain(
        self,
        belief_before: GaussianBeliefState,
        belief_after: GaussianBeliefState
    ) -> float:
        """
        Compute the realized information gain from a query.
        
        IG = H[θ_before] - H[θ_after]
        
        Args:
            belief_before: Belief state before query
            belief_after: Belief state after query
            
        Returns:
            Realized information gain in nats
        """
        entropy_before = belief_before.compute_differential_entropy()
        entropy_after = belief_after.compute_differential_entropy()
        return max(0, entropy_before - entropy_after)


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
        query_mode: str = "sampled_actions",
        action_confidence_threshold: float = 0.6,
        plackett_luce_learning_rate: float = 0.1,
        plackett_luce_gradient_steps: int = 5,
        linear_gaussian_noise_variance: float = 1.0,
        eig_num_mc_samples: int = 100,
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
            query_mode: Mode for query generation. Options:
                - "sampled_actions": Include Thompson sampling votes in prompt (default)
                - "state": Include objects on board with properties/distance and human-collected objects
                - "beliefs": Include unique features with distance and robot beliefs (mean/variance)
                - "preference": Ask which object the human prefers between two objects
                - "eig": Use Expected Information Gain to select optimal pairwise query.
                         Each option will have exactly one feature per active property category.
            action_confidence_threshold: Trigger query when action confidence < this value.
                Action confidence measures how consistently posterior samples agree on the
                best object to target. Range [0, 1]:
                - 1.0: All samples agree on best object (high confidence, no query needed)
                - 0.5: Half the samples disagree (low confidence, should query)
                - Recommended: 0.6-0.8 (query when less than 60-80% agreement)
            plackett_luce_learning_rate: Learning rate for Plackett-Luce mean updates
            plackett_luce_gradient_steps: Number of gradient steps per observation
            linear_gaussian_noise_variance: Observation noise for linear-Gaussian updates
            eig_num_mc_samples: Number of MC samples for EIG estimation
            verbose: Print debug information
            seed: Random seed
        """
        self.grid_size = grid_size
        self.num_objects = num_objects
        self.active_categories = active_categories or PROPERTY_CATEGORIES[:2]
        self.num_property_values = num_property_values
        self.llm = llm_interface
        self.query_budget = query_budget
        self.query_mode = query_mode
        self.action_confidence_threshold = action_confidence_threshold
        self.pl_learning_rate = plackett_luce_learning_rate
        self.pl_gradient_steps = plackett_luce_gradient_steps
        self.lg_noise_variance = linear_gaussian_noise_variance
        self.eig_num_mc_samples = eig_num_mc_samples
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

        # EIG query selector (initialized when needed)
        self.eig_selector: Optional[EIGQuerySelector] = None

        # Current target tracking
        self.current_target_object_id: Optional[int] = None
        self.current_target_position: Optional[Tuple[int, int]] = None
        self.current_path: List[Tuple[int, int]] = []

        # Query tracking
        self.queries_used: int = 0
        self.query_history: List[Dict] = []

        # Information gain tracking
        self.information_gain_history: List[Dict] = []
        # Convenience lists for different IG sources
        self.query_information_gain_history: List[Dict] = []
        self.observation_information_gain_history: List[Dict] = []

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

        # Initialize EIG selector if using EIG mode
        if self.query_mode == "eig":
            # For EIG mode, each option has one feature per active category
            features_per_option = len(self.active_categories)
            self.eig_selector = EIGQuerySelector(
                num_features=len(self.feature_names),
                features_per_option=features_per_option,
                num_mc_samples=self.eig_num_mc_samples,
                rng=self.rng
            )

        if self.verbose:
            print(f"\n[Belief Reset]")
            print(f"  Active Categories: {self.active_categories}")
            print(f"  Number of Features: {len(self.feature_names)}")
            print(f"  Initial Participation Ratio: {self.belief.compute_participation_ratio():.3f}")
            print(f"  Initial Entropy: {self.belief.compute_differential_entropy():.3f}")

        # Reset target tracking
        self.current_target_object_id = None
        self.current_target_position = None
        self.current_path = []

        # Reset query tracking
        self.queries_used = 0
        self.query_history = []

        # Reset information gain tracking
        self.information_gain_history = []
        self.query_information_gain_history = []
        self.observation_information_gain_history = []

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

    def _execute_query(self, observation: dict) -> Tuple[Dict[str, float], str, str, float]:
        """
        Execute a query to the human and return extracted weights.

        The query generation method depends on self.query_mode:
        - "sampled_actions": Use Thompson sampling votes (default)
        - "state": Include objects with properties/distance and human-collected objects
        - "beliefs": Include unique features with distance and robot beliefs (mean/variance)
        - "preference": Ask which object the human prefers between two objects
        - "eig": Use Expected Information Gain to select optimal pairwise query

        Returns:
            Tuple of (weights dict, query string, response string, entropy_before)
        """
        if self.llm is None or self.human_responder is None:
            return {}, "", "", 0.0

        # Store belief state before query for information gain calculation
        belief_before_mean = self.belief.mean.copy()
        belief_before_cov = self.belief.covariance.copy()
        entropy_before = self.belief.compute_differential_entropy()

        # Gather board properties
        objects = observation.get('objects', {})
        robot_position = observation.get('robot_position', (0, 0))
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

        # Generate query based on mode
        if self.query_mode == "state":
            query = self._generate_query_state_mode(observation, robot_position)
            response = self.human_responder.respond_to_query(
                query, board_properties, collected_properties
            )
            property_weights = self.llm.interpret_response(
                query, response, board_properties, collected_properties,
                self.feature_names, self.active_categories
            )
            update_method = "linear_gaussian"
        elif self.query_mode == "beliefs":
            query = self._generate_query_beliefs_mode(observation, robot_position)
            response = self.human_responder.respond_to_query(
                query, board_properties, collected_properties
            )
            property_weights = self.llm.interpret_response(
                query, response, board_properties, collected_properties,
                self.feature_names, self.active_categories
            )
            update_method = "linear_gaussian"
        elif self.query_mode == "preference":
            result = self._execute_preference_query(
                observation, robot_position, board_properties, collected_properties
            )
            property_weights, query, response = result
            update_method = "linear_gaussian"
        elif self.query_mode == "eig":
            result = self._execute_eig_query(
                observation, robot_position, board_properties, collected_properties
            )
            property_weights, query, response = result
            update_method = "bradley_terry"
        else:  # "sampled_actions" (default)
            query = self._generate_query_sampled_actions_mode(observation, board_properties)
            response = self.human_responder.respond_to_query(
                query, board_properties, collected_properties
            )
            property_weights = self.llm.interpret_response(
                query, response, board_properties, collected_properties,
                self.feature_names, self.active_categories
            )
            update_method = "linear_gaussian"

        # Normalize weights using L2 norm (for non-EIG modes)
        if update_method == "linear_gaussian":
            weights_array = np.array([property_weights.get(f, 0.0) for f in self.feature_names])
            l2_norm = np.linalg.norm(weights_array)
            if l2_norm > 1e-10:
                weights_array = weights_array / l2_norm
                property_weights = {
                    f: weights_array[i] for i, f in enumerate(self.feature_names)
                }

        # For EIG mode, belief update already happened inside _execute_eig_query
        # For non-EIG modes, belief update will happen in caller
        # In both cases, information gain will be computed in caller after belief update
        
        # Store basic query info in history (information_gain will be updated later)
        self.query_history.append({
            'query': query,
            'response': response,
            'weights': property_weights,
            'board_properties': board_properties,
            'collected_properties': collected_properties,
            'query_mode': self.query_mode,
            'information_gain': 0.0  # Will be updated after belief update
        })
        
        return property_weights, query, response, entropy_before

    def _generate_query_sampled_actions_mode(self, observation: dict, board_properties: List[str]) -> str:
        """Generate query using Thompson sampling votes (default mode)."""
        # Compute Thompson sampling vote distribution for features
        confidence, best_obj, details = self._compute_action_confidence(observation, num_samples=100)

        # Extract feature vote counts from Thompson sampling
        thompson_votes = self._compute_feature_votes_from_thompson(observation, details)

        # Print Thompson sampling vote distribution if verbose
        if self.verbose:
            print("\n[Thompson Sampling Vote Distribution]")
            print(f"{'Feature':<15} | {'Normalized Value':<20}")
            print("-" * 38)
            sorted_votes = sorted(thompson_votes.items(), key=lambda x: x[1], reverse=True)
            for feature, norm_value in sorted_votes:
                print(f"{feature:<15} | {norm_value:<20.3f}")
            print()

        return self.llm.generate_query(
            board_properties,
            thompson_votes,
            self.active_categories
        )

    def _generate_query_state_mode(self, observation: dict, robot_position: Tuple[int, int]) -> str:
        """Generate query using objects on board with properties/distance and human-collected objects."""
        objects = observation.get('objects', {})
        human_collected = observation.get('human_collected', [])

        # Format objects info
        objects_info = []
        for obj_id, obj_data in objects.items():
            objects_info.append({
                'id': obj_id,
                'position': obj_data['position'],
                'properties': obj_data['properties']
            })

        # Format collected objects info
        collected_objects_info = []
        for collected in human_collected:
            collected_objects_info.append({
                'id': collected.get('id', 'unknown'),
                'properties': collected['properties']
            })

        if self.verbose:
            print(f"\n[Query Mode: state]")
            print(f"  Objects on board: {len(objects_info)}")
            print(f"  Human collected: {len(collected_objects_info)}")

        return self.llm.generate_query_with_state(
            objects_info,
            collected_objects_info,
            robot_position
        )

    def _generate_query_beliefs_mode(self, observation: dict, robot_position: Tuple[int, int]) -> str:
        """Generate query using unique features with distance and robot beliefs."""
        objects = observation.get('objects', {})

        # Compute feature distances (distance to closest object with each feature)
        feature_distances = {}
        for feature in self.feature_names:
            min_dist = float('inf')
            for obj_data in objects.values():
                if feature in obj_data['properties'].values():
                    pos = obj_data['position']
                    dist = ((pos[0] - robot_position[0])**2 + (pos[1] - robot_position[1])**2)**0.5
                    min_dist = min(min_dist, dist)
            if min_dist < float('inf'):
                feature_distances[feature] = min_dist

        # Get belief mean and variance for each feature
        belief_mean = {}
        belief_variance = {}
        if self.belief is not None:
            for i, feature in enumerate(self.feature_names):
                belief_mean[feature] = self.belief.mean[i]
                belief_variance[feature] = self.belief.covariance[i, i]

        if self.verbose:
            print(f"\n[Query Mode: beliefs]")
            print(f"  Features on board: {len(feature_distances)}")
            if belief_mean:
                print("  Top 5 features by mean:")
                sorted_by_mean = sorted(belief_mean.items(), key=lambda x: x[1], reverse=True)[:5]
                for feat, mean in sorted_by_mean:
                    var = belief_variance.get(feat, 0)
                    print(f"    {feat}: mean={mean:.3f}, var={var:.3f}")

        return self.llm.generate_query_with_beliefs(
            feature_distances,
            belief_mean,
            belief_variance
        )

    def _execute_preference_query(
        self,
        observation: dict,
        robot_position: Tuple[int, int],
        board_properties: List[str],
        collected_properties: List[str]
    ) -> Tuple[Dict[str, float], str, str]:
        """Execute a preference query asking which of two objects the human prefers."""
        objects = observation.get('objects', {})

        if len(objects) < 2:
            # Not enough objects to compare
            return {}, "", ""

        # Select two objects to compare
        # Strategy: Pick two objects with high uncertainty (variance in predicted value)
        object_list = list(objects.items())

        if self.belief is not None:
            # Score each object by variance in predicted reward
            object_scores = []
            for obj_id, obj_data in object_list:
                feature_vec = self.belief.get_object_feature_vector(obj_data['properties'])
                # Compute variance of predicted reward: var(w^T x) = x^T Sigma x
                pred_var = feature_vec @ self.belief.covariance @ feature_vec
                object_scores.append((obj_id, obj_data, pred_var))

            # Sort by variance (descending) and pick top 2
            object_scores.sort(key=lambda x: x[2], reverse=True)
            obj1_id, obj1_data, _ = object_scores[0]
            obj2_id, obj2_data, _ = object_scores[1] if len(object_scores) > 1 else object_scores[0]
        else:
            # Random selection if no belief
            obj1_id, obj1_data = object_list[0]
            obj2_id, obj2_data = object_list[1] if len(object_list) > 1 else object_list[0]

        object1_info = {
            'id': obj1_id,
            'position': obj1_data['position'],
            'properties': obj1_data['properties']
        }
        object2_info = {
            'id': obj2_id,
            'position': obj2_data['position'],
            'properties': obj2_data['properties']
        }

        if self.verbose:
            print(f"\n[Query Mode: preference]")
            print(f"  Object A: {object1_info['properties']}")
            print(f"  Object B: {object2_info['properties']}")

        # Generate preference query
        query = self.llm.generate_preference_query(
            object1_info,
            object2_info,
            robot_position
        )

        # Get human response
        response = self.human_responder.respond_to_query(
            query,
            board_properties,
            collected_properties
        )

        # Interpret preference response
        preference = self.llm.interpret_preference_response(
            query,
            response,
            object1_info,
            object2_info
        )

        # Convert preference to weights
        # If human prefers object A, increase weights for A's features, decrease for B's
        property_weights = {f: 0.0 for f in self.feature_names}

        if preference == 1:
            # Human prefers object A
            for prop_val in object1_info['properties'].values():
                if prop_val in property_weights:
                    property_weights[prop_val] = 1.0
            for prop_val in object2_info['properties'].values():
                if prop_val in property_weights:
                    property_weights[prop_val] = -1.0
        elif preference == 2:
            # Human prefers object B
            for prop_val in object2_info['properties'].values():
                if prop_val in property_weights:
                    property_weights[prop_val] = 1.0
            for prop_val in object1_info['properties'].values():
                if prop_val in property_weights:
                    property_weights[prop_val] = -1.0

        return property_weights, query, response

    def _execute_eig_query(
        self,
        observation: dict,
        robot_position: Tuple[int, int],
        board_properties: List[str],
        collected_properties: List[str]
    ) -> Tuple[Dict[str, float], str, str]:
        """
        Execute a query using Expected Information Gain to select optimal pairwise comparison.
        
        Each option will have exactly one feature from each active property category.
        Uses Plackett-Luce update treating the chosen option as if collected at distance 1.
        """
        if self.eig_selector is None:
            features_per_option = len(self.active_categories)
            self.eig_selector = EIGQuerySelector(
                num_features=len(self.feature_names),
                features_per_option=features_per_option,
                num_mc_samples=self.eig_num_mc_samples,
                rng=self.rng
            )

        # Generate candidate queries where each option has one feature per category
        # Group feature indices by category
        category_to_indices = {}
        for i, feature in enumerate(self.feature_names):
            for category in self.active_categories:
                if feature in PROPERTY_VALUES[category]:
                    category_to_indices.setdefault(category, []).append(i)
                    break
        
        # Get all possible combinations of features (one per category)
        category_options = []
        for category in self.active_categories:
            indices = category_to_indices.get(category, [])
            if indices:
                category_options.append(indices)
            else:
                # If no features for this category, use empty
                category_options.append([])
        
        # Generate all possible options (one feature per category)
        all_options = []
        for combo in product(*category_options):
            option_vec = np.zeros(len(self.feature_names))
            for idx in combo:
                option_vec[idx] = 1.0
            all_options.append(option_vec)
        
        # Generate candidate pairs and compute EIG for each
        best_eig = -float('inf')
        best_query = None
        
        num_candidates = min(200, len(all_options) * (len(all_options) - 1) // 2)
        candidates_checked = 0
        
        for i, opt_a in enumerate(all_options):
            for opt_b in all_options[i+1:]:
                eig = self.eig_selector._estimate_eig(self.belief, opt_a, opt_b)
                if eig > best_eig:
                    best_eig = eig
                    best_query = (opt_a, opt_b)
                
                candidates_checked += 1
                if candidates_checked >= num_candidates:
                    break
            if candidates_checked >= num_candidates:
                break
        
        if best_query is None:
            # Fallback: random query
            opt_a = all_options[0] if all_options else np.zeros(len(self.feature_names))
            opt_b = all_options[1] if len(all_options) > 1 else np.zeros(len(self.feature_names))
            best_query = (opt_a, opt_b)
            best_eig = 0.0
        
        option_a, option_b = best_query
        expected_eig = best_eig

        # Convert feature vectors to feature names and properties dict
        features_a = [self.feature_names[i] for i in range(len(self.feature_names)) if option_a[i] > 0.5]
        features_b = [self.feature_names[i] for i in range(len(self.feature_names)) if option_b[i] > 0.5]
        
        # Create properties dicts for Plackett-Luce update
        properties_a = {}
        properties_b = {}
        for feature in features_a:
            for category in self.active_categories:
                if feature in PROPERTY_VALUES[category]:
                    properties_a[category] = feature
                    break
        for feature in features_b:
            for category in self.active_categories:
                if feature in PROPERTY_VALUES[category]:
                    properties_b[category] = feature
                    break

        # Store previous belief state for comparison
        prev_mean = self.belief.mean.copy()
        prev_variances = np.array([self.belief.covariance[i, i] for i in range(self.belief.num_features)])

        # Generate natural language query using LLM
        query = self.llm.generate_eig_query(
            features_a,
            features_b,
            self.active_categories
        )
        
        print("\n" + "=" * 120)
        print("EIG QUERY")
        print("=" * 120)
        print(f"\nQuery: {query}")
        print(f"\nOption A: {properties_a}")
        print(f"  Features: {features_a}")
        print(f"\nOption B: {properties_b}")
        print(f"  Features: {features_b}")
        print(f"\nExpected EIG: {expected_eig:.4f}")

        # Get human response
        response = self.human_responder.respond_to_eig_query(
            query,
            features_a,
            features_b,
            board_properties,
            collected_properties
        )
        
        print(f"\nHuman Response: {response}")

        # Interpret response to determine preference
        preference = self.llm.interpret_eig_response(
            query,
            response,
            features_a,
            features_b
        )
        
        print(f"\nInterpreted Preference: {'Option A' if preference == 1 else 'Option B' if preference == 2 else 'Unclear'}")

        # Update belief using Plackett-Luce model
        # Treat as if human collected the preferred option at distance 1
        # with the other option as the only alternative
        if preference == 1:
            chosen_properties = properties_a
            alternative_properties = properties_b
        elif preference == 2:
            chosen_properties = properties_b
            alternative_properties = properties_a
        else:
            chosen_properties = None
            alternative_properties = None

        if chosen_properties is not None:
            # Use Plackett-Luce update (reduces to Bradley-Terry for 2 options)
            self.belief.update_from_choice_plackett_luce(
                chosen_object=chosen_properties,
                alternative_objects=[chosen_properties, alternative_properties],
                chosen_distance=1.0,
                alternative_distances=[1.0, 1.0],
                learning_rate=self.pl_learning_rate,
                num_gradient_steps=self.pl_gradient_steps
            )
            
            # Print belief update
            curr_mean = self.belief.mean
            curr_variances = np.array([self.belief.covariance[i, i] for i in range(self.belief.num_features)])
            mean_deltas = curr_mean - prev_mean
            variance_deltas = curr_variances - prev_variances
            
            print(f"\n" + "-" * 120)
            print("BELIEF UPDATE (Plackett-Luce)")
            print("-" * 120)
            print(f"{'Feature':<15} {'Prev Mean':>12} {'New Mean':>12} {'Δ Mean':>12} {'Prev Var':>12} {'New Var':>12} {'Δ Var':>12}")
            print("-" * 120)
            
            # Sort by absolute mean delta
            feature_deltas = [(i, feature, abs(mean_deltas[i])) for i, feature in enumerate(self.feature_names)]
            feature_deltas.sort(key=lambda x: x[2], reverse=True)
            
            for i, feature, _ in feature_deltas:
                marker = " *" if self.true_reward_properties and feature in self.true_reward_properties else ""
                print(f"{feature:<15} {prev_mean[i]:>+12.4f} {curr_mean[i]:>+12.4f} {mean_deltas[i]:>+12.4f} "
                      f"{prev_variances[i]:>12.4f} {curr_variances[i]:>12.4f} {variance_deltas[i]:>+12.4f}{marker}")
            
            print("-" * 120)
            print("* indicates true rewarding property")
            print("=" * 120 + "\n")

        # Return empty weights dict since we updated belief directly
        property_weights = {}
        
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

        # Information gain: entropy before update
        entropy_before = self.belief.compute_differential_entropy()

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
        # Information gain: entropy after update
        entropy_after = self.belief.compute_differential_entropy()
        realized_ig = max(0, entropy_before - entropy_after)

        # Record information gain for this observation-based update
        ig_entry = {
            'source': 'observation',
            'entropy_before': entropy_before,
            'entropy_after': entropy_after,
            'information_gain': realized_ig,
            'num_alternatives': len(all_objects_props) - 1,
            'chosen_distance': chosen_distance,
            'chosen_properties': chosen_object['properties'],
        }
        self.information_gain_history.append(ig_entry)
        self.observation_information_gain_history.append(ig_entry)

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
                
                weights, query, response, entropy_before = self._execute_query(observation)
                
                # For non-EIG modes, update belief from weights
                if self.query_mode != "eig" and weights:
                    self.belief.update_from_linear_gaussian(
                        weights,
                        observation_noise_variance=self.lg_noise_variance
                    )
                
                # Compute realized information gain AFTER belief update
                entropy_after = self.belief.compute_differential_entropy()
                realized_ig = max(0, entropy_before - entropy_after)
                
                # Store information gain in history
                self.information_gain_history.append({
                    'query_num': self.queries_used + 1,
                    'query_mode': self.query_mode,
                    'entropy_before': entropy_before,
                    'entropy_after': entropy_after,
                    'information_gain': realized_ig,
                    'query': query,
                    'response': response
                })
                
                # Update the last entry in query_history with the realized information gain
                if self.query_history:
                    self.query_history[-1]['information_gain'] = realized_ig
                
                # Increment queries used counter
                self.queries_used += 1
                
                # Print information gain
                if self.verbose:
                    print(f"[Information Gain] Query {self.queries_used}: IG={realized_ig:.4f} nats")
                    print(f"  Entropy before: {entropy_before:.4f}, after: {entropy_after:.4f}")
                
                # Print query update table
                if self.verbose and weights:
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
        # Separate information gain from queries vs observations
        query_igs = [
            h['information_gain']
            for h in self.information_gain_history
            if h.get('source', 'query') == 'query'
        ]
        obs_igs = [
            h['information_gain']
            for h in self.information_gain_history
            if h.get('source', 'query') == 'observation'
        ]

        avg_query_ig = float(np.mean(query_igs)) if query_igs else 0.0
        total_query_ig = float(np.sum(query_igs)) if query_igs else 0.0
        avg_obs_ig = float(np.mean(obs_igs)) if obs_igs else 0.0
        total_obs_ig = float(np.sum(obs_igs)) if obs_igs else 0.0

        return {
            'queries_used': self.queries_used,
            'query_budget': self.query_budget,
            'query_history': self.query_history,
            'participation_ratio': self.belief.compute_participation_ratio() if self.belief else 0.0,
            'expected_weights': self.belief.get_expected_weights() if self.belief else {},
            'entropy_history': self.entropy_history,
            'decision_history': self.decision_history,
            'information_gain_history': self.information_gain_history,
            'average_query_information_gain': avg_query_ig,
            'total_query_information_gain': total_query_ig,
            'average_observation_information_gain': avg_obs_ig,
            'total_observation_information_gain': total_obs_ig,
        }

    def get_inferred_properties(self) -> List[Tuple[str, float]]:
        """Get inferred property weights sorted by value."""
        if self.belief is None:
            return []
        weights = self.belief.get_expected_weights()
        return sorted(weights.items(), key=lambda x: x[1], reverse=True)