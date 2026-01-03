"""Query-augmented robot agent that uses LLM queries at test time."""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .hierarchical_dqn_robot import HierarchicalDQNRobotAgent
from .human import HumanAgent
from ..objects import PROPERTY_CATEGORIES, PROPERTY_VALUES
from ..llm_interface import LLMInterface, SimulatedHumanResponder


@dataclass
class PreferenceBeliefs:
    """Tracks beliefs about which properties are rewarding."""
    
    # Property value -> estimated preference score
    # Positive = human values this (robot should also collect)
    # Negative = human doesn't value this (robot should avoid)
    values: Dict[str, float] = field(default_factory=dict)
    
    # Property value -> confidence (accumulated evidence)
    confidence: Dict[str, float] = field(default_factory=dict)
    
    # Track if any queries have been made
    has_queries: bool = False
    
    def initialize(self, property_values: List[str]):
        """Initialize with uniform prior."""
        for value in property_values:
            self.values[value] = 0.0
            self.confidence[value] = 0.0
    
    def update_from_observation(self, collected_properties: Dict[str, str]):
        """
        Update beliefs from observing human collect an object.
        Human collecting an object suggests they like those properties.
        """
        for prop_value in collected_properties.values():
            if prop_value in self.values:
                old_conf = self.confidence[prop_value]
                new_conf = old_conf + 1.0
                # Positive value = human likes this
                self.values[prop_value] = (
                    (old_conf * self.values[prop_value] + 1.0) / new_conf
                )
                self.confidence[prop_value] = new_conf
    
    def update_from_query(
        self, 
        property_weights: Dict[str, float],
        query_confidence: float = 2.0
    ):
        """
        Update beliefs from LLM-interpreted query response.
        
        Args:
            property_weights: Property value -> weight from LLM interpretation
                             Positive = human values this (robot should collect too)
                             Negative = human doesn't value this
            query_confidence: How much to weight query info
        """
        self.has_queries = True
        
        for prop_value, weight in property_weights.items():
            if prop_value in self.values:
                old_conf = self.confidence[prop_value]
                new_conf = old_conf + query_confidence
                self.values[prop_value] = (
                    (old_conf * self.values[prop_value] + query_confidence * weight)
                    / new_conf
                )
                self.confidence[prop_value] = new_conf
    
    def get_robot_preference_score(self, properties: Dict[str, str]) -> float:
        """
        Score an object for the ROBOT to collect.
        
        Robot should target objects the human DOES want (cooperative behavior).
        Returns sum of preference scores for object's properties.
        """
        # Sum of human preferences for this object's properties
        # Positive values = human values this, so robot should collect it too
        return sum(self.values.get(v, 0.0) for v in properties.values())
    
    def get_uncertain_properties(self, threshold: float = 1.0) -> List[str]:
        """Get properties with confidence below threshold."""
        return [p for p, c in self.confidence.items() if c < threshold]


class QueryAugmentedRobotAgent:
    """
    Robot agent that augments learned policy with LLM queries at test time.
    
    Wraps a pre-trained HierarchicalDQNRobotAgent and adds:
    - Query mechanism to ask human about property preferences
    - Belief tracking from queries and observations
    - Blending of Q-values with belief-based scores
    
    The robot cooperates with the human to maximize the same reward function.
    """
    
    def __init__(
        self,
        base_agent: HierarchicalDQNRobotAgent,
        llm_interface: Optional[LLMInterface] = None,
        query_budget: int = 5,
        query_threshold: float = 0.3,
        blend_factor: float = 0.5,
        confidence_threshold: float = 3.0,
        verbose: bool = False
    ):
        """
        Args:
            base_agent: Pre-trained hierarchical DQN agent
            llm_interface: Interface to LLM (None = no queries, just blending)
            query_budget: Maximum queries per episode
            query_threshold: Score gap threshold for triggering queries
            blend_factor: Base blend weight for beliefs vs Q-values (0=Q, 1=beliefs)
            confidence_threshold: Confidence level for full belief trust
            verbose: Print debug information
        """
        self.base_agent = base_agent
        self.llm = llm_interface
        self.query_budget = query_budget
        self.query_threshold = query_threshold
        self.blend_factor = blend_factor
        self.confidence_threshold = confidence_threshold
        self.verbose = verbose
        
        # Get property values from base agent
        self.property_values = base_agent.property_values
        self.active_categories = base_agent.active_categories
        
        # Initialize beliefs
        self.beliefs = PreferenceBeliefs()
        self.beliefs.initialize(self.property_values)
        
        # Query tracking
        self.queries_used = 0
        self.query_history: List[Dict] = []
        self.observed_object_ids: Set[int] = set()
        
        # Human responder (set during episode)
        self.human_responder: Optional[SimulatedHumanResponder] = None
        
        # Episode state
        self.has_started = False
    
    def reset(self):
        """Reset for new episode."""
        self.base_agent.reset()
        
        self.beliefs = PreferenceBeliefs()
        self.beliefs.initialize(self.property_values)
        
        self.queries_used = 0
        self.query_history = []
        self.observed_object_ids = set()
        self.human_responder = None
        self.has_started = False
    
    def set_human_responder(self, reward_properties: set):
        """Set up simulated human responder with true preferences."""
        if self.llm is None:
            raise ValueError("LLM interface required for human responder")
        self.human_responder = SimulatedHumanResponder(
            reward_properties,
            llm_interface=self.llm,
            verbose=self.verbose
        )
    
    def _update_beliefs_from_observations(self, observation: dict):
        """Update preference beliefs from human's collected objects."""
        human_collected = observation.get('human_collected', [])
        
        for collected in human_collected:
            obj_id = collected['id']
            if obj_id not in self.observed_object_ids:
                self.observed_object_ids.add(obj_id)
                self.beliefs.update_from_observation(collected['properties'])
                
                if self.verbose:
                    print(f"[Beliefs] Updated from human collecting object {obj_id}")
                    print(f"  Properties: {collected['properties']}")
    
    def _should_query(self, observation: dict) -> bool:
        """
        Decide whether to query based on overall uncertainty.
        
        Returns:
            True if should query, False otherwise
        """
        if self.llm is None:
            return False
        
        if self.queries_used >= self.query_budget:
            return False
        
        objects = observation.get('objects', {})
        if len(objects) == 0:
            return False
        
        # Calculate average confidence across all properties
        avg_confidence = np.mean(list(self.beliefs.confidence.values()))
        
        # Query if we're generally uncertain
        should_query = avg_confidence < self.confidence_threshold
        
        if self.verbose and should_query:
            print(f"[Query] Triggering query: avg_confidence={avg_confidence:.3f}")
        
        return should_query
    
    def _execute_query(self, observation: dict) -> Dict[str, float]:
        """
        Execute a query about property preferences.
        
        Returns:
            Property weights from LLM interpretation
        """
        if self.llm is None or self.human_responder is None:
            return {}
        
        # Gather board properties (from objects currently on board)
        objects = observation.get('objects', {})
        board_properties = []
        for obj_data in objects.values():
            board_properties.extend(obj_data['properties'].values())
        # Remove duplicates while preserving order
        board_properties = list(dict.fromkeys(board_properties))
        
        # Gather collected properties (from objects human has collected)
        human_collected = observation.get('human_collected', [])
        collected_properties = []
        for collected in human_collected:
            collected_properties.extend(collected['properties'].values())
        # Remove duplicates while preserving order
        collected_properties = list(dict.fromkeys(collected_properties))
        
        # Generate query
        query = self.llm.generate_query(
            board_properties, 
            collected_properties, 
            self.active_categories
        )
        
        if self.verbose:
            print(f"[Query] Generated question: {query}")
        
        # Get simulated human response
        response = self.human_responder.respond_to_query(
            query, 
            board_properties, 
            collected_properties
        )
        
        if self.verbose:
            print(f"[Query] Human response: {response}")
        
        # Interpret response
        property_weights = self.llm.interpret_response(
            query, response, 
            board_properties, collected_properties,
            self.property_values, self.active_categories
        )
        
        if self.verbose:
            print(f"[Query] Extracted weights: {property_weights}")
        
        # Store in history
        self.query_history.append({
            'query': query,
            'response': response,
            'weights': property_weights,
            'board_properties': board_properties,
            'collected_properties': collected_properties
        })
        self.queries_used += 1
        
        return property_weights
    
    def _compute_blended_action(self, observation: dict) -> int:
        """
        Compute action blending Q-values with preference beliefs.
        
        Returns:
            Action index (property action)
        """
        valid_actions = self.base_agent._get_valid_property_actions(observation)
        
        if not valid_actions:
            return 0
        
        # Get Q-values from base agent
        high_level_input = self.base_agent._encode_high_level_input(observation)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(high_level_input).unsqueeze(0)
            state_tensor = state_tensor.to(self.base_agent.device)
            q_values = self.base_agent.high_level_q_network(state_tensor)
            q_values = q_values.cpu().numpy()[0]
        
        # Get belief scores for each property action
        # Robot should target properties human likes (positive values in beliefs)
        belief_scores = np.array([
            self.beliefs.values.get(self.base_agent.property_values[a], 0.0)
            for a in range(len(self.base_agent.property_values))
        ])
        
        # Compute adaptive blend factor based on confidence
        confidences = np.array([
            self.beliefs.confidence.get(self.base_agent.property_values[a], 0.0)
            for a in range(len(self.base_agent.property_values))
        ])
        
        # Only blend if we have some confidence (from queries or observations)
        avg_confidence = confidences[valid_actions].mean() if len(valid_actions) > 0 else 0
        
        if avg_confidence > 0:
            # Adaptive: trust beliefs more when confident
            adaptive_blend = min(
                self.blend_factor * (avg_confidence / self.confidence_threshold),
                0.9
            )
        else:
            # No confidence yet - just use Q-values
            adaptive_blend = 0.0
        
        # Convert to probabilities for valid actions
        valid_q = q_values[valid_actions]
        valid_beliefs = belief_scores[valid_actions]
        
        # Softmax for Q-values
        q_max = valid_q.max()
        q_probs = np.exp(valid_q - q_max)
        q_probs = q_probs / (q_probs.sum() + 1e-8)
        
        # Softmax for belief scores
        if valid_beliefs.std() > 1e-8:
            b_max = valid_beliefs.max()
            belief_probs = np.exp(valid_beliefs - b_max)
            belief_probs = belief_probs / (belief_probs.sum() + 1e-8)
        else:
            # Uniform if no differentiation
            belief_probs = np.ones_like(valid_beliefs) / len(valid_beliefs)
        
        # Blend
        blended = (1 - adaptive_blend) * q_probs + adaptive_blend * belief_probs
        
        if self.verbose:
            print(f"[Blend] adaptive_blend={adaptive_blend:.3f}")
            print(f"  Q-probs: {dict(zip(valid_actions, q_probs))}")
            print(f"  Belief-probs: {dict(zip(valid_actions, belief_probs))}")
            print(f"  Blended: {dict(zip(valid_actions, blended))}")
        
        # Select best valid action
        best_valid_idx = np.argmax(blended)
        return valid_actions[best_valid_idx]
    
    def _should_select_new_goal(self, observation: dict) -> bool:
        """Check if we need to select a new goal."""
        if self.base_agent.current_target_property is None:
            return True
        
        if self.base_agent.current_goal_object_id is not None:
            objects = observation.get('objects', {})
            if self.base_agent.current_goal_object_id not in objects:
                return True
        
        return False
    
    def _set_target(self, observation: dict, target_action: int):
        """Set the target property and find nearest object."""
        target_property = self.base_agent.property_values[target_action]
        
        self.base_agent.current_target_property = target_property
        self.base_agent.current_target_action = target_action
        
        result = self.base_agent._find_nearest_object_with_property(
            observation, target_property
        )
        
        if result:
            self.base_agent.current_goal_object_id = result[0]
            self.base_agent.current_goal_position = result[1]
        else:
            self.base_agent.current_goal_object_id = None
            self.base_agent.current_goal_position = None
        
        self.base_agent.current_path = []
        
        if self.verbose:
            print(f"[Target] Set target property: {target_property}")
            print(f"  Goal object: {self.base_agent.current_goal_object_id}")
    
    def get_action(self, observation: dict, training: bool = False) -> int:
        """
        Get action, potentially querying first at high-level decision points.
        
        Args:
            observation: Current observation
            training: If True, don't query (training mode)
            
        Returns:
            Action (movement direction)
        """
        robot_can_collect = observation.get('robot_can_collect', False)
        if not robot_can_collect:
            return 4  # stay
        
        if not self.has_started:
            self.has_started = True
            self.base_agent.has_started = True
        
        # Always update beliefs from observations
        self._update_beliefs_from_observations(observation)
        
        # High-level decision point?
        if self._should_select_new_goal(observation):
            
            # === QUERY PHASE (only at high-level decisions) ===
            if not training and self.llm is not None and self.human_responder is not None:
                should_query = self._should_query(observation)
                if should_query:
                    property_weights = self._execute_query(observation)
                    self.beliefs.update_from_query(property_weights)
            
            # === BLENDED DECISION ===
            target_action = self._compute_blended_action(observation)
            self._set_target(observation, target_action)
        
        # === LOW-LEVEL NAVIGATION ===
        return self.base_agent._get_navigation_action(observation)
    
    def get_query_stats(self) -> Dict:
        """Get statistics about queries used."""
        return {
            'queries_used': self.queries_used,
            'query_budget': self.query_budget,
            'query_history': self.query_history,
            'belief_values': dict(self.beliefs.values),
            'belief_confidence': dict(self.beliefs.confidence)
        }
