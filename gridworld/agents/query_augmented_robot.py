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
    
    # Observational beliefs (what human has collected)
    # Property value -> count of times observed
    observed_counts: Dict[str, int] = field(default_factory=dict)
    
    # Query-derived beliefs (ONLY from LLM)
    # Property value -> LLM-derived weight
    # Positive = human values this (robot should also collect)
    # Negative = human doesn't value this (robot should avoid)
    query_weights: Dict[str, float] = field(default_factory=dict)
    
    # Property value -> confidence from queries
    query_confidence: Dict[str, float] = field(default_factory=dict)
    
    # Track if any queries have been made
    has_queries: bool = False
    
    def initialize(self, property_values: List[str]):
        """Initialize with uniform prior."""
        for value in property_values:
            self.observed_counts[value] = 0
            self.query_weights[value] = 0.0
            self.query_confidence[value] = 0.0
    
    def update_from_observation(self, collected_properties: Dict[str, str]):
        """
        Update observational beliefs from seeing human collect an object.
        This tracks what the human collected but doesn't affect blending.
        """
        for prop_value in collected_properties.values():
            if prop_value in self.observed_counts:
                self.observed_counts[prop_value] += 1
    
    def update_from_query(
        self,
        property_weights: Dict[str, float],
        query_confidence: float = 2.0
    ):
        """
        Update query-derived beliefs from LLM-interpreted query response.
        ONLY these beliefs are used for blending with Q-values.

        Args:
            property_weights: Property value -> weight from LLM interpretation
                             Positive = human values this (robot should collect too)
                             Negative = human doesn't value this
            query_confidence: How much to weight query info
        """
        self.has_queries = True

        for prop_value, weight in property_weights.items():
            # Only update confidence for properties where we learned something
            # (non-zero weight means the LLM provided actual information)
            if prop_value in self.query_weights and weight != 0.0:
                old_conf = self.query_confidence[prop_value]
                new_conf = old_conf + query_confidence
                self.query_weights[prop_value] = (
                    (old_conf * self.query_weights[prop_value] + query_confidence * weight)
                    / new_conf
                )
                self.query_confidence[prop_value] = new_conf
    
    def get_robot_preference_score(self, properties: Dict[str, str]) -> float:
        """
        Score an object for the ROBOT to collect based on query-derived beliefs.
        
        Robot should target objects the human DOES want (cooperative behavior).
        Returns sum of preference scores for object's properties.
        """
        # Sum of query-derived preferences for this object's properties
        # Positive values = human values this, so robot should collect it too
        return sum(self.query_weights.get(v, 0.0) for v in properties.values())
    
    def get_uncertain_properties(self, threshold: float = 1.0) -> List[str]:
        """Get properties with low confidence from queries."""
        return [p for p, c in self.query_confidence.items() if c < threshold]


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
        blend_factor: float = 0.5,
        query_threshold: float = 0.8,
        verbose: bool = False
    ):
        """
        Args:
            base_agent: Pre-trained hierarchical DQN agent
            llm_interface: Interface to LLM (None = no queries, just blending)
            query_budget: Maximum queries per episode
            blend_factor: Blend weight for beliefs vs Q-values (0=Q, 1=beliefs)
            query_threshold: Threshold for number of competitive options to trigger query.
                            Query is triggered when num_competitive > query_threshold.
                            E.g., 3 = query when more than 3 competitive options exist.
            verbose: Print debug information
        """
        self.base_agent = base_agent
        self.llm = llm_interface
        self.query_budget = query_budget
        self.blend_factor = blend_factor
        self.query_threshold = query_threshold
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
        self.entropy_history = []
    
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
    
    def _should_query(self, observation: dict) -> bool:
        """
        Decide whether to query based on number of competitive Q-value options.

        Many competitive options = Q-values are similar = robot is uncertain = should query.
        Few competitive options = one action dominates = robot is confident = don't query.
        
        An option is "competitive" if its Q-value is within 20% of the range from the max.
        Query is triggered when num_competitive > self.query_threshold.
        
        After queries have been made, uses blended Q-values (incorporating belief scores)
        to determine competitiveness, so new information is taken into account when
        deciding whether to query again.

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

        valid_actions = self.base_agent._get_valid_property_actions(observation)
        if len(valid_actions) < 2:
            # Need at least 2 actions to have meaningful entropy
            return False

        # Get Q-values from base agent
        high_level_input = self.base_agent._encode_high_level_input(observation)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(high_level_input).unsqueeze(0)
            state_tensor = state_tensor.to(self.base_agent.device)
            q_values = self.base_agent.high_level_q_network(state_tensor)
            q_values = q_values.cpu().numpy()[0]

        # Get Q-values for valid actions
        valid_q = q_values[valid_actions]

        # If we have query-derived beliefs, blend Q-values with beliefs before checking competitiveness
        has_beliefs = self.beliefs.has_queries
        if has_beliefs:
            # Get belief scores for each property action
            belief_scores = np.array([
                self.beliefs.query_weights.get(self.base_agent.property_values[a], 0.0)
                for a in range(len(self.base_agent.property_values))
            ])
            valid_beliefs = belief_scores[valid_actions]
            
            # Convert Q-values to probabilities via softmax
            q_max = valid_q.max()
            q_probs = np.exp(valid_q - q_max)
            q_probs = q_probs / (q_probs.sum() + 1e-8)
            
            # Check if we have meaningful belief signal
            if np.abs(valid_beliefs).sum() > 1e-8 and valid_beliefs.std() > 1e-8:
                # Convert belief scores to probabilities via softmax
                b_max = valid_beliefs.max()
                belief_probs = np.exp(valid_beliefs - b_max)
                belief_probs = belief_probs / (belief_probs.sum() + 1e-8)
                
                # Blend Q-probs with belief-probs
                blended_probs = (1 - self.blend_factor) * q_probs + self.blend_factor * belief_probs
                
                # Convert blended probabilities back to Q-value-like scores (log scale)
                # Use log transform to get back to value space
                valid_q = np.log(blended_probs + 1e-10)

        # Count competitive options based on Q-value spread
        max_q = valid_q.max()
        min_q = valid_q.min()
        q_range = max_q - min_q
        
        # A Q-value is "competitive" if it's within 20% of the range from max
        threshold = max_q - 0.2 * q_range
        num_competitive = sum(1 for q in valid_q if q > threshold)
        
        # Query if there are more than query_threshold competitive options (high uncertainty)
        should_query = num_competitive > self.query_threshold

        # Store debug info for later analysis
        if not hasattr(self, 'entropy_history'):
            self.entropy_history = []
        
        self.entropy_history.append({
            'num_competitive': num_competitive,
            'max_q': float(max_q),
            'min_q': float(min_q),
            'q_range': float(q_range),
            'competitive_threshold': float(threshold),
            'num_valid_actions': len(valid_actions),
            'should_query': should_query,
            'q_values': valid_q.tolist(),
            'valid_actions': valid_actions,
            'used_blended_values': has_beliefs
        })

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
        
        # Get simulated human response
        response = self.human_responder.respond_to_query(
            query, 
            board_properties, 
            collected_properties
        )
        
        # Interpret response
        property_weights = self.llm.interpret_response(
            query, response, 
            board_properties, collected_properties,
            self.property_values, self.active_categories
        )
        
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
        # ONLY use query-derived beliefs, not observational data
        belief_scores = np.array([
            self.beliefs.query_weights.get(self.base_agent.property_values[a], 0.0)
            for a in range(len(self.base_agent.property_values))
        ])

        # Convert to probabilities for valid actions
        valid_q = q_values[valid_actions]
        valid_beliefs = belief_scores[valid_actions]

        # Softmax for Q-values
        q_max = valid_q.max()
        q_probs = np.exp(valid_q - q_max)
        q_probs = q_probs / (q_probs.sum() + 1e-8)

        # Check if we have any query-derived belief signal
        has_beliefs = self.beliefs.has_queries and np.abs(valid_beliefs).sum() > 1e-8

        if has_beliefs and valid_beliefs.std() > 1e-8:
            # Softmax for belief scores
            b_max = valid_beliefs.max()
            belief_probs = np.exp(valid_beliefs - b_max)
            belief_probs = belief_probs / (belief_probs.sum() + 1e-8)

            # Blend Q-probs with belief-probs
            blended = (1 - self.blend_factor) * q_probs + self.blend_factor * belief_probs
        else:
            # No query-derived belief signal - just use Q-values
            blended = q_probs

        # Select best valid action
        best_valid_idx = np.argmax(blended)
        
        # Store blend info for verbose output
        if self.verbose:
            self._last_blend_info = {
                'valid_actions': valid_actions,
                'q_values': valid_q,
                'q_probs': q_probs,
                'belief_scores': valid_beliefs,
                'has_beliefs': has_beliefs,
                'blended': blended,
                'selected_action': valid_actions[best_valid_idx]
            }
            if has_beliefs and valid_beliefs.std() > 1e-8:
                self._last_blend_info['belief_probs'] = belief_probs
        
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
    
    def _print_decision_summary(self, observation: dict, query_triggered: bool, query_info: dict):
        """Print a concise summary of the high-level decision."""
        print("\n" + "="*80)
        print("HIGH-LEVEL DECISION")
        print("="*80)
        
        # Print rewarding properties
        if self.human_responder is not None:
            rewarding_props = sorted(list(self.human_responder.reward_properties))
            print(f"True Rewarding Properties: {rewarding_props}")
        
        # Print observation stats
        total_observed = sum(self.beliefs.observed_counts.values())
        print(f"Human objects collected: {total_observed}")
        print(f"Queries used: {self.queries_used}/{self.query_budget}")
        
        # Get competitive options info
        if hasattr(self, 'entropy_history') and self.entropy_history:
            last_info = self.entropy_history[-1]
            num_competitive = last_info['num_competitive']
            print(f"Competitive options: {num_competitive} (query if > {self.query_threshold})")
        
        # Print Q-values table
        if hasattr(self, '_last_blend_info'):
            info = self._last_blend_info
            valid_actions = info['valid_actions']
            q_values = info['q_values']
            
            # Get rewarding properties set for marking
            rewarding_set = set()
            if self.human_responder is not None:
                rewarding_set = self.human_responder.reward_properties
            
            print(f"\n{'Action':<15} {'Property':<15} {'Q-value':<12} {'Observed':<12} {'R?'}")
            print("-" * 65)
            for i, action_idx in enumerate(valid_actions):
                prop = self.base_agent.property_values[action_idx]
                q_val = q_values[i]
                obs_count = self.beliefs.observed_counts.get(prop, 0)
                is_rewarding = "✓" if prop in rewarding_set else ""
                print(f"{action_idx:<15} {prop:<15} {q_val:.4f}       {obs_count:<12} {is_rewarding}")
        
        # If query was triggered
        if query_triggered:
            print("\n" + "-"*80)
            print("QUERY TRIGGERED!")
            print("-"*80)
            if query_info:
                print(f"True rewarding properties: {query_info['rewarding_properties']}")
                print(f"Human response: {query_info['response']}")
                print(f"\nExtracted weights (top 5):")
                weights = query_info['weights']
                sorted_weights = sorted(weights.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
                for prop, weight in sorted_weights:
                    if weight != 0:
                        print(f"  {prop}: {weight:+.3f}")
        else:
            print(f"\nQuery NOT triggered (competitive options {num_competitive} <= {self.query_threshold})")
        
        # Print blended decision
        if hasattr(self, '_last_blend_info'):
            info = self._last_blend_info
            print("\n" + "-"*80)
            print("BLENDED DECISION")
            print("-"*80)
            
            # Get rewarding properties set for marking
            rewarding_set = set()
            if self.human_responder is not None:
                rewarding_set = self.human_responder.reward_properties
            
            # Show if blending is active
            has_queries = self.beliefs.has_queries
            blend_status = f"Blending ACTIVE (query weights used)" if has_queries else "Blending INACTIVE (no queries yet, using Q-values only)"
            print(f"{blend_status}")
            
            print(f"\n{'Action':<15} {'Property':<15} {'Q-prob':<12} {'LLM Weight':<12} {'Blended':<12} {'R?'}")
            print("-" * 80)
            
            valid_actions = info['valid_actions']
            q_probs = info['q_probs']
            belief_scores = info['belief_scores']
            blended = info['blended']
            has_beliefs = info['has_beliefs']
            
            for i, action_idx in enumerate(valid_actions):
                prop = self.base_agent.property_values[action_idx]
                q_prob = q_probs[i]
                belief = belief_scores[i]
                blend = blended[i]
                is_rewarding = "✓" if prop in rewarding_set else ""
                marker = " <--" if action_idx == info['selected_action'] else ""
                print(f"{action_idx:<15} {prop:<15} {q_prob:.4f}      {belief:+.4f}       {blend:.4f}      {is_rewarding}{marker}")
            
            selected_prop = self.base_agent.property_values[info['selected_action']]
            is_selected_rewarding = "✓ (CORRECT)" if selected_prop in rewarding_set else "✗ (WRONG)"
            print(f"\nSelected: Action {info['selected_action']} (property: {selected_prop}) {is_selected_rewarding}")
            print("="*80 + "\n")
    
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
            # Initialize base agent's inference on first step
            self.base_agent._update_inference(observation)
        
        # Update base agent's inference (affects Q-network state encoding)
        self.base_agent._update_inference(observation)
        
        # Always update beliefs from observations
        self._update_beliefs_from_observations(observation)
        
        # High-level decision point?
        if self._should_select_new_goal(observation):
            
            # === QUERY PHASE (only at high-level decisions) ===
            query_triggered = False
            query_info = {}
            if not training and self.llm is not None and self.human_responder is not None:
                should_query = self._should_query(observation)
                if should_query:
                    query_triggered = True
                    property_weights = self._execute_query(observation)
                    self.beliefs.update_from_query(property_weights)
                    
                    # Store query info for verbose output
                    if self.verbose and self.query_history:
                        last_query = self.query_history[-1]
                        query_info = {
                            'rewarding_properties': self.human_responder.reward_properties,
                            'response': last_query['response'],
                            'weights': last_query['weights']
                        }
            
            # === BLENDED DECISION ===
            target_action = self._compute_blended_action(observation)
            self._set_target(observation, target_action)
            
            # === VERBOSE OUTPUT ===
            if self.verbose:
                self._print_decision_summary(observation, query_triggered, query_info)
        
        # === LOW-LEVEL NAVIGATION ===
        return self.base_agent._get_navigation_action(observation)
    
    def get_query_stats(self) -> Dict:
        """Get statistics about queries used."""
        return {
            'queries_used': self.queries_used,
            'query_budget': self.query_budget,
            'query_history': self.query_history,
            'query_weights': dict(self.beliefs.query_weights),
            'query_confidence': dict(self.beliefs.query_confidence),
            'observed_counts': dict(self.beliefs.observed_counts),
            'entropy_history': getattr(self, 'entropy_history', [])
        }
