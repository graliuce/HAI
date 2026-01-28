"""LLM interface for query generation and interpretation."""

import json
import os
from typing import Dict, List, Optional, Tuple
import numpy as np


class LLMInterface:
    """Interface for LLM query generation and interpretation."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-5-chat-latest"):
        """
        Initialize LLM interface.
        
        Args:
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            model: Model to use for queries
        """
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY')
        self.model = model
        
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Provide via api_key parameter "
                "or OPENAI_API_KEY environment variable."
            )
    
    def _call_llm(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
        """Call OpenAI API."""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)
            
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature
            )
            
            response_text = response.choices[0].message.content
            return response_text
            
        except ImportError:
            # Fallback to requests
            import requests
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            data = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=data
            )
            
            if response.status_code != 200:
                raise Exception(f"OpenAI API error: {response.status_code}, {response.text}")
            
            response_text = response.json()['choices'][0]['message']['content']
            return response_text
    
    def generate_query(
        self, 
        board_properties: List[str],
        thompson_votes: Dict[str, float],
        active_categories: List[str]
    ) -> str:
        """
        Generate a natural language query about property preferences.
        
        Args:
            board_properties: List of property values currently on the board
            thompson_votes: Dict mapping feature names to normalized vote frequencies from Thompson sampling
            active_categories: List of active property categories
            
        Returns:
            Natural language question string
        """
        # Format the properties for the prompt
        board_props_str = ", ".join(board_properties) if board_properties else "none yet"
        
        # Format Thompson sampling votes as a table
        if thompson_votes:
            # Sort by normalized value (descending) for better readability
            sorted_votes = sorted(thompson_votes.items(), key=lambda x: x[1], reverse=True)
            
            # Create table header
            beliefs_str = "Robot's Current Beliefs (from Thompson Sampling):\n"
            beliefs_str += f"{'Feature':<15} | {'Normalized Value':<20}\n"
            beliefs_str += "-" * 38 + "\n"
            
            # Add rows for each feature
            for feature, norm_value in sorted_votes:
                beliefs_str += f"{feature:<15} | {norm_value:<20.3f}\n"
        else:
            beliefs_str = "Robot's Current Beliefs: (no beliefs yet)"
        
        prompt = f"""You are a robot working with a human to collect objects in a shared environment. You both want to maximize the same reward. You need to understand what properties give reward so you can help collect valuable objects.

Current situation:
- Properties on the board: {board_props_str}

{beliefs_str}

The normalized value represents how often each feature appeared in objects selected during Thompson sampling (higher = more often selected).

Generate an open-ended, informative question (1 sentence) to ask the human that will help you decide which features are valuable to collect. 

Focus on asking about properties that are:
1. Relevant (currently on the board)
2. Important (have a high normalized value)
2. Uncertain according to your current beliefs

Output only the question, nothing else."""

        messages = [
            {"role": "system", "content": "You are a robot assistant that generates clear, strategic questions to reduce uncertainty."},
            {"role": "user", "content": prompt}
        ]
        
        response = self._call_llm(messages, temperature=0.7)
        return response.strip()
    
    def interpret_response(
        self,
        query: str,
        response: str,
        board_properties: List[str],
        collected_properties: List[str],
        property_values: List[str],
        active_categories: List[str]
    ) -> Dict[str, float]:
        """
        Interpret human response to extract property preferences.
        
        Args:
            query: The question that was asked
            response: The human's response
            board_properties: List of property values currently on the board
            collected_properties: List of property values from objects human has collected
            property_values: List of all possible property values
            active_categories: List of active property categories
            
        Returns:
            Dict mapping property values to preference weights (0 to 1, where higher = more preferred)
        """
        # Build property value list for the prompt
        prop_list = ", ".join(property_values)
        board_props_str = ", ".join(board_properties) if board_properties else "none"
        collected_props_str = ", ".join(collected_properties) if collected_properties else "none"
        
        prompt = f"""Analyze this conversation and extract property preferences.

Question asked: "{query}"
Human's response: "{response}"

Context:
- Properties on the board: {board_props_str}
- Properties the human has collected: {collected_props_str}
- All possible property values: {prop_list}

The robot and human are working together to maximize the same reward. Based on the human's response, determine what properties give reward and should be collected.

Output a JSON object mapping property values to preference weights (-1.0 to 1.0):
- Use positive weights for properties the human indicates are VALUABLE/REWARDING (these should be collected)
- Use negative weights for properties the human indicates are NOT valuable
- Use 0 for properties that weren't clearly indicated

Output only valid JSON in this format: {{"property_value1": weight1, "property_value2": weight2, ...}}"""

        messages = [
            {"role": "system", "content": "You are an assistant that extracts preferences from natural language. Output only valid JSON."},
            {"role": "user", "content": prompt}
        ]
        
        llm_output = self._call_llm(messages, temperature=0.3)
        
        # Clean up response
        llm_output = llm_output.strip()
        if llm_output.startswith("```"):
            llm_output = llm_output.split("```")[1]
            if llm_output.startswith("json"):
                llm_output = llm_output[4:]
        llm_output = llm_output.strip()
        
        weights = json.loads(llm_output)
        
        # Ensure all property values have a weight
        for prop in property_values:
            if prop not in weights:
                weights[prop] = 0.0
        
        return weights


class SimulatedHumanResponder:
    """Simulates human responses using LLM based on true reward properties."""

    def __init__(
        self,
        reward_properties: set,
        llm_interface: LLMInterface,
        verbose: bool = False,
        property_value_rewards: Optional[Dict[str, float]] = None
    ):
        """
        Args:
            reward_properties: Set of property values that give reward
            llm_interface: LLM interface for generating responses
            verbose: Whether to print debug info
            property_value_rewards: For additive valuation mode, dict mapping
                property values to their reward amounts
        """
        self.reward_properties = reward_properties
        self.llm = llm_interface
        self.verbose = verbose
        self.property_value_rewards = property_value_rewards  # For additive mode
    
    def respond_to_query(
        self,
        query: str,
        board_properties: List[str],
        collected_properties: List[str]
    ) -> str:
        """
        Generate a response using LLM based on true preferences.

        The human indicates which properties are valuable (give reward).

        Args:
            query: The question asked
            board_properties: Properties currently on the board
            collected_properties: Properties from objects human has collected

        Returns:
            Natural language response
        """
        # Format context for LLM
        board_props_str = ", ".join(board_properties) if board_properties else "none"
        collected_props_str = ", ".join(collected_properties) if collected_properties else "none"

        # Determine rewarding properties string (different format for each mode)
        if self.property_value_rewards:
            # Additive mode: list all properties with their actual reward values
            sorted_rewards = sorted(
                self.property_value_rewards.items(),
                key=lambda x: x[1],
                reverse=True
            )
            if self.verbose:
                print(f"[SimulatedHuman] Property value rewards:")
                for prop, reward in sorted_rewards:
                    print(f"    {prop}: {reward:+.2f}")
            
            # Format all reward values for the prompt
            rewards_list = [f"{prop}: {reward:+.2f}" for prop, reward in sorted_rewards]
            rewarding_props_str = "\n  ".join(rewards_list)
            
            prompt = f"""You are simulating a human in a collaborative environment where you and a robot are collecting objects together to maximize reward.

Your true reward values for each property:
  {rewarding_props_str}

Question from the robot: "{query}"

Generate a natural, helpful, and accurate response (1-3 sentences) that answers the question based on your TRUE reward values above. Be specific and accurate about which properties have positive vs negative rewards.
Do not provide information that is not directly asked for in the question.

Output only your response, nothing else."""
        else:
            # Standard mode: list rewarding properties
            rewarding_props_str = ", ".join(self.reward_properties)
            if self.verbose:
                print(f"[SimulatedHuman] Rewarding properties: {self.reward_properties}")
            
            prompt = f"""You are simulating a human in a collaborative environment where you and a robot are collecting objects together to maximize reward.

Your true preferences (properties that give reward): {rewarding_props_str}

Question from the robot: "{query}"

Generate a natural, helpful, and accurate response (1-3 sentences) that answers the question. Do not provide information that is not directly asked for in the question.
Do not mention any preferences that are not in your list of true preferences. Don't mention numbers explicitly but instead describe the values in natural language.

Output only your response, nothing else."""

        messages = [
            {"role": "system", "content": "You are simulating a helpful human collaborator who clearly communicates their preferences."},
            {"role": "user", "content": prompt}
        ]

        response = self.llm._call_llm(messages, temperature=0.7)
        return response.strip()
