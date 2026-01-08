"""Human agent with fixed greedy policy."""

from typing import Dict, Optional, Set, Tuple
import heapq


class HumanAgent:
    """
    A simulated human agent that knows the reward specification.

    Policy:
    - Standard mode: Greedy - finds the nearest rewarding object by L2 distance
      and takes the shortest path to collect it.
    - Additive valuation mode: Targets object with highest reward/distance ratio.
    """

    def __init__(self):
        """Initialize the human agent."""
        self.reward_properties: Set[str] = set()
        self.current_target: Optional[Tuple[int, int]] = None
        self.additive_valuation: bool = False
        self.object_rewards: Dict[int, float] = {}

    def reset(self, reward_properties: Set[str], additive_valuation: bool = False,
              object_rewards: Optional[Dict[int, float]] = None):
        """
        Reset the agent for a new episode.

        Args:
            reward_properties: Set of property values that give reward
            additive_valuation: Whether using additive valuation mode
            object_rewards: Dict mapping object_id -> total reward (for additive mode)
        """
        self.reward_properties = reward_properties
        self.current_target = None
        self.additive_valuation = additive_valuation
        self.object_rewards = object_rewards or {}

    def get_action(self, observation: dict) -> int:
        """
        Get the action for the human agent.

        Policy:
        - Standard mode: Find the nearest rewarding object by L2 distance
        - Additive mode: Find the object with highest reward/distance ratio

        Args:
            observation: Full observation including reward_properties

        Returns:
            Action (0=up, 1=down, 2=left, 3=right, 4=stay)
        """
        position = observation['human_position']
        objects = observation['objects']

        # Update object_rewards from observation if in additive mode
        if observation.get('additive_valuation', False):
            self.additive_valuation = True
            self.object_rewards = observation.get('object_rewards', {})

        if self.additive_valuation:
            return self._get_action_additive(position, objects, observation)
        else:
            return self._get_action_standard(position, objects)

    def _get_action_standard(self, position: Tuple[int, int], objects: dict) -> int:
        """Standard mode: target nearest rewarding object."""
        # Find all rewarding objects
        rewarding_objects = []
        for obj_id, obj_data in objects.items():
            props = obj_data['properties']
            # Check if any property value matches reward properties
            for prop_value in props.values():
                if prop_value in self.reward_properties:
                    rewarding_objects.append((obj_id, obj_data['position']))
                    break

        if not rewarding_objects:
            # No rewarding objects left, stay in place
            return 4

        # Find nearest rewarding object by L2 distance
        nearest_pos = None
        nearest_dist = float('inf')

        for obj_id, obj_pos in rewarding_objects:
            dist = self._l2_distance(position, obj_pos)
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_pos = obj_pos

        if nearest_pos is None:
            return 4

        self.current_target = nearest_pos

        # Get action toward nearest object (avoiding non-rewarding objects)
        return self._get_action_toward(position, nearest_pos, objects)

    def _get_action_additive(self, position: Tuple[int, int], objects: dict,
                              observation: dict) -> int:
        """Additive mode: target object with highest reward/distance ratio."""
        best_target = None
        best_ratio = float('-inf')
        object_rewards = observation.get('object_rewards', {})

        for obj_id, obj_data in objects.items():
            obj_pos = obj_data['position']
            reward = object_rewards.get(obj_id, 0.0)

            # Calculate distance (minimum 1 to avoid division by zero)
            dist = max(1.0, self._l2_distance(position, obj_pos))

            # Calculate reward/distance ratio
            ratio = reward / dist

            if ratio > best_ratio:
                best_ratio = ratio
                best_target = obj_pos

        if best_target is None:
            return 4

        self.current_target = best_target

        # Get action toward target (avoiding negative-reward objects)
        return self._get_action_toward_additive(position, best_target, objects, object_rewards)

    def _get_action_toward_additive(
        self,
        current: Tuple[int, int],
        target: Tuple[int, int],
        objects: dict,
        object_rewards: dict
    ) -> int:
        """
        Get the action to move toward target, avoiding negative-reward objects.
        """
        cx, cy = current
        tx, ty = target

        if current == target:
            return 4  # stay

        # Create a map of negative-reward object positions (to avoid)
        avoid_positions = set()
        for obj_id, obj_data in objects.items():
            reward = object_rewards.get(obj_id, 0.0)
            if reward < 0:
                avoid_positions.add(obj_data['position'])

        dx = tx - cx
        dy = ty - cy

        # Try actions in order of preference based on distance to target
        actions = []
        if abs(dx) >= abs(dy):
            # Prioritize horizontal movement
            if dx > 0:
                actions = [3, 1 if dy > 0 else 0, 2]  # right, then vertical, then left
            else:
                actions = [2, 1 if dy > 0 else 0, 3]  # left, then vertical, then right
        else:
            # Prioritize vertical movement
            if dy > 0:
                actions = [1, 3 if dx > 0 else 2, 0]  # down, then horizontal, then up
            else:
                actions = [0, 3 if dx > 0 else 2, 1]  # up, then horizontal, then down

        # Try each action, avoiding negative-reward objects
        for action in actions:
            next_pos = self._get_next_position(current, action)
            if next_pos not in avoid_positions:
                return action

        # If all preferred actions lead to negative-reward objects, stay
        return 4

    def _get_action_toward(
        self,
        current: Tuple[int, int],
        target: Tuple[int, int],
        objects: dict
    ) -> int:
        """
        Get the action to move from current position toward target.

        Uses a simple greedy approach: move in the direction that
        reduces the distance to the target the most, while avoiding
        non-rewarding objects.

        Args:
            current: Current position (x, y)
            target: Target position (x, y)
            objects: Dictionary of objects in the environment

        Returns:
            Action (0=up, 1=down, 2=left, 3=right, 4=stay)
        """
        cx, cy = current
        tx, ty = target

        if current == target:
            return 4  # stay

        # Create a map of non-rewarding object positions
        non_rewarding_positions = set()
        for obj_id, obj_data in objects.items():
            props = obj_data['properties']
            is_rewarding = any(prop_value in self.reward_properties 
                             for prop_value in props.values())
            if not is_rewarding:
                non_rewarding_positions.add(obj_data['position'])

        dx = tx - cx
        dy = ty - cy

        # Try actions in order of preference based on distance to target
        actions = []
        if abs(dx) >= abs(dy):
            # Prioritize horizontal movement
            if dx > 0:
                actions = [3, 1 if dy > 0 else 0, 2]  # right, then vertical, then left
            else:
                actions = [2, 1 if dy > 0 else 0, 3]  # left, then vertical, then right
        else:
            # Prioritize vertical movement
            if dy > 0:
                actions = [1, 3 if dx > 0 else 2, 0]  # down, then horizontal, then up
            else:
                actions = [0, 3 if dx > 0 else 2, 1]  # up, then horizontal, then down

        # Try each action, avoiding non-rewarding objects
        for action in actions:
            next_pos = self._get_next_position(current, action)
            if next_pos not in non_rewarding_positions:
                return action

        # If all preferred actions lead to non-rewarding objects, stay
        return 4

    def _get_next_position(self, position: Tuple[int, int], action: int) -> Tuple[int, int]:
        """Get the next position after taking an action."""
        x, y = position
        if action == 0:  # up
            return (x, y - 1)
        elif action == 1:  # down
            return (x, y + 1)
        elif action == 2:  # left
            return (x - 1, y)
        elif action == 3:  # right
            return (x + 1, y)
        else:  # stay
            return position

    @staticmethod
    def _l2_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate L2 distance between two positions."""
        return ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5


class HumanAgentAStar(HumanAgent):
    """
    Human agent that uses A* for pathfinding.

    This version finds the true shortest path, accounting for grid boundaries.
    """

    def __init__(self, grid_size: int = 10):
        """Initialize the agent."""
        super().__init__()
        self.grid_size = grid_size
        self.path: list = []

    def reset(self, reward_properties: Set[str], additive_valuation: bool = False,
              object_rewards: Optional[Dict[int, float]] = None):
        """Reset the agent."""
        super().reset(reward_properties, additive_valuation, object_rewards)
        self.path = []

    def get_action(self, observation: dict) -> int:
        """Get action using A* pathfinding that avoids non-rewarding objects."""
        # Delegate to parent class for additive valuation mode
        if observation.get('additive_valuation', False) or self.additive_valuation:
            return super().get_action(observation)
        position = observation['human_position']
        objects = observation['objects']

        # Build set of non-rewarding object positions
        non_rewarding_positions = set()
        for obj_id, obj_data in objects.items():
            props = obj_data['properties']
            is_rewarding = any(prop_value in self.reward_properties 
                             for prop_value in props.values())
            if not is_rewarding:
                non_rewarding_positions.add(obj_data['position'])

        # If we have a valid path, follow it
        if self.path and self.path[0] != position:
            # Path is invalid, recalculate
            self.path = []

        if self.path and len(self.path) > 1:
            # Pop current position and move to next
            self.path.pop(0)
            next_pos = self.path[0]
            return self._get_action_to_adjacent(position, next_pos)

        # Find nearest rewarding object
        rewarding_positions = []
        for obj_id, obj_data in objects.items():
            props = obj_data['properties']
            for prop_value in props.values():
                if prop_value in self.reward_properties:
                    rewarding_positions.append(obj_data['position'])
                    break

        if not rewarding_positions:
            return 4  # stay

        # Find nearest by L2 distance
        nearest_pos = min(
            rewarding_positions,
            key=lambda p: self._l2_distance(position, p)
        )

        # Calculate path using A* (avoiding non-rewarding objects)
        self.path = self._astar(position, nearest_pos, non_rewarding_positions)

        if len(self.path) <= 1:
            return 4  # Already at target or no path

        # Move to next position in path
        self.path.pop(0)  # Remove current position
        next_pos = self.path[0]
        return self._get_action_to_adjacent(position, next_pos)

    def _get_action_to_adjacent(
        self,
        current: Tuple[int, int],
        adjacent: Tuple[int, int]
    ) -> int:
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

    def _astar(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        obstacles: Set[Tuple[int, int]]
    ) -> list:
        """
        A* pathfinding algorithm that avoids obstacles.

        Args:
            start: Starting position
            goal: Goal position
            obstacles: Set of positions to avoid (non-rewarding objects)

        Returns:
            List of positions from start to goal.
        """
        if start == goal:
            return [start]

        # Priority queue: (f_score, counter, position)
        counter = 0
        open_set = [(0, counter, start)]
        came_from = {}
        g_score = {start: 0}

        while open_set:
            _, _, current = heapq.heappop(open_set)

            if current == goal:
                # Reconstruct path
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

        # No path found (shouldn't happen in open grid)
        return [start]

    def _get_neighbors(self, pos: Tuple[int, int], obstacles: Set[Tuple[int, int]]) -> list:
        """Get valid neighboring positions, excluding obstacles."""
        x, y = pos
        neighbors = []

        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                if (nx, ny) not in obstacles:
                    neighbors.append((nx, ny))

        return neighbors
