"""Human agent with fixed greedy policy."""

from typing import Optional, Set, Tuple
import heapq


class HumanAgent:
    """
    A simulated human agent that knows the reward specification.

    Policy: Greedy - finds the nearest rewarding object by L2 distance
    and takes the shortest path to collect it.
    """

    def __init__(self):
        """Initialize the human agent."""
        self.reward_properties: Set[str] = set()
        self.current_target: Optional[Tuple[int, int]] = None

    def reset(self, reward_properties: Set[str]):
        """
        Reset the agent for a new episode.

        Args:
            reward_properties: Set of property values that give reward
        """
        self.reward_properties = reward_properties
        self.current_target = None

    def get_action(self, observation: dict) -> int:
        """
        Get the action for the human agent.

        Policy:
        1. Find the nearest rewarding object by L2 distance
        2. Take a step on the shortest path toward it

        Args:
            observation: Full observation including reward_properties

        Returns:
            Action (0=up, 1=down, 2=left, 3=right, 4=stay)
        """
        position = observation['human_position']
        objects = observation['objects']

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

        # Get action toward nearest object (greedy step)
        return self._get_action_toward(position, nearest_pos)

    def _get_action_toward(
        self,
        current: Tuple[int, int],
        target: Tuple[int, int]
    ) -> int:
        """
        Get the action to move from current position toward target.

        Uses a simple greedy approach: move in the direction that
        reduces the distance to the target the most.

        Args:
            current: Current position (x, y)
            target: Target position (x, y)

        Returns:
            Action (0=up, 1=down, 2=left, 3=right, 4=stay)
        """
        cx, cy = current
        tx, ty = target

        if current == target:
            return 4  # stay

        dx = tx - cx
        dy = ty - cy

        # Prioritize the larger displacement
        if abs(dx) >= abs(dy):
            if dx > 0:
                return 3  # right
            else:
                return 2  # left
        else:
            if dy > 0:
                return 1  # down
            else:
                return 0  # up

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

    def reset(self, reward_properties: Set[str]):
        """Reset the agent."""
        super().reset(reward_properties)
        self.path = []

    def get_action(self, observation: dict) -> int:
        """Get action using A* pathfinding."""
        position = observation['human_position']
        objects = observation['objects']

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

        # Calculate path using A*
        self.path = self._astar(position, nearest_pos)

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
        goal: Tuple[int, int]
    ) -> list:
        """
        A* pathfinding algorithm.

        Returns list of positions from start to goal.
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

            for neighbor in self._get_neighbors(current):
                tentative_g = g_score[current] + 1

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self._l2_distance(neighbor, goal)
                    counter += 1
                    heapq.heappush(open_set, (f_score, counter, neighbor))

        # No path found (shouldn't happen in open grid)
        return [start]

    def _get_neighbors(self, pos: Tuple[int, int]) -> list:
        """Get valid neighboring positions."""
        x, y = pos
        neighbors = []

        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                neighbors.append((nx, ny))

        return neighbors
