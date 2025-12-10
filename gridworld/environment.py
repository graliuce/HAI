"""GridWorld environment for multi-agent object collection."""

from typing import Dict, List, Optional, Set, Tuple
import random
import numpy as np

from .objects import (
    GridObject,
    PROPERTY_CATEGORIES,
    PROPERTY_VALUES,
    create_random_object,
    sample_reward_properties
)


class GridWorld:
    """
    A gridworld environment where a human and robot collect objects.

    The human knows which object properties give reward.
    The robot must infer this from observing the human's behavior.
    """

    # Actions: 0=up, 1=down, 2=left, 3=right, 4=stay
    ACTIONS = ['up', 'down', 'left', 'right', 'stay']
    NUM_ACTIONS = 5

    def __init__(
        self,
        grid_size: int = 10,
        num_objects: int = 20,
        reward_ratio: float = 0.3,
        num_rewarding_properties: int = 2,
        num_distinct_properties: int = 2,
        seed: Optional[int] = None
    ):
        """
        Initialize the GridWorld environment.

        Args:
            grid_size: Size of the square grid (grid_size x grid_size)
            num_objects: Total number of objects to place in the grid
            reward_ratio: Proportion of objects that should be rewarding (approximate)
            num_rewarding_properties: K - number of properties that give reward
            num_distinct_properties: Number of property categories that vary (1-5)
            seed: Random seed for reproducibility
        """
        self.grid_size = grid_size
        self.num_objects = num_objects
        self.reward_ratio = reward_ratio
        self.num_rewarding_properties = num_rewarding_properties
        self.num_distinct_properties = min(num_distinct_properties, 5)

        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)

        # Active property categories (first N categories)
        self.active_categories = PROPERTY_CATEGORIES[:self.num_distinct_properties]

        # State variables (initialized in reset)
        self.objects: Dict[int, GridObject] = {}
        self.reward_properties: Set[str] = set()
        self.human_position: Tuple[int, int] = (0, 0)
        self.robot_position: Tuple[int, int] = (0, 0)
        self.human_collected: List[GridObject] = []
        self.robot_collected: List[GridObject] = []
        self.step_count: int = 0
        self.max_steps: int = 100
        self.done: bool = False

    def reset(self) -> Dict:
        """
        Reset the environment for a new episode.

        Returns:
            Initial observation for the robot
        """
        self.objects = {}
        self.human_collected = []
        self.robot_collected = []
        self.step_count = 0
        self.done = False

        # Sample reward properties
        self.reward_properties = sample_reward_properties(
            self.num_rewarding_properties,
            self.active_categories,
            self.rng
        )

        # Generate objects ensuring some are rewarding
        self._generate_objects()

        # Place agents at random positions (not on objects)
        occupied = {obj.position for obj in self.objects.values()}
        self.human_position = self._get_random_empty_position(occupied)
        occupied.add(self.human_position)
        self.robot_position = self._get_random_empty_position(occupied)

        return self._get_robot_observation()

    def _generate_objects(self):
        """Generate objects ensuring approximately reward_ratio are rewarding."""
        target_rewarding = int(self.num_objects * self.reward_ratio)
        target_non_rewarding = self.num_objects - target_rewarding

        occupied_positions = set()
        obj_id = 0

        # First, create rewarding objects
        rewarding_created = 0
        attempts = 0
        max_attempts = self.num_objects * 10

        while rewarding_created < target_rewarding and attempts < max_attempts:
            position = self._get_random_empty_position(occupied_positions)
            obj = create_random_object(obj_id, position, self.active_categories, self.rng)

            if obj.has_any_property(self.reward_properties):
                self.objects[obj_id] = obj
                occupied_positions.add(position)
                obj_id += 1
                rewarding_created += 1

            attempts += 1

        # Then, create non-rewarding objects
        non_rewarding_created = 0
        attempts = 0

        while non_rewarding_created < target_non_rewarding and attempts < max_attempts:
            position = self._get_random_empty_position(occupied_positions)
            obj = create_random_object(obj_id, position, self.active_categories, self.rng)

            if not obj.has_any_property(self.reward_properties):
                self.objects[obj_id] = obj
                occupied_positions.add(position)
                obj_id += 1
                non_rewarding_created += 1

            attempts += 1

        # If we couldn't create enough non-rewarding, just fill with random
        while len(self.objects) < self.num_objects:
            position = self._get_random_empty_position(occupied_positions)
            obj = create_random_object(obj_id, position, self.active_categories, self.rng)
            self.objects[obj_id] = obj
            occupied_positions.add(position)
            obj_id += 1

    def _get_random_empty_position(
        self,
        occupied: Set[Tuple[int, int]]
    ) -> Tuple[int, int]:
        """Get a random position that is not occupied."""
        while True:
            pos = (
                self.rng.randint(0, self.grid_size - 1),
                self.rng.randint(0, self.grid_size - 1)
            )
            if pos not in occupied:
                return pos

    def _get_robot_observation(self) -> Dict:
        """
        Get the current observation for the robot.

        The robot observes:
        - Its own position
        - Human's position
        - All objects with their positions and properties
        - Objects collected by the human (key for inference!)
        """
        return {
            'robot_position': self.robot_position,
            'human_position': self.human_position,
            'objects': {
                obj_id: {
                    'position': obj.position,
                    'properties': obj.get_properties()
                }
                for obj_id, obj in self.objects.items()
            },
            'human_collected': [
                {
                    'id': obj.id,
                    'properties': obj.get_properties()
                }
                for obj in self.human_collected
            ],
            'robot_collected': [obj.id for obj in self.robot_collected],
            'active_categories': self.active_categories,
            'step': self.step_count
        }

    def get_human_observation(self) -> Dict:
        """
        Get the current observation for the human.

        The human also observes the reward properties.
        """
        obs = self._get_robot_observation()
        obs['reward_properties'] = self.reward_properties
        return obs

    def step(
        self,
        robot_action: int,
        human_action: Optional[int] = None
    ) -> Tuple[Dict, float, bool, Dict]:
        """
        Execute one step in the environment.

        Args:
            robot_action: Action for the robot (0-4)
            human_action: Action for the human (if None, human uses its policy)

        Returns:
            observation: Robot's observation
            reward: Robot's reward for this step
            done: Whether the episode is finished
            info: Additional information
        """
        if self.done:
            return self._get_robot_observation(), 0.0, True, {}

        self.step_count += 1

        # Move human first (if action not provided, will be handled by agent)
        if human_action is not None:
            self.human_position = self._apply_action(
                self.human_position, human_action
            )

        # Move robot
        self.robot_position = self._apply_action(self.robot_position, robot_action)

        # Check for object collection
        robot_reward = self._check_collections()

        # Check if done
        self.done = (
            self.step_count >= self.max_steps or
            len(self.objects) == 0 or
            self._no_rewarding_objects_left()
        )

        info = {
            'human_collected_this_step': len(self.human_collected),
            'robot_collected_this_step': len(self.robot_collected),
            'reward_properties': self.reward_properties
        }

        return self._get_robot_observation(), robot_reward, self.done, info

    def _apply_action(
        self,
        position: Tuple[int, int],
        action: int
    ) -> Tuple[int, int]:
        """Apply an action and return the new position."""
        x, y = position

        if action == 0:  # up
            y = max(0, y - 1)
        elif action == 1:  # down
            y = min(self.grid_size - 1, y + 1)
        elif action == 2:  # left
            x = max(0, x - 1)
        elif action == 3:  # right
            x = min(self.grid_size - 1, x + 1)
        # action == 4 is stay

        return (x, y)

    def _check_collections(self) -> float:
        """Check if agents collected any objects. Return robot's reward."""
        robot_reward = 0.0

        # Find objects at agent positions
        objects_to_remove = []

        for obj_id, obj in self.objects.items():
            # Human collection
            if obj.position == self.human_position:
                self.human_collected.append(obj)
                objects_to_remove.append(obj_id)

            # Robot collection (only if human didn't also collect it)
            elif obj.position == self.robot_position:
                self.robot_collected.append(obj)
                objects_to_remove.append(obj_id)

                # Robot gets reward for collecting rewarding objects
                if obj.has_any_property(self.reward_properties):
                    robot_reward += 1.0

        # Remove collected objects
        for obj_id in objects_to_remove:
            del self.objects[obj_id]

        return robot_reward

    def _no_rewarding_objects_left(self) -> bool:
        """Check if there are no more rewarding objects."""
        for obj in self.objects.values():
            if obj.has_any_property(self.reward_properties):
                return False
        return True

    def get_state_for_robot(self) -> Tuple:
        """
        Get a hashable state representation for Q-learning.

        This is a simplified state that captures:
        - Robot's position (discretized)
        - Human's recent collection history (property pattern)
        - Nearest objects and their properties
        """
        # Encode human collected properties as a frozenset
        human_collected_props = frozenset()
        if self.human_collected:
            # Get properties from recently collected objects
            recent = self.human_collected[-3:]  # Last 3 collected
            props = set()
            for obj in recent:
                for cat in self.active_categories:
                    props.add((cat, obj.get_properties()[cat]))
            human_collected_props = frozenset(props)

        # Find nearest objects to robot
        nearest_info = self._get_nearest_objects_info(self.robot_position, n=3)

        return (
            self.robot_position,
            self.human_position,
            human_collected_props,
            nearest_info
        )

    def _get_nearest_objects_info(
        self,
        position: Tuple[int, int],
        n: int = 3
    ) -> Tuple:
        """Get info about nearest n objects to a position."""
        if not self.objects:
            return tuple()

        # Calculate distances
        distances = []
        for obj_id, obj in self.objects.items():
            dist = self._l2_distance(position, obj.position)
            distances.append((dist, obj_id, obj))

        distances.sort(key=lambda x: x[0])

        # Get info for nearest objects
        info = []
        for i, (dist, obj_id, obj) in enumerate(distances[:n]):
            # Relative position
            rel_x = obj.position[0] - position[0]
            rel_y = obj.position[1] - position[1]

            # Discretized relative position
            rel_x_disc = max(-2, min(2, rel_x // 2))
            rel_y_disc = max(-2, min(2, rel_y // 2))

            # Object properties (only active ones)
            props = tuple(
                obj.get_properties()[cat]
                for cat in self.active_categories
            )

            info.append((rel_x_disc, rel_y_disc, props))

        return tuple(info)

    @staticmethod
    def _l2_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate L2 distance between two positions."""
        return ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5

    def get_rewarding_objects(self) -> List[GridObject]:
        """Get list of objects that give reward."""
        return [
            obj for obj in self.objects.values()
            if obj.has_any_property(self.reward_properties)
        ]

    def render(self) -> str:
        """Render the environment as a string."""
        grid = [['.' for _ in range(self.grid_size)] for _ in range(self.grid_size)]

        # Place objects
        for obj in self.objects.values():
            x, y = obj.position
            if obj.has_any_property(self.reward_properties):
                grid[y][x] = '*'  # Rewarding object
            else:
                grid[y][x] = 'o'  # Non-rewarding object

        # Place agents (overwrite objects if on same position)
        hx, hy = self.human_position
        rx, ry = self.robot_position

        grid[hy][hx] = 'H'
        grid[ry][rx] = 'R'

        # If both agents on same position
        if self.human_position == self.robot_position:
            grid[hy][hx] = 'X'

        # Build string
        lines = ['=' * (self.grid_size * 2 + 1)]
        for row in grid:
            lines.append('|' + ' '.join(row) + '|')
        lines.append('=' * (self.grid_size * 2 + 1))

        # Add info
        lines.append(f"Step: {self.step_count}/{self.max_steps}")
        lines.append(f"Objects remaining: {len(self.objects)}")
        lines.append(f"Human collected: {len(self.human_collected)}")
        lines.append(f"Robot collected: {len(self.robot_collected)}")
        lines.append(f"Reward properties: {self.reward_properties}")

        return '\n'.join(lines)
