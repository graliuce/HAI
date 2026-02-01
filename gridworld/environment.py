"""GridWorld environment for multi-agent object collection."""

from typing import Dict, List, Optional, Set, Tuple
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, Rectangle, RegularPolygon

from .objects import (
    GridObject,
    PROPERTY_CATEGORIES,
    PROPERTY_VALUES,
    MAX_PROPERTY_VALUES,
    create_random_object,
    get_property_values,
)


class GridWorld:
    """
    A gridworld environment where a human and robot collect objects.

    The human knows which object properties give reward.
    The robot must infer this from observing the human's behavior.

    Uses additive valuation: each property value has a Gaussian reward,
    and object reward is the sum of its property value rewards.
    """

    # Actions: 0=up, 1=down, 2=left, 3=right, 4=stay
    ACTIONS = ['up', 'down', 'left', 'right', 'stay']
    NUM_ACTIONS = 5

    def __init__(
        self,
        grid_size: int = 10,
        num_objects: int = 20,
        num_distinct_properties: int = 2,
        num_property_values: int = MAX_PROPERTY_VALUES,
        seed: Optional[int] = None
    ):
        """
        Initialize the GridWorld environment.

        Args:
            grid_size: Size of the square grid (grid_size x grid_size)
            num_objects: Total number of objects to place in the grid
            num_distinct_properties: Number of property categories that vary (1-5)
            num_property_values: Number of values per property category (1-5, default 5)
            seed: Random seed for reproducibility
        """
        self.grid_size = grid_size
        self.num_objects = num_objects
        self.num_distinct_properties = min(num_distinct_properties, 5)
        self.num_property_values = max(1, min(num_property_values, MAX_PROPERTY_VALUES))

        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)

        # Active property categories (first N categories)
        self.active_categories = PROPERTY_CATEGORIES[:self.num_distinct_properties]

        # State variables (initialized in reset)
        self.objects: Dict[int, GridObject] = {}
        self.reward_properties: Set[str] = set()
        # Maps property value -> reward
        self.property_value_rewards: Dict[str, float] = {}
        self.human_position: Tuple[int, int] = (0, 0)
        self.robot_position: Tuple[int, int] = (0, 0)
        self.human_collected: List[GridObject] = []
        self.robot_collected: List[GridObject] = []
        self.step_count: int = 0
        self.max_steps: int = 20
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
        self.property_value_rewards = {}

        # Sample Gaussian rewards for each property value
        self._sample_property_value_rewards()
        # Generate objects (no need for reward_properties in additive mode)
        self.reward_properties = set()
        self._generate_objects()

        # Place agents at random positions (not on objects)
        occupied = {obj.position for obj in self.objects.values()}
        self.human_position = self._get_random_empty_position(occupied)
        occupied.add(self.human_position)
        self.robot_position = self._get_random_empty_position(occupied)

        return self._get_robot_observation()

    def _sample_property_value_rewards(self):
        """Sample Gaussian rewards for each property value in active categories.

        For ordinal properties (size, opacity), imposes linear structure where
        adjacent values have similar rewards. For other properties, samples independently.
        """
        self.property_value_rewards = {}

        # Properties with natural ordinal structure
        ordinal_properties = ['size', 'opacity']

        for category in self.active_categories:
            values = get_property_values(category, self.num_property_values)

            if category in ordinal_properties:
                # For ordinal properties, sample a slope and create linear structure
                # The slope determines the strength and direction of preference
                slope = self.np_rng.randn()

                # Create linearly spaced rewards centered at 0
                n = len(values)
                for i, value in enumerate(values):
                    # Map index to [-1, 1] range, then scale by slope
                    # This keeps expected value at 0 and creates monotonic structure
                    position = 2 * (i / (n - 1)) - 1 if n > 1 else 0
                    self.property_value_rewards[value] = slope * position
            else:
                # For non-ordinal properties (color, shape, pattern),
                # sample independently from standard Gaussian
                for value in values:
                    self.property_value_rewards[value] = self.np_rng.randn()

    def get_object_reward(self, obj: GridObject) -> float:
        """
        Get the total reward for an object.

        This is the sum of rewards for all its property values.
        """
        total = 0.0
        for value in obj.get_property_values():
            if value in self.property_value_rewards:
                total += self.property_value_rewards[value]
        return total

    def _generate_objects(self):
        """Generate objects for the environment."""
        occupied_positions = set()

        for obj_id in range(self.num_objects):
            position = self._get_random_empty_position(occupied_positions)
            obj = create_random_object(
                obj_id, position, self.active_categories, self.rng, self.num_property_values
            )
            self.objects[obj_id] = obj
            occupied_positions.add(position)

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
        - Whether robot can collect (human has collected at least one)
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
            'robot_can_collect': len(self.human_collected) > 0,
            'active_categories': self.active_categories,
            'step': self.step_count
        }

    def get_human_observation(self) -> Dict:
        """
        Get the current observation for the human.

        The human also observes the reward properties and precomputed object_rewards.
        """
        obs = self._get_robot_observation()
        obs['reward_properties'] = self.reward_properties
        obs['property_value_rewards'] = self.property_value_rewards.copy()
        # Precompute object rewards for human decision-making
        obs['object_rewards'] = {
            obj_id: self.get_object_reward(obj)
            for obj_id, obj in self.objects.items()
        }
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
            self.step_count >= self.max_steps
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

        # Robot can only collect after human has collected at least one object
        robot_can_collect = len(self.human_collected) > 0

        # Find objects at agent positions
        objects_to_remove = []

        for obj_id, obj in self.objects.items():
            # Human collection
            if obj.position == self.human_position:
                self.human_collected.append(obj)
                objects_to_remove.append(obj_id)

            # Robot collection (only if human didn't also collect it AND robot is unlocked)
            elif obj.position == self.robot_position and robot_can_collect:
                self.robot_collected.append(obj)
                objects_to_remove.append(obj_id)

                # Robot gets reward based on object value
                robot_reward += self.get_object_reward(obj)

        # Remove collected objects
        for obj_id in objects_to_remove:
            del self.objects[obj_id]

        return robot_reward

    def _no_rewarding_objects_left(self) -> bool:
        """Check if there are no more rewarding objects."""
        # Consider objects with positive reward as rewarding
        for obj in self.objects.values():
            if self.get_object_reward(obj) > 0:
                return False
        return True

    @staticmethod
    def _l2_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate L2 distance between two positions."""
        return ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5

    def get_rewarding_objects(self) -> List[GridObject]:
        """Get list of objects that give positive reward."""
        return [
            obj for obj in self.objects.values()
            if self.get_object_reward(obj) > 0
        ]

    def render(self) -> str:
        """Render the environment as a string."""
        grid = [['.' for _ in range(self.grid_size)] for _ in range(self.grid_size)]

        # Place objects
        for obj in self.objects.values():
            x, y = obj.position
            is_rewarding = self.get_object_reward(obj) > 0
            if is_rewarding:
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

        return '\n'.join(lines)

    def render_to_image(self, save_path: Optional[str] = None, show: bool = False):
        """
        Render the environment as a matplotlib figure and optionally save to PNG.

        Args:
            save_path: Path to save the PNG image (optional)
            show: Whether to display the plot

        Returns:
            matplotlib figure object
        """
        # Color mappings
        COLOR_MAP = {
            'red': '#E74C3C',
            'blue': '#3498DB',
            'green': '#2ECC71',
            'yellow': '#F1C40F',
            'purple': '#9B59B6'
        }

        # Size mappings (relative to cell size)
        SIZE_MAP = {
            'tiny': 0.10,
            'small': 0.15,
            'medium': 0.22,
            'large': 0.30,
            'huge': 0.38
        }

        # Pattern mappings (hatch patterns)
        PATTERN_MAP = {
            'solid': '',
            'striped': '///',
            'dotted': '...',
            'checkered': 'xx',
            'gradient': '\\\\',
        }

        # Opacity mappings
        OPACITY_MAP = {
            'transparent': 0.2,
            'faint': 0.4,
            'translucent': 0.6,
            'semi-opaque': 0.8,
            'opaque': 1.0
        }

        fig, ax = plt.subplots(figsize=(10, 10))

        # Draw grid
        for i in range(self.grid_size + 1):
            ax.axhline(y=i, color='lightgray', linewidth=0.5)
            ax.axvline(x=i, color='lightgray', linewidth=0.5)

        # Draw objects
        for obj in self.objects.values():
            x, y = obj.position
            props = obj.get_properties()

            # Get visual properties
            color = COLOR_MAP.get(props['color'], '#888888')
            size = SIZE_MAP.get(props['size'], 0.25)
            pattern = PATTERN_MAP.get(props['pattern'], '')
            opacity = OPACITY_MAP.get(props['opacity'], 1.0)

            # Check if rewarding
            is_rewarding = self.get_object_reward(obj) > 0
            edgecolor = 'gold' if is_rewarding else 'black'
            linewidth = 3 if is_rewarding else 1

            # Draw shape based on shape property
            center_x = x + 0.5
            center_y = self.grid_size - y - 0.5  # Flip y for display

            if props['shape'] == 'circle':
                patch = Circle(
                    (center_x, center_y), size,
                    facecolor=color, edgecolor=edgecolor,
                    linewidth=linewidth, alpha=opacity, hatch=pattern
                )
            elif props['shape'] == 'square':
                patch = Rectangle(
                    (center_x - size, center_y - size), size * 2, size * 2,
                    facecolor=color, edgecolor=edgecolor,
                    linewidth=linewidth, alpha=opacity, hatch=pattern
                )
            elif props['shape'] == 'triangle':
                patch = RegularPolygon(
                    (center_x, center_y), numVertices=3, radius=size * 1.2,
                    facecolor=color, edgecolor=edgecolor,
                    linewidth=linewidth, alpha=opacity, hatch=pattern
                )
            elif props['shape'] == 'diamond':
                patch = RegularPolygon(
                    (center_x, center_y), numVertices=4, radius=size * 1.2,
                    facecolor=color, edgecolor=edgecolor,
                    linewidth=linewidth, alpha=opacity, hatch=pattern
                )
            else:  # pentagon
                patch = RegularPolygon(
                    (center_x, center_y), numVertices=5, radius=size * 1.2,
                    orientation=np.pi / 10,  # Rotate so one vertex points up
                    facecolor=color, edgecolor=edgecolor,
                    linewidth=linewidth, alpha=opacity, hatch=pattern
                )

            ax.add_patch(patch)

        # Draw human agent
        hx, hy = self.human_position
        human_patch = Circle(
            (hx + 0.5, self.grid_size - hy - 0.5), 0.35,
            facecolor='#9B59B6', edgecolor='black', linewidth=2
        )
        ax.add_patch(human_patch)
        ax.text(hx + 0.5, self.grid_size - hy - 0.5, 'H',
                ha='center', va='center', fontsize=14, fontweight='bold', color='white')

        # Draw robot agent
        rx, ry = self.robot_position
        robot_patch = Rectangle(
            (rx + 0.15, self.grid_size - ry - 0.85), 0.7, 0.7,
            facecolor='#34495E', edgecolor='black', linewidth=2
        )
        ax.add_patch(robot_patch)
        ax.text(rx + 0.5, self.grid_size - ry - 0.5, 'R',
                ha='center', va='center', fontsize=14, fontweight='bold', color='white')

        # Set axis properties
        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(0, self.grid_size)
        ax.set_aspect('equal')
        ax.set_xticks(range(self.grid_size + 1))
        ax.set_yticks(range(self.grid_size + 1))
        ax.tick_params(axis='both', which='both', length=0)
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        # Calculate rewards for display
        human_reward = sum(self.get_object_reward(obj) for obj in self.human_collected)
        robot_reward = sum(self.get_object_reward(obj) for obj in self.robot_collected)

        # Add title with episode info
        reward_info = 'Additive Valuation Mode'
        reward_display = f'Human Reward: {human_reward:+.2f} | Robot Reward: {robot_reward:+.2f}'

        title_lines = [
            f'Step: {self.step_count}/{self.max_steps}',
            f'Distinct Properties: {self.num_distinct_properties} ({", ".join(self.active_categories)})',
            reward_info,
            reward_display
        ]
        ax.set_title('\n'.join(title_lines), fontsize=11, pad=10)

        # Add legend
        legend_elements = [
            mpatches.Patch(facecolor='#9B59B6', edgecolor='black', label='Human (H)'),
            mpatches.Patch(facecolor='#34495E', edgecolor='black', label='Robot (R)'),
            mpatches.Patch(facecolor='gray', edgecolor='gold', linewidth=2, label='Rewarding Object'),
            mpatches.Patch(facecolor='gray', edgecolor='black', label='Non-rewarding Object'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1),
                  fontsize=9)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def render_to_array(self, query_info: dict = None) -> np.ndarray:
        """
        Render the environment to a numpy array for GIF generation.

        Args:
            query_info: Optional dictionary with query information to display
                       Keys: 'weights' (dict), 'query_num' (int)

        Returns:
            numpy array of shape (height, width, 3) with RGB values
        """
        # Color mappings
        COLOR_MAP = {
            'red': '#E74C3C',
            'blue': '#3498DB',
            'green': '#2ECC71',
            'yellow': '#F1C40F',
            'purple': '#9B59B6'
        }

        SIZE_MAP = {
            'tiny': 0.10,
            'small': 0.15,
            'medium': 0.22,
            'large': 0.30,
            'huge': 0.38
        }

        PATTERN_MAP = {
            'solid': '',
            'striped': '///',
            'dotted': '...',
            'checkered': 'xx',
            'gradient': '\\\\',
        }

        OPACITY_MAP = {
            'transparent': 0.2,
            'faint': 0.4,
            'translucent': 0.6,
            'semi-opaque': 0.8,
            'opaque': 1.0
        }

        # Use wider figure to accommodate side panels for rewards (left) and queries (right)
        fig, ax = plt.subplots(figsize=(14, 8))

        # Draw grid
        for i in range(self.grid_size + 1):
            ax.axhline(y=i, color='lightgray', linewidth=0.5)
            ax.axvline(x=i, color='lightgray', linewidth=0.5)

        # Draw objects
        for obj in self.objects.values():
            x, y = obj.position
            props = obj.get_properties()

            color = COLOR_MAP.get(props['color'], '#888888')
            size = SIZE_MAP.get(props['size'], 0.22)
            pattern = PATTERN_MAP.get(props['pattern'], '')
            opacity = OPACITY_MAP.get(props['opacity'], 1.0)

            is_rewarding = self.get_object_reward(obj) > 0
            edgecolor = 'gold' if is_rewarding else 'black'
            linewidth = 3 if is_rewarding else 1

            center_x = x + 0.5
            center_y = self.grid_size - y - 0.5

            if props['shape'] == 'circle':
                patch = Circle(
                    (center_x, center_y), size,
                    facecolor=color, edgecolor=edgecolor,
                    linewidth=linewidth, alpha=opacity, hatch=pattern
                )
            elif props['shape'] == 'square':
                patch = Rectangle(
                    (center_x - size, center_y - size), size * 2, size * 2,
                    facecolor=color, edgecolor=edgecolor,
                    linewidth=linewidth, alpha=opacity, hatch=pattern
                )
            elif props['shape'] == 'triangle':
                patch = RegularPolygon(
                    (center_x, center_y), numVertices=3, radius=size * 1.2,
                    facecolor=color, edgecolor=edgecolor,
                    linewidth=linewidth, alpha=opacity, hatch=pattern
                )
            elif props['shape'] == 'diamond':
                patch = RegularPolygon(
                    (center_x, center_y), numVertices=4, radius=size * 1.2,
                    facecolor=color, edgecolor=edgecolor,
                    linewidth=linewidth, alpha=opacity, hatch=pattern
                )
            else:  # pentagon
                patch = RegularPolygon(
                    (center_x, center_y), numVertices=5, radius=size * 1.2,
                    orientation=np.pi / 10,  # Rotate so one vertex points up
                    facecolor=color, edgecolor=edgecolor,
                    linewidth=linewidth, alpha=opacity, hatch=pattern
                )

            ax.add_patch(patch)

        # Draw human agent
        hx, hy = self.human_position
        human_patch = Circle(
            (hx + 0.5, self.grid_size - hy - 0.5), 0.35,
            facecolor='#9B59B6', edgecolor='black', linewidth=2
        )
        ax.add_patch(human_patch)
        ax.text(hx + 0.5, self.grid_size - hy - 0.5, 'H',
                ha='center', va='center', fontsize=12, fontweight='bold', color='white')

        # Draw robot agent
        rx, ry = self.robot_position
        robot_patch = Rectangle(
            (rx + 0.15, self.grid_size - ry - 0.85), 0.7, 0.7,
            facecolor='#34495E', edgecolor='black', linewidth=2
        )
        ax.add_patch(robot_patch)
        ax.text(rx + 0.5, self.grid_size - ry - 0.5, 'R',
                ha='center', va='center', fontsize=12, fontweight='bold', color='white')

        # Set axis properties
        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(0, self.grid_size)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])

        # Calculate rewards for display
        human_reward = sum(self.get_object_reward(obj) for obj in self.human_collected)
        robot_reward = sum(self.get_object_reward(obj) for obj in self.robot_collected)

        # Add title
        reward_info = 'Additive Mode'
        reward_display = f'H: {human_reward:+.2f}  R: {robot_reward:+.2f}'

        title_lines = [
            f'Step: {self.step_count}/{self.max_steps}  |  '
            f'Props: {self.num_distinct_properties} ({", ".join(self.active_categories)})',
            f'{reward_info}  |  {reward_display}'
        ]
        ax.set_title('\n'.join(title_lines), fontsize=10, pad=5)

        # Add true reward values on the left side
        if hasattr(self, 'property_value_rewards'):
            # Get property value rewards
            prop_rewards = self.property_value_rewards
            sorted_rewards = sorted(prop_rewards.items(), key=lambda x: x[1], reverse=True)

            # Create text showing top positive and negative rewards
            reward_text = "Property Rewards:\n" + "-" * 20 + "\n"
            for prop, reward in sorted_rewards[:8]:  # Show top 8
                sign = "+" if reward >= 0 else ""
                reward_text += f"{prop:12s} {sign}{reward:.2f}\n"

            # Add text box on the left side
            fig.text(
                0.02, 0.5, reward_text,
                fontsize=9,
                verticalalignment='center',
                horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='none', edgecolor='gray', linewidth=2),
                family='monospace'
            )

        # Add query information on the right side (always show if query has been made)
        if query_info:
            weights = query_info.get('weights', {})
            query_num = query_info.get('query_num', 0)

            # Filter to show only non-zero weights, sorted by absolute value
            non_zero_weights = {k: v for k, v in weights.items() if abs(v) > 0.01}
            sorted_weights = sorted(non_zero_weights.items(), key=lambda x: abs(x[1]), reverse=True)

            # Create text for top weights
            query_text = f"Query {query_num}:\n" + "-" * 20 + "\n"
            for i, (prop, weight) in enumerate(sorted_weights[:8]):  # Show up to 8
                sign = "+" if weight > 0 else ""
                query_text += f"{prop:12s} {sign}{weight:.2f}\n"

            # Add text box on the right side
            fig.text(
                0.98, 0.5, query_text,
                fontsize=9,
                verticalalignment='center',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='none', edgecolor='gray', linewidth=2),
                family='monospace'
            )

        # Convert to array
        fig.canvas.draw()
        # Get RGBA buffer and convert to RGB
        buf = np.asarray(fig.canvas.buffer_rgba())
        data = buf[:, :, :3]  # Drop alpha channel

        plt.close(fig)

        return data
