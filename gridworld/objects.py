"""Object definitions for the gridworld environment."""

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple
import random

# Property categories and their possible values
PROPERTY_CATEGORIES = ['color', 'shape', 'size', 'pattern', 'opacity']

PROPERTY_VALUES = {
    'color': ['red', 'blue', 'green'],
    'shape': ['circle', 'square', 'triangle'],
    'size': ['small', 'medium', 'large'],
    'pattern': ['solid', 'striped', 'dotted'],
    'opacity': ['transparent', 'translucent', 'opaque']
}

# Default values for inactive property categories
DEFAULT_VALUES = {
    'color': 'red',
    'shape': 'circle',
    'size': 'medium',
    'pattern': 'solid',
    'opacity': 'opaque'
}


@dataclass
class GridObject:
    """An object in the gridworld with visual properties."""

    id: int
    position: Tuple[int, int]
    color: str
    shape: str
    size: str
    pattern: str
    opacity: str

    def get_properties(self) -> Dict[str, str]:
        """Return all properties as a dictionary."""
        return {
            'color': self.color,
            'shape': self.shape,
            'size': self.size,
            'pattern': self.pattern,
            'opacity': self.opacity
        }

    def get_property_values(self) -> Set[str]:
        """Return the set of all property values this object has."""
        return {self.color, self.shape, self.size, self.pattern, self.opacity}

    def has_property(self, prop_value: str) -> bool:
        """Check if this object has a specific property value."""
        return prop_value in self.get_property_values()

    def has_any_property(self, prop_values: Set[str]) -> bool:
        """Check if this object has any of the given property values."""
        return bool(self.get_property_values() & prop_values)

    def to_feature_vector(self, active_categories: List[str]) -> List[int]:
        """
        Convert object to a feature vector for the robot's observation.
        Only includes active property categories.
        """
        features = []
        props = self.get_properties()

        for category in active_categories:
            value = props[category]
            # One-hot encode the property value
            for possible_value in PROPERTY_VALUES[category]:
                features.append(1 if value == possible_value else 0)

        return features


def create_random_object(
    obj_id: int,
    position: Tuple[int, int],
    active_categories: List[str],
    rng: random.Random = None
) -> GridObject:
    """
    Create a random object with properties.

    Args:
        obj_id: Unique identifier for the object
        position: (x, y) position in the grid
        active_categories: List of property categories that should vary
        rng: Random number generator (optional)

    Returns:
        A GridObject with random properties for active categories
    """
    if rng is None:
        rng = random.Random()

    props = {}
    for category in PROPERTY_CATEGORIES:
        if category in active_categories:
            props[category] = rng.choice(PROPERTY_VALUES[category])
        else:
            props[category] = DEFAULT_VALUES[category]

    return GridObject(
        id=obj_id,
        position=position,
        **props
    )


def sample_reward_properties(
    k: int,
    active_categories: List[str],
    rng: random.Random = None
) -> Set[str]:
    """
    Sample K property values that will give reward.

    Args:
        k: Number of rewarding properties to sample
        active_categories: List of active property categories
        rng: Random number generator (optional)

    Returns:
        Set of property values that give reward
    """
    if rng is None:
        rng = random.Random()

    # Get all possible property values from active categories
    all_values = []
    for category in active_categories:
        all_values.extend(PROPERTY_VALUES[category])

    # Sample K unique values
    return set(rng.sample(all_values, min(k, len(all_values))))
