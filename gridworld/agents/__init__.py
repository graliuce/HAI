"""Agent implementations."""

from .human import HumanAgent, HumanAgentAStar
from .belief_based_robot import BeliefBasedRobotAgent, GaussianBeliefState

# Import DQN-based agents only if torch is available
try:
    from .hierarchical_dqn_robot import HierarchicalDQNRobotAgent
    from .query_augmented_robot import QueryAugmentedRobotAgent, PreferenceBeliefs
    TORCH_AGENTS_AVAILABLE = True
except ImportError:
    HierarchicalDQNRobotAgent = None
    QueryAugmentedRobotAgent = None
    PreferenceBeliefs = None
    TORCH_AGENTS_AVAILABLE = False

__all__ = [
    'HumanAgent',
    'HumanAgentAStar',
    'BeliefBasedRobotAgent',
    'GaussianBeliefState',
    'TORCH_AGENTS_AVAILABLE',
]

if TORCH_AGENTS_AVAILABLE:
    __all__.extend([
        'HierarchicalDQNRobotAgent',
        'QueryAugmentedRobotAgent',
        'PreferenceBeliefs',
    ])
