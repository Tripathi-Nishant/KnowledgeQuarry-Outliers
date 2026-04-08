from dataclasses import dataclass, field
from typing import Optional

@dataclass
class Vehicle:
    id: int
    x: float                # Longitudinal position
    lane: int               # Current lane (0, 1, 2)
    target_lane: int        # Lane it is merging toward
    speed: float            # Current speed (units/step)
    type: str               # "cooperative", "aggressive", "rl"
    color: str              # Hex color for rendering
    born_step: int          # Step when spawned
    
    # State flags
    merging: bool = False
    merge_progress: float = 0.0
    passed: bool = False
    just_passed: bool = False  # Track if it passed in the current step
    in_conflict: bool = False  # Track if it's in a collision/near-miss state
    
    # Metrics
    wait_time: float = 0.0     # Cumulative time spent below a threshold speed
    distance_traveled: float = 0.0

    def __post_init__(self):
        self.target_lane = self.lane
