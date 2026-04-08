import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from typing import List, Dict, Optional, Tuple
from envs.vehicle import Vehicle
from agents.behavior_profiles import PROFILES, idm_acceleration
from agents.vehicle_spawner import VehicleSpawner

class BottleneckEnv(gym.Env):
    def __init__(self,
                 n_lanes=3,
                 road_length=1000,
                 bottleneck_x=620,
                 merge_start=430,
                 max_vehicles=40,
                 arrival_rate=4.0,           # vehicles per second
                 aggressive_ratio=0.3,
                 cooperative_ratio=0.5,
                 rl_ratio=0.2,
                 max_steps=2000,
                 reward_mode="mixed",      # "selfish", "cooperative", or "mixed"
                 alpha=0.4,                # weight for individual reward
                 seed=None):
        super(BottleneckEnv, self).__init__()
        
        self.n_lanes = n_lanes
        self.road_length = road_length
        self.bottleneck_x = bottleneck_x
        self.merge_start = merge_start
        self.max_steps = max_steps
        self.reward_mode = reward_mode
        self.alpha = alpha
        
        self.spawner = VehicleSpawner(
            arrival_rate=arrival_rate,
            aggressive_ratio=aggressive_ratio,
            cooperative_ratio=cooperative_ratio,
            rl_ratio=rl_ratio,
            max_vehicles=max_vehicles
        )

        # 0=Accel (+0.3), 1=Maintain, 2=Decel (-0.3), 3=Merge Left, 4=Merge Right
        self.action_space = spaces.Discrete(5)
        
        # Observation vector (shape: 12,)
        # [self_x/L, self_s/S, self_lane/N, dist_bottleneck/L, dist_leader/M, leader_s/S, 
        #  dist_left/M, dist_right/M, lane_density/10, queue_density/V, is_merge_zone, time_pressure]
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(12,), dtype=np.float32)

        self.vehicles: List[Vehicle] = []
        self.current_step = 0
        self.passed_count = 0
        self.conflict_count = 0
        self.total_wait = 0
        self.wait_count = 0
        
        # Simulation settings
        self.dt = 0.1  # 0.1s simulation time resolution
        self.MAX_SPEED = 4.0
        self.MIN_SPEED = 0.0
        self.MAX_ACCEL = 0.4
        self.SAFE_DISTANCE = 28
        self.EMERGENCY_DISTANCE = 16
        self.SENSOR_RANGE = 100.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.vehicles = []
        self.current_step = 0
        self.passed_count = 0
        self.conflict_count = 0
        self.total_wait = 0
        self.wait_count = 0
        
        # Initial spawn: several cooperative and aggressive vehicles pre-placed
        # For simplicity, start with empty road until spawner acts
        # Or place a few initial ones spread out
        for _ in range(5):
            v = self.spawner.spawn(0)
            v.x = random.uniform(50, 400)
            self.vehicles.append(v)
            
        return self._get_obs(self.vehicles[0] if self.vehicles else None), {}

    def _get_obs(self, v: Optional[Vehicle]):
        if v is None:
            return np.zeros(12, dtype=np.float32)
        
        def norm_x(val): return np.clip(val / self.road_length, 0, 1)
        def norm_s(val): return np.clip(val / self.MAX_SPEED, 0, 1)
        def norm_l(val): return np.clip(val / (self.n_lanes - 1), 0, 1) if self.n_lanes > 1 else 0.5
        def norm_dist(val): return np.clip(val / self.SENSOR_RANGE, 0, 1)

        leader, l_dist = self._get_leader(v)
        left_near, ln_dist = self._get_neighbor(v, v.lane - 1)
        right_near, rn_dist = self._get_neighbor(v, v.lane + 1)
        
        # Count vehicles in lane within 80 units ahead
        lane_density = len([ov for ov in self.vehicles if ov.lane == v.lane and 0 < ov.x - v.x < 80]) / 8.0
        # Queue density: vehicles between merge_start and bottleneck with speed < 0.5
        queue_count = len([ov for ov in self.vehicles if self.merge_start < ov.x < self.bottleneck_x and ov.speed < 0.5])
        queue_density = queue_count / 10.0 # Normalized by 10

        obs = np.array([
            norm_x(v.x),
            norm_s(v.speed),
            norm_l(v.lane),
            norm_x(self.bottleneck_x - v.x),
            norm_dist(l_dist),
            norm_s(leader.speed if leader else 0),
            norm_dist(ln_dist),
            norm_dist(rn_dist),
            np.clip(lane_density, 0, 1),
            np.clip(queue_density, 0, 1),
            1.0 if v.x > self.merge_start else 0.0,
            norm_x(self.bottleneck_x - v.x) * (self.current_step / self.max_steps) # Urgency
        ], dtype=np.float32)

        return obs

    def step(self, action):
        self.current_step += 1
        self.passed_count_this_step = 0
        
        # 1. Update RL agent(s)
        # Note: In single-agent training, we pick the first 'rl' type vehicle as the agent.
        # Other RL vehicles behave with rule-based Cooperative logic or Maintain
        agent_vehicle = None
        for v in self.vehicles:
            if v.type == "rl" and not v.passed:
                agent_vehicle = v
                break # We take the first one found for the single step action
        
        if agent_vehicle and action is not None:
            self._apply_rl_action(agent_vehicle, action)

        # 2. Update background vehicles & non-active RL vehicles
        for v in self.vehicles:
            if v.passed: continue
            if v == agent_vehicle: continue # Already handled action translation to speed/lane
            
            # Simple heuristic for non-active RL agents or background vehicles
            self._update_rule_based_behavior(v)

        # 3. Physics update (Move everyone)
        self._apply_physics()

        # 4. Spawning logic
        if self.spawner.should_spawn(len([v for v in self.vehicles if not v.passed])):
            new_v = self.spawner.spawn(self.current_step)
            # Ensure no overlapping at spawn
            if not any(ov.lane == new_v.lane and abs(ov.x - new_v.x) < 15 for ov in self.vehicles if not ov.passed):
                self.vehicles.append(new_v)

        # 5. Reward & State
        obs = self._get_obs(agent_vehicle)
        reward = self._compute_reward(agent_vehicle) if agent_vehicle else 0.0
        
        terminated = (self.current_step >= self.max_steps)
        if agent_vehicle and agent_vehicle.passed:
            # If agent passed, we could terminate or wait. 
            # In multi-agent shared policy, we usually continue until max steps.
            pass
            
        info = self._get_info()
        
        # Remove passed vehicles from the simulation list to keep it fast
        # (Though we mark 'passed' to keep them in metrics for a step)
        self.vehicles = [v for v in self.vehicles if v.x < self.road_length + 50]

        return obs, reward, terminated, False, info

    def _apply_rl_action(self, v: Vehicle, action: int):
        # 0=Accel (+0.3), 1=Maintain, 2=Decel (-0.3), 3=Merge Left, 4=Merge Right
        if action == 0: v.speed = min(self.MAX_SPEED, v.speed + 0.3)
        elif action == 2: v.speed = max(0, v.speed - 0.3)
        
        if action == 3 and v.lane > 0 and not v.merging:
            # Check availability
            v.target_lane = v.lane - 1
            v.merging = True
            v.merge_progress = 0.0
        elif action == 4 and v.lane < self.n_lanes - 1 and not v.merging:
            v.target_lane = v.lane + 1
            v.merging = True
            v.merge_progress = 0.0

    def _update_rule_based_behavior(self, v: Vehicle):
        leader, dist = self._get_leader(v)
        profile = PROFILES.get(v.type, PROFILES["cooperative"])
        
        # IDM Acceleration
        target_speed = profile["max_speed"]
        acc = idm_acceleration(v.speed, dist, leader.speed if leader else target_speed, v.type)
        v.speed = np.clip(v.speed + acc, self.MIN_SPEED, self.MAX_SPEED)

        # Merging logic
        # After bottleneck, forced lane 1
        if v.x > self.bottleneck_x:
            if v.lane != 1 and not v.merging:
                v.target_lane = 1
                v.merging = True
                v.merge_progress = 0.0
        # If before bottleneck, check profile merge_start_x
        elif v.x > profile.get("merge_start_x", 430):
            if v.lane != 1 and not v.merging:
                v.target_lane = 1
                v.merging = True
                v.merge_progress = 0.0

    def _apply_physics(self):
        for v in self.vehicles:
            if v.passed: continue
            
            # Lane merging progress
            if v.merging:
                v.merge_progress += 0.2 # 5 steps to complete
                if v.merge_progress >= 1.0:
                    v.lane = v.target_lane
                    v.merging = False
                    v.merge_progress = 0.0
            
            # Forward motion
            v.x += v.speed
            v.distance_traveled += v.speed
            
            # Bottleneck collision / constraint logic
            # Past bottleneck, if not in lane 1, you hit the wall?
            # We strictly enforce lane 1 past bottleneck.
            if v.x > self.bottleneck_x and v.lane != 1:
                v.x = self.bottleneck_x # Stuck at wall
                v.speed = 0.0
                v.in_conflict = True
                self.conflict_count += 1
            else:
                v.in_conflict = False

            # Check if passed bottleneck
            if not v.passed and v.x > self.bottleneck_x:
                v.just_passed = True
                v.passed = True
                self.passed_count += 1
            else:
                v.just_passed = False

            # Metrics
            if v.speed < 0.5:
                v.wait_time += self.dt
                self.total_wait += self.dt
                self.wait_count += 1

    def _compute_reward(self, v: Vehicle) -> float:
        # Individual speed reward
        r_individual = v.speed / self.MAX_SPEED
        
        # Conflict penalty
        r_conflict = -2.0 if v.in_conflict else 0.0
        
        # Blocked penalty
        r_blocked = -0.5 if v.speed < 0.2 else 0.0
        
        # Collective: passed count normalized
        passed_this_step = len([ov for ov in self.vehicles if ov.just_passed])
        r_collective = passed_this_step / max(1, len(self.vehicles))
        
        # Bonus for early merging
        r_merge_bonus = 0.0
        if v.x > self.merge_start and v.lane != 1:
            # Penalty proportional to closeness to bottleneck if NOT merged
            dist_to_bottleneck = self.bottleneck_x - v.x
            merge_zone_len = self.bottleneck_x - self.merge_start
            r_merge_bonus = -0.1 * (1 - (dist_to_bottleneck / merge_zone_len))

        if self.reward_mode == "selfish":
            return r_individual + r_conflict + r_blocked
        elif self.reward_mode == "cooperative":
            return r_collective + r_conflict * 0.5 + r_blocked + r_merge_bonus
        else: # mixed
            return (self.alpha * r_individual + (1 - self.alpha) * r_collective + 
                    r_conflict + r_blocked + r_merge_bonus)

    def _get_leader(self, v: Vehicle) -> Tuple[Optional[Vehicle], float]:
        leaders = [ov for ov in self.vehicles if ov.lane == v.lane and ov.x > v.x and not ov.passed]
        if not leaders:
            return None, self.SENSOR_RANGE
        closest = min(leaders, key=lambda ov: ov.x - v.x)
        return closest, (closest.x - v.x)

    def _get_neighbor(self, v: Vehicle, lane: int) -> Tuple[Optional[Vehicle], float]:
        if lane < 0 or lane >= self.n_lanes:
            return None, self.SENSOR_RANGE
        neighbors = [ov for ov in self.vehicles if ov.lane == lane and not ov.passed]
        if not neighbors:
            return None, self.SENSOR_RANGE
        # Closest regardless of ahead/behind for neighbor detection
        closest = min(neighbors, key=lambda ov: abs(ov.x - v.x))
        return closest, abs(closest.x - v.x)

    def _get_info(self) -> Dict:
        return {
            "throughput": self.passed_count / max(1, self.current_step),
            "avg_speed": np.mean([v.speed for v in self.vehicles if not v.passed]) if self.vehicles else 0,
            "conflicts": self.conflict_count,
            "total_passed": self.passed_count,
            "avg_wait": self.total_wait / max(1, self.wait_count)
        }
