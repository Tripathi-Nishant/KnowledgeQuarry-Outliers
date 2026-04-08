import random
import numpy as np
from envs.vehicle import Vehicle
from agents.behavior_profiles import PROFILES

class VehicleSpawner:
    def __init__(self, 
                 arrival_rate=4.0,           # Mean vehicles per second (steps assumed 0.1s?)
                 aggressive_ratio=0.3, 
                 cooperative_ratio=0.5, 
                 rl_ratio=0.2,
                 max_vehicles=40):
        self.arrival_rate = arrival_rate
        self.aggressive_ratio = aggressive_ratio
        self.cooperative_ratio = cooperative_ratio
        self.rl_ratio = rl_ratio
        self.max_vehicles = max_vehicles
        self.id_counter = 0

    def should_spawn(self, current_vehicle_count, dt=0.1) -> bool:
        """
        Poisson process spawning logic.
        dt is the time interval per simulation step.
        """
        if current_vehicle_count >= self.max_vehicles:
            return False
        
        # Prob(spawn in dt) = arrival_rate * dt
        return random.random() < (self.arrival_rate * dt)

    def spawn(self, current_step, road_length=1000, n_lanes=3) -> Vehicle:
        self.id_counter += 1
        
        # Deterministic sampling of types based on ratios
        r = random.random()
        if r < self.aggressive_ratio:
            v_type = "aggressive"
        elif r < self.aggressive_ratio + self.cooperative_ratio:
            v_type = "cooperative"
        else:
            v_type = "rl"
            
        profile = PROFILES.get(v_type, PROFILES["cooperative"])
        
        # Start at x=0, across any of the initial lanes
        lane = random.randint(0, n_lanes - 1)
        
        return Vehicle(
            id=self.id_counter,
            x=0.0,
            lane=lane,
            target_lane=lane,
            speed=profile.get("max_speed", 2.0) * 0.8, # Start at 80% max speed
            type=v_type,
            color=profile.get("color", "#CCCCCC"),
            born_step=current_step
        )
