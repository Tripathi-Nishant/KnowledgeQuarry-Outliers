PROFILES = {
    "cooperative": {
        "max_speed": 2.8,
        "merge_start_x": 430,      # Merges at zone start
        "following_distance": 32,
        "yields_to_merger": True,
        "acceleration": 0.3,
        "deceleration": 0.5,
        "color": "#378ADD",        # Blue
    },
    "aggressive": {
        "max_speed": 3.8,
        "merge_start_x": 560,      # Merges at last moment
        "following_distance": 14,
        "yields_to_merger": False,
        "acceleration": 0.5,
        "deceleration": 0.8,
        "color": "#E24B4A",        # Red
    },
    "rl": {
        "max_speed": 4.0,          # Learns its own limits
        "acceleration": 0.4,       # Default for background RL agents
        "deceleration": 0.6,
        "color": "#639922",        # Green
    }
}

def idm_acceleration(v_speed, leader_dist, leader_speed, profile_type="cooperative"):
    """
    IDM (Intelligent Driver Model) lite for rule-based movement.
    """
    profile = PROFILES.get(profile_type, PROFILES["cooperative"])
    v0 = profile["max_speed"]      # Desired speed
    T = 1.5                        # Safe time headway (seconds)
    a = profile["acceleration"]    # Max acceleration
    b = profile["deceleration"]    # Comfortable deceleration
    s0 = 8                         # Minimum gap (units)

    # Leader distance is already adjusted for vehicle lengths in environment
    if leader_dist < s0:
        return -b * 2.0  # Emergency brake

    # Standard IDM formula:
    # s* = s0 + vT + v(deltaV) / (2 * sqrt(ab))
    dv = v_speed - leader_speed
    s_star = s0 + max(0, v_speed * T + (v_speed * dv) / (2 * (a * b)**0.5))
    
    # acc = a * (1 - (v/v0)^4 - (s*/s)^2)
    term1 = (v_speed / max(0.1, v0))**4
    term2 = (s_star / max(0.1, leader_dist))**2
    acc = a * (1 - term1 - term2)
    
    return float(acc)
