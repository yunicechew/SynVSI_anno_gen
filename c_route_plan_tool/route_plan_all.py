import pandas as pd
import numpy as np
import os
import json
from itertools import product

# Robot dimensions
ROBOT_WIDTH = 0.1  # meters, X-axis
ROBOT_DEPTH = 0.1  # meters, Y-axis (front-to-back)

def ensure_output_directory():
    """Create output directory if it doesn't exist and return its path."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def load_csv_data(project_root):
    """Loads all necessary CSV files."""
    actors_df_path = os.path.join(project_root, '0_data_cleanup_tool', 'output', 'ranked_unique_actor_anno.csv')
    abs_dist_df_path = os.path.join(project_root, 'm_absolute_distance_tool', 'output', 'absolute_distances_all.csv')
    rel_dir_df_path = os.path.join(project_root, 'c_relative_direction_tool', 'output', 'relative_direction_all.csv')

    if not os.path.exists(actors_df_path):
        raise FileNotFoundError(f"Actor data CSV not found: {actors_df_path}")
    if not os.path.exists(abs_dist_df_path):
        raise FileNotFoundError(f"Absolute distance CSV not found: {abs_dist_df_path}")
    if not os.path.exists(rel_dir_df_path):
        raise FileNotFoundError(f"Relative direction CSV not found: {rel_dir_df_path}")

    actors_df = pd.read_csv(actors_df_path)
    abs_dist_df = pd.read_csv(abs_dist_df_path)
    rel_dir_df = pd.read_csv(rel_dir_df_path)
    
    # Create a dictionary for faster distance lookups
    dist_dict = {}
    for _, row in abs_dist_df.iterrows():
        dist_dict[(row['Actor1'], row['Actor2'])] = row['Answer']
        dist_dict[(row['Actor2'], row['Actor1'])] = row['Answer']
        
    return actors_df, abs_dist_df, rel_dir_df, dist_dict

def get_actor_details(actor_name, actors_df):
    """Gets 3D coordinates, 2D coordinates (XY), display name, and full data for an actor."""
    actor_row_df = actors_df[actors_df['ActorName'] == actor_name]
    if actor_row_df.empty:
        # print(f"Warning: Actor {actor_name} not found in actors_df.")
        return None, None, None, None # Added None for actor_data
    actor_data = actor_row_df.iloc[0]
    coords_3d = np.array([actor_data['WorldX'], actor_data['WorldY'], actor_data['WorldZ']])
    coords_2d = coords_3d[:2] # XY plane for existing 2D logic
    display_name = actor_data.get('ActorDescription')
    if pd.isna(display_name) or str(display_name).strip() == "":
        display_name = actor_data['ShortActorName']
    return coords_3d, coords_2d, display_name, actor_data # Return actor_data

def calculate_stand_next_to_target_2d(robot_center_2d, target_actor_data, robot_width, robot_depth):
    """
    Calculates the 2D position and orientation for the robot to stand "next to" a target actor.
    The robot is positioned adjacent to one of the target's AABB faces (in 2D XY plane),
    with its front (depth axis) oriented towards the target's center.

    Args:
        robot_center_2d (np.array): The current 2D center of the robot (e.g., begin_at, go_to, end_at point).
                                      This is used to determine which face of the target is closest.
        target_actor_data (pd.Series): Data for the target actor, including WorldX/Y and WorldSizeX/Y.
        robot_width (float): The width of the robot (X-axis dimension).
        robot_depth (float): The depth of the robot (Y-axis, front-to-back dimension).

    Returns:
        tuple: (new_robot_position_2d, robot_orientation_angle_rad)
               - new_robot_position_2d (np.array): The adjusted 2D center position of the robot.
               - robot_orientation_angle_rad (float): The robot's orientation angle in radians,
                                                      where 0 rad is along the positive X-axis.
                                                      The robot's Y-axis (front) will point towards the target.
    """
    target_center_2d = np.array([target_actor_data['WorldX'], target_actor_data['WorldY']])
    target_half_size_x = target_actor_data['WorldSizeX'] / 2
    target_half_size_y = target_actor_data['WorldSizeY'] / 2

    # Define the four faces of the target's 2D AABB and their outward normals
    # (face_center_offset_x, face_center_offset_y, normal_x, normal_y)
    faces_properties = [
        (target_half_size_x, 0, 1, 0),  # Right face (+X)
        (-target_half_size_x, 0, -1, 0), # Left face (-X)
        (0, target_half_size_y, 0, 1),   # Top face (+Y)
        (0, -target_half_size_y, 0, -1)  # Bottom face (-Y)
    ]

    min_dist_sq = float('inf')
    best_position = None
    best_orientation_rad = None

    for off_x, off_y, norm_x, norm_y in faces_properties:
        face_center_2d = target_center_2d + np.array([off_x, off_y])
        outward_normal = np.array([norm_x, norm_y])
        
        # Position the robot's center such that its edge is touching the target's face.
        # The robot's front (depth axis) should point towards the target.
        # So, we offset by half the robot's depth along the *negative* normal (towards the target).
        robot_pos_candidate = face_center_2d + outward_normal * (robot_depth / 2)
        
        dist_sq = np.sum((robot_pos_candidate - robot_center_2d)**2)
        
        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq
            best_position = robot_pos_candidate
            # Orientation: robot's Y-axis (front) points opposite to the outward_normal
            # (i.e., towards the target's center from this face)
            # Angle of the vector (-normal_x, -normal_y) relative to positive X-axis
            best_orientation_rad = np.arctan2(-outward_normal[1], -outward_normal[0])
            
    # If for some reason no best position is found (e.g., target has zero size, though unlikely here)
    if best_position is None:
        # Fallback: place robot at original point, facing towards target center
        direction_to_target = target_center_2d - robot_center_2d
        if np.linalg.norm(direction_to_target) < 1e-6: # if robot_center is target_center
            orientation_rad = 0 # Default orientation
        else:
            orientation_rad = np.arctan2(direction_to_target[1], direction_to_target[0])
        return robot_center_2d, orientation_rad

    return best_position, best_orientation_rad

def calculate_relative_info(origin_coords_2d, facing_target_coords_2d, locate_coords_2d):
    """
    Calculates relative coordinates and angle of locate_coords_2d 
    with respect to origin_coords_2d facing facing_target_coords_2d.
    Input coordinates are 2D.
    Returns: dict with 'x_coord', 'y_coord', 'angle'
    """
    if np.array_equal(origin_coords_2d, facing_target_coords_2d):
        # print(f"Warning: Origin and facing_target are the same: {origin_coords}")
        return {'x_coord': 0, 'y_coord': 0, 'angle': 0, 'valid': False}

    y_axis_vec = facing_target_coords_2d - origin_coords_2d
    norm_y_axis_vec = np.linalg.norm(y_axis_vec)
    if norm_y_axis_vec == 0: # Should be caught by above check, but as safeguard
        # print(f"Warning: Norm of y_axis_vec is zero between {origin_coords} and {facing_target_coords}")
        return {'x_coord': 0, 'y_coord': 0, 'angle': 0, 'valid': False}
    
    y_axis = y_axis_vec / norm_y_axis_vec
    x_axis = np.array([y_axis[1], -y_axis[0]]) # Perpendicular, to the right

    locate_at_rel_to_origin = locate_coords_2d - origin_coords_2d
    
    x_coord = np.dot(locate_at_rel_to_origin, x_axis)
    y_coord = np.dot(locate_at_rel_to_origin, y_axis)
    
    angle = np.degrees(np.arctan2(x_coord, y_coord))
    
    return {'x_coord': x_coord, 'y_coord': y_coord, 'angle': angle, 'valid': True}

def determine_turn_type(angle_degrees):
    """Determines turn type based on angle, referencing medium difficulty logic
       from relative_direction_all.py.
       Angle is in degrees, from -180 to 180.
       - Back: angle > 135 or angle < -135
       - Left: -135 <= angle < 0
       - Right: 0 <= angle <= 135
    """
    if angle_degrees > 135 or angle_degrees < -135:
        return "turn back"
    elif angle_degrees >= -135 and angle_degrees < 0:
        return "turn left"  # Corrected from 'turn right' in original logic for this range
    elif angle_degrees >= 0 and angle_degrees <= 135:
        return "turn right" # Corrected from 'turn left' in original logic for this range
    else:
        # This case should ideally not be reached if angle_degrees is within [-180, 180]
        # and the above conditions are exhaustive for that range.
        # However, as a fallback, or if angles can be outside this, treat as 'turn back'.
        # print(f"Warning: Unexpected angle {angle_degrees} in determine_turn_type. Defaulting to 'turn back'.")
        return "turn back"

def calculate_turn_angle_at_intermediate(prev_pt_coords_2d, current_pt_coords_2d, next_pt_coords_2d):
    """
    Calculates the angle for a turn at current_pt_coords.
    The robot arrived from prev_pt_coords and is heading to next_pt_coords.
    Input coordinates are 2D.
    """
    if np.array_equal(current_pt_coords_2d, prev_pt_coords_2d):
        # print(f"Warning: current_pt_coords and prev_pt_coords are the same: {current_pt_coords_2d}")
        return 0, False # Invalid turn

    vector_arrival = current_pt_coords_2d - prev_pt_coords_2d
    norm_vector_arrival = np.linalg.norm(vector_arrival)
    if norm_vector_arrival == 0: # Should be caught by above
        return 0, False

    # Current facing direction (robot's positive Y-axis)
    y_axis_robot = vector_arrival / norm_vector_arrival
    # Robot's right (positive X-axis)
    x_axis_robot = np.array([y_axis_robot[1], -y_axis_robot[0]])
    
    vector_to_next = next_pt_coords_2d - current_pt_coords_2d
    if np.linalg.norm(vector_to_next) == 0: # Current and next are same point
        # print(f"Warning: current_pt_coords and next_pt_coords are the same: {current_pt_coords_2d}")
        return 0, False # Invalid turn, effectively no movement

    x_rel_to_robot = np.dot(vector_to_next, x_axis_robot)
    y_rel_to_robot = np.dot(vector_to_next, y_axis_robot)
    
    angle = np.degrees(np.arctan2(x_rel_to_robot, y_rel_to_robot))
    return angle, True

def is_segment_intersecting_aabb(p1_3d, p2_3d, obstacle_actor_data, epsilon=1e-6):
    """
    Checks if a 3D line segment (p1_3d to p2_3d) intersects with the AABB of obstacle_actor_data.
    p1_3d, p2_3d: 3D numpy arrays (start and end of segment).
    obstacle_actor_data: pandas Series for the obstacle actor, containing WorldX/Y/Z and WorldSizeX/Y/Z.
    Returns True if intersection occurs, False otherwise.
    """
    obs_center = np.array([obstacle_actor_data['WorldX'], obstacle_actor_data['WorldY'], obstacle_actor_data['WorldZ']])
    obs_half_sizes = np.array([obstacle_actor_data['WorldSizeX']/2, obstacle_actor_data['WorldSizeY']/2, obstacle_actor_data['WorldSizeZ']/2])
    
    min_b = obs_center - obs_half_sizes
    max_b = obs_center + obs_half_sizes
    
    direction = p2_3d - p1_3d
    
    # Check if segment start or end point is inside the AABB
    # This handles cases where one endpoint is inside, which the slab test might miss if not careful with t ranges.
    p1_inside = np.all(p1_3d >= min_b) and np.all(p1_3d <= max_b)
    p2_inside = np.all(p2_3d >= min_b) and np.all(p2_3d <= max_b)
    if p1_inside or p2_inside:
        return True

    t_min_overall = 0.0
    t_max_overall = 1.0
    
    for i in range(3): # Iterate over X, Y, Z axes
        if abs(direction[i]) < epsilon: # Segment is parallel to slab planes for this axis
            # If parallel and p1 is outside the slab, no intersection with this slab
            if p1_3d[i] < min_b[i] or p1_3d[i] > max_b[i]:
                return False 
        else:
            t1 = (min_b[i] - p1_3d[i]) / direction[i]
            t2 = (max_b[i] - p1_3d[i]) / direction[i]
            
            if t1 > t2:
                t1, t2 = t2, t1 # Ensure t1 is entry, t2 is exit for the infinite line
            
            t_min_overall = max(t_min_overall, t1)
            t_max_overall = min(t_max_overall, t2)
            
            # If the intersection interval becomes invalid, no intersection
            if t_min_overall >= t_max_overall: # Using >= to be robust
                return False
                
    # If t_min_overall < t_max_overall, the segment intersects the AABB.
    # The conditions t_min_overall < 1 and t_max_overall > 0 are implicitly handled
    # by initializing t_min_overall=0, t_max_overall=1 and the max/min updates.
    return True

def get_swept_robot_radius():
    """Calculates the radius of a circle that encloses the robot's 2D footprint."""
    return max(ROBOT_WIDTH, ROBOT_DEPTH) / 2

def is_swept_robot_colliding_with_aabb(segment_start_2d, segment_end_2d, potential_obstacle_row, swept_robot_rad):
    # Note: I noticed the signature in the traceback for main was different, ensure this matches your latest definition
    # Original in main: is_swept_robot_colliding_with_aabb(robot_begin_pos_2d, robot_goto_pos_2d, potential_obstacle_row, swept_robot_rad)
    # Make sure the parameters align with how it's called.
    """
    Checks if a 2D circular robot footprint moving from segment_start_2d to segment_end_2d (capsule shape)
    intersects with the 2D AABB of an obstacle.

    Args:
        segment_start_2d (np.array): Start 2D center of the robot's circular footprint.
        segment_end_2d (np.array): End 2D center of the robot's circular footprint.
        potential_obstacle_row (pd.Series): Data for the obstacle actor (WorldX/Y, WorldSizeX/Y).
        swept_robot_rad (float): Radius of the robot's circular footprint.

    Returns:
        bool: True if collision, False otherwise.
    """
    # Corrected: Use potential_obstacle_row instead of undefined obstacle_actor_data
    obs_center_2d = np.array([potential_obstacle_row['WorldX'], potential_obstacle_row['WorldY']])
    obs_half_sizes_2d = np.array([potential_obstacle_row['WorldSizeX']/2, potential_obstacle_row['WorldSizeY']/2])
    
    obs_min_2d = obs_center_2d - obs_half_sizes_2d
    obs_max_2d = obs_center_2d + obs_half_sizes_2d

    # 1. Check collision with the expanded AABB (Minkowski sum of AABB and circle)
    # Expand the AABB by the swept_robot_rad
    expanded_min_2d = obs_min_2d - swept_robot_rad # Corrected: use swept_robot_rad parameter
    expanded_max_2d = obs_max_2d + swept_robot_rad # Corrected: use swept_robot_rad parameter

    # Line segment (segment_start_2d, segment_end_2d) vs expanded AABB intersection (Simplified Slab Test)
    # Corrected: Use segment_start_2d and segment_end_2d parameters
    segment_vector = segment_end_2d - segment_start_2d 
    t_min_overall = 0.0
    t_max_overall = 1.0

    for i in range(2): # X and Y axes
        if abs(segment_vector[i]) < 1e-6: # Segment parallel to slab planes
            if segment_start_2d[i] < expanded_min_2d[i] or segment_start_2d[i] > expanded_max_2d[i]:
                return False # Parallel and outside expanded slab
        else:
            t1 = (expanded_min_2d[i] - segment_start_2d[i]) / segment_vector[i]
            t2 = (expanded_max_2d[i] - segment_start_2d[i]) / segment_vector[i]
            
            if t1 > t2: t1, t2 = t2, t1
            
            t_min_overall = max(t_min_overall, t1)
            t_max_overall = min(t_max_overall, t2)
            
            if t_min_overall >= t_max_overall: # Using >= for robustness
                return False
    
    # If the segment intersects the expanded AABB, a collision is possible.
    return True

def get_obb_corners_2d(robot_center_2d, robot_orientation_rad, robot_width, robot_depth):
    """
    Calculates the 2D coordinates of the four corners of the robot's Oriented Bounding Box (OBB).

    Args:
        robot_center_2d (np.array): The 2D center of the robot.
        robot_orientation_rad (float): The robot's orientation angle in radians.
                                      (Angle of the robot's local Y-axis/depth axis relative to world X-axis).
        robot_width (float): The robot's width (local X-axis).
        robot_depth (float): The robot's depth (local Y-axis, front-to-back).

    Returns:
        list: A list of 4 np.array, each representing a 2D corner of the OBB.
    """
    # Half dimensions
    half_w = robot_width / 2
    half_d = robot_depth / 2

    # Local corner coordinates (assuming robot's Y is forward, X is right)
    # Order: front-right, front-left, back-left, back-right
    # If robot's front is along its depth axis (Y), then corners relative to center are:
    # (width/2, depth/2), (-width/2, depth/2), (-width/2, -depth/2), (width/2, -depth/2)
    local_corners = [
        np.array([half_w, half_d]),    # Front-right relative to robot's local frame
        np.array([-half_w, half_d]),   # Front-left
        np.array([-half_w, -half_d]),  # Back-left
        np.array([half_w, -half_d])    # Back-right
    ]

    # Rotation matrix for the given orientation
    # Note: If orientation_rad is angle of Y-axis (depth), then X-axis is Y-axis rotated by -PI/2
    # Or, more directly, use standard 2D rotation matrix where theta is orientation of local X-axis.
    # If robot_orientation_rad is the angle of the robot's *front* (depth/Y axis):
    cos_theta = np.cos(robot_orientation_rad)
    sin_theta = np.sin(robot_orientation_rad)

    # The robot's local X-axis (width) is perpendicular to its local Y-axis (depth/front)
    # If Y_robot = (cos_theta, sin_theta), then X_robot = (sin_theta, -cos_theta) or (-sin_theta, cos_theta)
    # Let's assume standard rotation: orientation_rad is angle of robot's +X axis.
    # If calculate_stand_next_to_target_2d orients Y-axis (depth) towards target,
    # then robot_orientation_rad is angle of that Y-axis.
    # The robot's local X-axis (width) would be robot_orientation_rad - PI/2.
    # Let's stick to the definition from calculate_stand_next_to_target_2d where
    # robot_orientation_rad is the angle of the robot's front (Y-axis).
    # So, the robot's local X-axis is oriented at robot_orientation_rad - pi/2
    # And its local Y-axis is oriented at robot_orientation_rad

    # Basis vectors for the robot's local coordinate system
    # Robot's local Y-axis (direction of depth)
    robot_y_axis = np.array([np.cos(robot_orientation_rad), np.sin(robot_orientation_rad)])
    # Robot's local X-axis (direction of width), perpendicular to Y-axis
    robot_x_axis = np.array([-robot_y_axis[1], robot_y_axis[0]]) # Rotated -90 deg from Y-axis

    world_corners = []
    for corner in local_corners:
        # corner[0] is along robot's local X (width), corner[1] is along robot's local Y (depth)
        world_corner = robot_center_2d + corner[0] * robot_x_axis + corner[1] * robot_y_axis
        world_corners.append(world_corner)

    return world_corners

def project_shape_onto_axis(corners, axis):
    """Projects a shape (defined by its corners) onto a given axis.

    Args:
        corners (list of np.array): List of 2D corner coordinates of the shape.
        axis (np.array): The 2D axis (unit vector) to project onto.

    Returns:
        tuple: (min_projection, max_projection) of the shape on the axis.
    """
    min_proj = np.dot(corners[0], axis)
    max_proj = min_proj
    for i in range(1, len(corners)):
        projection = np.dot(corners[i], axis)
        if projection < min_proj:
            min_proj = projection
        if projection > max_proj:
            max_proj = projection
    return min_proj, max_proj

def is_robot_obb_colliding_with_aabb(robot_center_2d, robot_orientation_rad, robot_width, robot_depth, \
                                     obstacle_aabb_center_2d, obstacle_aabb_size_2d):
    """
    Checks if the robot's 2D Oriented Bounding Box (OBB) intersects with the 2D AABB of an obstacle
    using the Separating Axis Theorem (SAT).

    Args:
        robot_center_2d (np.array): The 2D center of the robot.
        robot_orientation_rad (float): The robot's orientation angle in radians (angle of its front/depth axis).
        robot_width (float): The robot's width.
        robot_depth (float): The robot's depth (front-to-back).
        obstacle_aabb_center_2d (np.array): The 2D center of the obstacle's AABB.
        obstacle_aabb_size_2d (np.array): The full 2D size (width, height) of the obstacle's AABB.

    Returns:
        bool: True if collision, False otherwise.
    """
    # 1. Get OBB corners
    obb_corners = get_obb_corners_2d(robot_center_2d, robot_orientation_rad, robot_width, robot_depth)

    # 2. Get AABB corners
    obs_half_size = obstacle_aabb_size_2d / 2.0
    aabb_corners = [
        obstacle_aabb_center_2d + np.array([-obs_half_size[0], -obs_half_size[1]]),
        obstacle_aabb_center_2d + np.array([ obs_half_size[0], -obs_half_size[1]]),
        obstacle_aabb_center_2d + np.array([ obs_half_size[0],  obs_half_size[1]]),
        obstacle_aabb_center_2d + np.array([-obs_half_size[0],  obs_half_size[1]])
    ]

    # 3. Define axes to test (normals of OBB and AABB)
    axes = []
    # AABB axes (world X and Y)
    axes.append(np.array([1, 0]))
    axes.append(np.array([0, 1]))

    # OBB axes
    # Robot's local Y-axis (direction of depth/front)
    obb_axis1 = np.array([np.cos(robot_orientation_rad), np.sin(robot_orientation_rad)])
    # Robot's local X-axis (direction of width)
    obb_axis2 = np.array([-obb_axis1[1], obb_axis1[0]])
    axes.append(obb_axis1)
    axes.append(obb_axis2)

    # 4. Project shapes onto each axis and check for separation
    for axis in axes:
        # Normalize axis (important for correct projection interpretation, though SAT works with non-normalized too)
        # However, project_shape_onto_axis assumes a unit vector if we interpret min/max as distances.
        # For SAT, just needs to be consistent. Let's normalize for clarity.
        axis_normalized = axis / np.linalg.norm(axis)
        if np.linalg.norm(axis) < 1e-6: # Avoid division by zero for zero vector (should not happen for valid axes)
            continue
            
        min_obb, max_obb = project_shape_onto_axis(obb_corners, axis_normalized)
        min_aabb, max_aabb = project_shape_onto_axis(aabb_corners, axis_normalized)

        # Check for non-overlap (a separating axis is found)
        if max_obb < min_aabb or max_aabb < min_obb:
            return False  # Separating axis found, no collision

    return True  # No separating axis found after checking all axes, collision detected


# Main logic function
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir) # Goes up to SynVSI_anno_gen
    output_dir = ensure_output_directory()

    try:
        actors_df, abs_dist_df, rel_dir_df, dist_dict = load_csv_data(project_root)
    except FileNotFoundError as e:
        print(e)
        return

    all_results = []
    possibility_counter = 1
    
    processed_begin_end_pairs = set()

    for _, abs_row in abs_dist_df.iterrows():
        if abs_row['Answer'] <= 2:
            continue

        # Consider both (Actor1, Actor2) and (Actor2, Actor1) as (begin_at, end_at)
        for begin_at_name_cand, end_at_name_cand in [(abs_row['Actor1'], abs_row['Actor2']), (abs_row['Actor2'], abs_row['Actor1'])]:
            
            current_begin_at_name = begin_at_name_cand
            current_end_at_name = end_at_name_cand

            begin_coords_3d, begin_coords_2d, begin_display, begin_actor_data = get_actor_details(current_begin_at_name, actors_df)
            end_coords_3d, end_coords_2d, end_display, end_actor_data = get_actor_details(current_end_at_name, actors_df)

            if begin_coords_3d is None or end_coords_3d is None: # Check 3D coords
                continue

            # Calculate robot's actual standing position and orientation next to begin_at actor
            # For the initial position, the robot stands next to begin_at. Its orientation will be refined later based on facing_at.
            robot_begin_pos_2d, _ = calculate_stand_next_to_target_2d(begin_coords_2d, begin_actor_data, ROBOT_WIDTH, ROBOT_DEPTH)
            # robot_initial_orientation_rad will be determined by facing_at

            dist_begin_end = dist_dict.get((current_begin_at_name, current_end_at_name), float('inf'))

            # Find suitable facing_at actors
            # Facing_at must be different from begin_at and end_at
            possible_facing_ats = rel_dir_df[
                (rel_dir_df['standing_at'] == current_begin_at_name) &
                (rel_dir_df['facing_at'] != current_begin_at_name) &
                (rel_dir_df['facing_at'] != current_end_at_name)
            ]['facing_at'].unique()

            for current_facing_at_name in possible_facing_ats:
                facing_coords_3d, facing_coords_2d, facing_display, facing_actor_data = get_actor_details(current_facing_at_name, actors_df)
                if facing_coords_3d is None: # Check 3D coords
                    continue

                # Determine the robot's initial orientation at robot_begin_pos_2d, facing towards facing_coords_2d
                vec_to_facing = facing_coords_2d - robot_begin_pos_2d
                if np.linalg.norm(vec_to_facing) < 1e-6:
                    robot_initial_orientation_rad = 0 
                else:
                    robot_initial_orientation_rad = np.arctan2(vec_to_facing[1], vec_to_facing[0])

                # Static collision check for robot at begin_pos
                static_collision_at_begin = False
                for _, obs_row in actors_df.iterrows():
                    obs_name = obs_row['ActorName']
                    if obs_name == current_begin_at_name: # Don't check collision with the actor it's next to
                        continue
                    obs_center_3d, _, _, obs_data = get_actor_details(obs_name, actors_df)
                    if obs_center_3d is None or obs_data is None: continue
                    
                    obstacle_aabb_center_2d = np.array([obs_data['WorldX'], obs_data['WorldY']])
                    obstacle_aabb_size_2d = np.array([obs_data['WorldSizeX'], obs_data['WorldSizeY']])

                    if is_robot_obb_colliding_with_aabb(robot_begin_pos_2d, robot_initial_orientation_rad, 
                                                        ROBOT_WIDTH, ROBOT_DEPTH, 
                                                        obstacle_aabb_center_2d, obstacle_aabb_size_2d):
                        static_collision_at_begin = True
                        break
                if static_collision_at_begin:
                    continue # Skip this facing_at if robot collides at its start position

                rel_info_end_for_goto_select = calculate_relative_info(robot_begin_pos_2d, facing_coords_2d, end_coords_2d)
                if not rel_info_end_for_goto_select['valid']: continue
                end_at_rel_x_sign = np.sign(rel_info_end_for_goto_select['x_coord'])


                # Find suitable go_to actors
                potential_go_to_names = []
                for actor_name_potential_goto in actors_df['ActorName'].unique():
                    if actor_name_potential_goto in [current_begin_at_name, current_facing_at_name, current_end_at_name]:
                        continue

                    potential_goto_coords_3d, potential_goto_coords_2d, _, _ = get_actor_details(actor_name_potential_goto, actors_df)
                    if potential_goto_coords_3d is None: continue # Check 3D coords
                    
                    # Use 2D coordinates for relative info
                    rel_info_potential_goto = calculate_relative_info(begin_coords_2d, facing_coords_2d, potential_goto_coords_2d)
                    if not rel_info_potential_goto['valid']: continue
                    
                    # If end_at is on the axis (sign=0), any side for go_to is fine. Otherwise, signs must match.
                    # Or, more simply, go_to should not be on the opposite side of end_at unless end_at is on the axis.
                    # A simpler filter: go_to should be generally in front
                    if rel_info_potential_goto['y_coord'] < 0.1: # Must be somewhat in front
                        continue

                    dist_begin_goto = dist_dict.get((current_begin_at_name, actor_name_potential_goto), float('inf'))

                    if dist_begin_goto < dist_begin_end:
                        # Check if (begin_at, facing_at, actor_name_potential_goto) is a valid combo in rel_dir_df
                        # This means there's a relative direction question defined for this setup.
                        if not rel_dir_df[
                            (rel_dir_df['standing_at'] == current_begin_at_name) &
                            (rel_dir_df['facing_at'] == current_facing_at_name) &
                            (rel_dir_df['locate_at'] == actor_name_potential_goto)
                        ].empty:
                            potential_go_to_names.append(actor_name_potential_goto)
                
                for current_go_to_name in potential_go_to_names:
                    go_to_coords_3d, go_to_coords_2d, go_to_display, go_to_actor_data = get_actor_details(current_go_to_name, actors_df)
                    if go_to_coords_3d is None: continue # Check 3D coords

                    robot_goto_pos_2d, robot_orientation_at_goto_arrival_rad = calculate_stand_next_to_target_2d(go_to_coords_2d, go_to_actor_data, ROBOT_WIDTH, ROBOT_DEPTH)
                    # The orientation at goto is determined by arrival from robot_begin_pos_2d
                    # For static check, we need an orientation. Let's assume it faces the goto_actor for now.
                    # This might need refinement if a specific post-arrival orientation is required before turning.
                    # For now, use the orientation from calculate_stand_next_to_target_2d, which faces the target.
                    # However, a more accurate orientation for static check at goto would be its arrival orientation.
                    vec_arrival_at_goto = robot_goto_pos_2d - robot_begin_pos_2d
                    if np.linalg.norm(vec_arrival_at_goto) > 1e-6:
                        robot_orientation_at_goto_static_check_rad = np.arctan2(vec_arrival_at_goto[1], vec_arrival_at_goto[0])
                    else: # Should not happen if begin and goto are distinct
                        robot_orientation_at_goto_static_check_rad = robot_orientation_at_goto_arrival_rad # Fallback to facing target

                    # Static collision check for robot at goto_pos
                    static_collision_at_goto = False
                    for _, obs_row in actors_df.iterrows():
                        obs_name = obs_row['ActorName']
                        if obs_name == current_go_to_name: 
                            continue
                        obs_center_3d, _, _, obs_data = get_actor_details(obs_name, actors_df)
                        if obs_center_3d is None or obs_data is None: continue
                        obstacle_aabb_center_2d = np.array([obs_data['WorldX'], obs_data['WorldY']])
                        obstacle_aabb_size_2d = np.array([obs_data['WorldSizeX'], obs_data['WorldSizeY']])
                        if is_robot_obb_colliding_with_aabb(robot_goto_pos_2d, robot_orientation_at_goto_static_check_rad, 
                                                            ROBOT_WIDTH, ROBOT_DEPTH, 
                                                            obstacle_aabb_center_2d, obstacle_aabb_size_2d):
                            static_collision_at_goto = True
                            break
                    if static_collision_at_goto:
                        continue # Skip this go_to if robot collides

                    robot_end_pos_2d, robot_orientation_at_end_arrival_rad = calculate_stand_next_to_target_2d(end_coords_2d, end_actor_data, ROBOT_WIDTH, ROBOT_DEPTH)
                    # Similar to goto, orientation for static check at end is based on arrival from goto.
                    vec_arrival_at_end = robot_end_pos_2d - robot_goto_pos_2d
                    if np.linalg.norm(vec_arrival_at_end) > 1e-6:
                        robot_orientation_at_end_static_check_rad = np.arctan2(vec_arrival_at_end[1], vec_arrival_at_end[0])
                    else: # Should not happen if goto and end are distinct
                        robot_orientation_at_end_static_check_rad = robot_orientation_at_end_arrival_rad # Fallback

                    # Static collision check for robot at end_pos
                    static_collision_at_end = False
                    for _, obs_row in actors_df.iterrows():
                        obs_name = obs_row['ActorName']
                        if obs_name == current_end_at_name: 
                            continue
                        obs_center_3d, _, _, obs_data = get_actor_details(obs_name, actors_df)
                        if obs_center_3d is None or obs_data is None: continue
                        obstacle_aabb_center_2d = np.array([obs_data['WorldX'], obs_data['WorldY']])
                        obstacle_aabb_size_2d = np.array([obs_data['WorldSizeX'], obs_data['WorldSizeY']])
                        if is_robot_obb_colliding_with_aabb(robot_end_pos_2d, robot_orientation_at_end_static_check_rad, 
                                                            ROBOT_WIDTH, ROBOT_DEPTH, 
                                                            obstacle_aabb_center_2d, obstacle_aabb_size_2d):
                            static_collision_at_end = True
                            break
                    if static_collision_at_end:
                        continue # Skip this go_to if robot collides at end position

                    # Obstacle Check - using robot's actual path points (3D for AABB check)
                    # We need 3D points for the robot's path for the existing AABB check.
                    # For simplicity, we'll use the robot's 2D 'next to' positions and keep original Z.
                    robot_begin_path_3d = np.array([robot_begin_pos_2d[0], robot_begin_pos_2d[1], begin_coords_3d[2]])
                    robot_goto_path_3d = np.array([robot_goto_pos_2d[0], robot_goto_pos_2d[1], go_to_coords_3d[2]])
                    robot_end_path_3d = np.array([robot_end_pos_2d[0], robot_end_pos_2d[1], end_coords_3d[2]])

                    path_is_obstructed = False
                    actors_in_route = {current_begin_at_name, current_go_to_name, current_end_at_name}
                    swept_robot_rad = get_swept_robot_radius()
                    
                    for _, potential_obstacle_row in actors_df.iterrows():
                        potential_obstacle_name = potential_obstacle_row['ActorName']
                        if potential_obstacle_name in actors_in_route:
                            continue

                        # Check segment: robot_begin_pos_2d -> robot_goto_pos_2d
                        if is_swept_robot_colliding_with_aabb(robot_begin_pos_2d, robot_goto_pos_2d, potential_obstacle_row, swept_robot_rad):
                            path_is_obstructed = True
                            break
                        
                        # Check segment: robot_goto_pos_2d -> robot_end_pos_2d
                        if is_swept_robot_colliding_with_aabb(robot_goto_pos_2d, robot_end_pos_2d, potential_obstacle_row, swept_robot_rad):
                            path_is_obstructed = True
                            break
                    
                    if path_is_obstructed:
                        continue # This go_to_name leads to an obstructed path, try the next one

                    # Determine Turn 1 (from robot_begin_pos_2d, oriented by robot_initial_orientation_rad, towards robot_goto_pos_2d)
                    # calculate_relative_info needs origin, a point defining the Y-axis, and the target point.
                    # The robot's Y-axis is defined by robot_initial_orientation_rad.
                    # We can construct a facing_point based on this orientation.
                    facing_point_for_turn1 = robot_begin_pos_2d + np.array([np.cos(robot_initial_orientation_rad), np.sin(robot_initial_orientation_rad)])
                    turn1_rel_info = calculate_relative_info(robot_begin_pos_2d, facing_point_for_turn1, robot_goto_pos_2d)
                    if not turn1_rel_info['valid']: continue
                    action1 = determine_turn_type(turn1_rel_info['angle'])

                    # Determine Turn 2 (at robot_goto_pos_2d, arrived from robot_begin_pos_2d, towards robot_end_pos_2d)
                    turn2_angle, turn2_valid = calculate_turn_angle_at_intermediate(robot_begin_pos_2d, robot_goto_pos_2d, robot_end_pos_2d)
                    if not turn2_valid: continue
                    action2 = determine_turn_type(turn2_angle)
                    
                    # Ensure actions are not None if determine_turn_type could return None
                    if action1 is None or action2 is None: continue


                    question = (
                        f"You are a robot beginning at the {begin_display} facing the {facing_display}. "
                        f"You want to navigate to the {end_display}. "
                        "You will perform the following actions (Note: for each [please fill in], choose either 'turn back,' 'turn left,' or 'turn right.'): "
                        f"1. [please fill in] 2. Go forward until the {go_to_display} 3. [please fill in] 4. Go forward until the {end_display}. "
                        "You have reached the final destination."
                    )
                    
                    # Check for duplicate questions based on all display names and actions
                    # This is a simple way to reduce redundancy if actor names map to same display names
                    question_key = (begin_display, facing_display, go_to_display, end_display, action1, action2)
                    if question_key in processed_begin_end_pairs : # Reusing this set for a different purpose here
                        continue
                    processed_begin_end_pairs.add(question_key)


                    all_results.append({
                        'Possibility': possibility_counter,
                        'BeginAt_Name': current_begin_at_name,
                        'FacingAt_Name': current_facing_at_name,
                        'GoTo_Name': current_go_to_name,
                        'EndAt_Name': current_end_at_name,
                        'BeginAt_Display': begin_display,
                        'FacingAt_Display': facing_display,
                        'GoTo_Display': go_to_display,
                        'EndAt_Display': end_display,
                        'Question': question,
                        'AnswerAction1': action1,
                        'AnswerAction2': action2 # Renamed from AnswerAction3 for clarity
                    })
                    possibility_counter += 1
                    if possibility_counter % 100 == 0:
                        print(f"Generated {possibility_counter-1} possibilities...")


    if all_results:
        output_df = pd.DataFrame(all_results)
        output_csv_path = os.path.join(output_dir, 'route_plan_all.csv')
        output_df.to_csv(output_csv_path, index=False)
        print(f"Successfully generated {len(all_results)} route plan questions to {output_csv_path}")
    else:
        print("No route plan questions generated.")

# Script execution
if __name__ == "__main__":
    main()