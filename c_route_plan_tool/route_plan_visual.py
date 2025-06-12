import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches # For 2D rectangles

# Robot dimensions (mirroring route_plan_all.py)
ROBOT_WIDTH = 0.1  # meters, X-axis
ROBOT_DEPTH = 0.1  # meters, Y-axis (front-to-back)

# Configuration variables
POSSIBILITY_ID = 3  # Change this to visualize different route plans
SHOW_OTHER_ACTORS = False # Set to False to hide non-route actors

def ensure_output_directory(script_dir):
    """Create an output directory for visualizations if it doesn't exist."""
    output_dir = os.path.join(script_dir, 'output') # Ensure this is output_visuals
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def get_actor_data(actor_name, df_actors):
    """Retrieve data for a specific actor."""
    actor_data = df_actors[df_actors['ActorName'] == actor_name]
    if actor_data.empty:
        print(f"Warning: Actor '{actor_name}' not found in actors data.")
        return None
    return actor_data.iloc[0]

def get_actor_bounds(actor):
    """Calculate the 2D bounding box coordinates for an actor."""
    x_min = actor['WorldX'] - actor['WorldSizeX'] / 2
    x_max = actor['WorldX'] + actor['WorldSizeX'] / 2
    y_min = actor['WorldY'] - actor['WorldSizeY'] / 2
    y_max = actor['WorldY'] + actor['WorldSizeY'] / 2
    # z_min = actor['WorldZ'] - actor['WorldSizeZ'] / 2 # Removed Z
    # z_max = actor['WorldZ'] + actor['WorldSizeZ'] / 2 # Removed Z
    return [x_min, x_max, y_min, y_max]

def plot_2d_rect(ax, bounds, color, label, alpha=0.1, edge_alpha=0.8, linewidth=1):
    """Plot a 2D rectangle with translucent face and solid edges."""
    x_min, x_max, y_min, y_max = bounds
    width = x_max - x_min
    height = y_max - y_min
    rect = patches.Rectangle((x_min, y_min), width, height, 
                             linewidth=linewidth, edgecolor=color, 
                             facecolor=color, alpha=alpha, label=label)
    ax.add_patch(rect)

    # Plot edges with different alpha if needed (optional, could simplify)
    ax.plot([x_min, x_max], [y_min, y_min], color=color, alpha=edge_alpha, linewidth=linewidth) # Bottom
    ax.plot([x_min, x_max], [y_max, y_max], color=color, alpha=edge_alpha, linewidth=linewidth) # Top
    ax.plot([x_min, x_min], [y_min, y_max], color=color, alpha=edge_alpha, linewidth=linewidth) # Left
    ax.plot([x_max, x_max], [y_min, y_max], color=color, alpha=edge_alpha, linewidth=linewidth) # Right


def get_obb_corners_2d(robot_center_2d, robot_orientation_rad, robot_width, robot_depth):
    """
    Calculates the 2D coordinates of the four corners of the robot's Oriented Bounding Box (OBB).
    Mirrors the logic from route_plan_all.py for consistency.
    Args:
        robot_center_2d (np.array): The 2D center of the robot.
        robot_orientation_rad (float): The robot's orientation angle in radians
                                      (angle of its front/depth axis).
        robot_width (float): The robot's width.
        robot_depth (float): The robot's depth (front-to-back).
    Returns:
        list: A list of 4 np.array, each representing a 2D corner of the OBB.
    """
    half_w = robot_width / 2
    half_d = robot_depth / 2
    local_corners = [
        np.array([half_w, half_d]),    # Front-right relative to robot's local frame
        np.array([-half_w, half_d]),   # Front-left
        np.array([-half_w, -half_d]),  # Back-left
        np.array([half_w, -half_d])    # Back-right
    ]
    robot_y_axis = np.array([np.cos(robot_orientation_rad), np.sin(robot_orientation_rad)])
    robot_x_axis = np.array([-robot_y_axis[1], robot_y_axis[0]])
    world_corners = []
    for corner in local_corners:
        world_corner = robot_center_2d + corner[0] * robot_x_axis + corner[1] * robot_y_axis
        world_corners.append(world_corner)
    return world_corners

def plot_robot_obb(ax, center_2d, orientation_rad, robot_width, robot_depth, color, label, alpha=0.3, edge_alpha=0.9, linewidth=1.5):
    """Plots the robot's OBB as a 2D polygon."""
    corners = get_obb_corners_2d(center_2d, orientation_rad, robot_width, robot_depth)
    polygon = patches.Polygon(corners, closed=True, 
                              linewidth=linewidth, edgecolor=color, 
                              facecolor=color, alpha=alpha, label=label)
    ax.add_patch(polygon)


def calculate_stand_next_to_target_2d(robot_center_2d, target_actor_data, robot_width, robot_depth):
    """
    Calculates the 2D position and orientation for the robot to stand "next to" a target actor.
    The robot is positioned adjacent to one of the target's AABB faces (in 2D XY plane),
    with its front (depth axis) oriented towards the target's center.
    Mirrors the logic from route_plan_all.py for consistency.

    Args:
        robot_center_2d (np.array): The current 2D center of the robot (e.g., from actor it's starting at).
                                      This is used to determine which face of the target is closest to this initial point.
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
    if target_actor_data is None:
        # Fallback if target_actor_data is None (e.g. actor not found)
        # Place robot at original point, default orientation
        return robot_center_2d, 0.0

    target_center_2d = np.array([target_actor_data['WorldX'], target_actor_data['WorldY']])
    target_half_size_x = target_actor_data['WorldSizeX'] / 2
    target_half_size_y = target_actor_data['WorldSizeY'] / 2

    faces_properties = [
        (target_half_size_x, 0, 1, 0),  # Right face (+X)
        (-target_half_size_x, 0, -1, 0), # Left face (-X)
        (0, target_half_size_y, 0, 1),   # Top face (+Y)
        (0, -target_half_size_y, 0, -1)  # Bottom face (-Y)
    ]

    min_dist_sq = float('inf')
    best_position = None
    best_orientation_rad = None

    # The robot_center_2d is the point from which we are trying to reach the target's face.
    # For example, if this is for BeginAt, robot_center_2d is the center of BeginAt actor.
    # We want to find which face of target_actor_data is closest to robot_center_2d
    # and then place the robot next to that face.

    for off_x, off_y, norm_x, norm_y in faces_properties:
        # This is the center of the target's face
        face_center_on_target_aabb = target_center_2d + np.array([off_x, off_y])
        outward_normal = np.array([norm_x, norm_y])
        
        # Position the robot's center such that its edge is touching the target's face.
        # The robot's front (depth axis) should point towards the target.
        # So, we offset from the face_center_on_target_aabb by half the robot's depth along the outward_normal.
        robot_pos_candidate = face_center_on_target_aabb + outward_normal * (robot_depth / 2)
        
        # Calculate distance from the original robot_center_2d to this candidate position
        # This helps select the 'closest' face of the target actor to stand next to,
        # relative to where the robot is conceptually starting (e.g. center of BeginAt actor)
        dist_sq = np.sum((robot_pos_candidate - robot_center_2d)**2)
        
        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq
            best_position = robot_pos_candidate
            # Orientation: robot's Y-axis (front) points opposite to the outward_normal
            # (i.e., towards the target's center from this face)
            best_orientation_rad = np.arctan2(-outward_normal[1], -outward_normal[0])
            
    if best_position is None:
        direction_to_target = target_center_2d - robot_center_2d
        if np.linalg.norm(direction_to_target) < 1e-6:
            orientation_rad = 0 
        else:
            orientation_rad = np.arctan2(direction_to_target[1], direction_to_target[0])
        return robot_center_2d, orientation_rad

    return best_position, best_orientation_rad


def visualize_route_plan(plan_data, df_actors, possibility_id, output_dir):
    """Visualize a single route plan in 2D."""
    fig = plt.figure(figsize=(19, 10))
    ax = fig.add_subplot(111)

    begin_at_actor = get_actor_data(plan_data['BeginAt_Name'], df_actors)
    facing_at_actor = get_actor_data(plan_data['FacingAt_Name'], df_actors)
    go_to_actor = get_actor_data(plan_data['GoTo_Name'], df_actors)
    end_at_actor = get_actor_data(plan_data['EndAt_Name'], df_actors)

    if not all([begin_at_actor is not None, facing_at_actor is not None, go_to_actor is not None, end_at_actor is not None]):
        print(f"Could not visualize possibility {possibility_id} due to missing actor data.")
        plt.close(fig)
        return

    # Original actor centers (used as reference for calculate_stand_next_to_target_2d)
    begin_at_center_2d = np.array([begin_at_actor['WorldX'], begin_at_actor['WorldY']])
    go_to_center_2d = np.array([go_to_actor['WorldX'], go_to_actor['WorldY']])
    end_at_center_2d = np.array([end_at_actor['WorldX'], end_at_actor['WorldY']])
    facing_at_center_2d = np.array([facing_at_actor['WorldX'], facing_at_actor['WorldY']])

    # Calculate robot's actual positions and orientations
    # 1. Robot at BeginAt, oriented towards FacingAt
    robot_begin_pos_2d, _ = calculate_stand_next_to_target_2d(begin_at_center_2d, begin_at_actor, ROBOT_WIDTH, ROBOT_DEPTH)
    # Initial orientation: robot's front (Y-axis) points from robot_begin_pos_2d to facing_at_center_2d
    direction_to_facing = facing_at_center_2d - robot_begin_pos_2d
    if np.linalg.norm(direction_to_facing) < 1e-6:
        robot_initial_orientation_rad = 0 # Default if coincident
    else:
        robot_initial_orientation_rad = np.arctan2(direction_to_facing[1], direction_to_facing[0])

    # 2. Robot at GoTo
    # The robot stands next to the GoTo actor. Its orientation is determined by calculate_stand_next_to_target_2d,
    # meaning its front faces the GoTo actor's center from the edge it's positioned at.
    robot_goto_pos_2d, robot_goto_orientation_rad = calculate_stand_next_to_target_2d(go_to_center_2d, go_to_actor, ROBOT_WIDTH, ROBOT_DEPTH)

    # 3. Robot at EndAt
    # Similar to GoTo, robot stands next to EndAt actor, facing its center.
    robot_end_pos_2d, robot_end_orientation_rad = calculate_stand_next_to_target_2d(end_at_center_2d, end_at_actor, ROBOT_WIDTH, ROBOT_DEPTH)

    actors_in_plan_visual_info = {
        plan_data['BeginAt_Name']: (begin_at_actor, 'blue', 'Begin At (Actor)'),
        plan_data['FacingAt_Name']: (facing_at_actor, 'cyan', 'Facing At (Actor)'),
        plan_data['GoTo_Name']: (go_to_actor, 'orange', 'Go To (Actor)'),
        plan_data['EndAt_Name']: (end_at_actor, 'red', 'End At (Actor)')
    }

    all_plot_bounds = []

    # Plot key ACTORS in the plan (their AABBs)
    for name, (actor_data, color, label_suffix) in actors_in_plan_visual_info.items():
        bounds = get_actor_bounds(actor_data)
        all_plot_bounds.append(bounds)
        plot_2d_rect(ax, bounds, color, f"{actor_data['ShortActorName']}\n({label_suffix})", alpha=0.2, edge_alpha=0.9, linewidth=1.5)
        ax.text(actor_data['WorldX'], actor_data['WorldY'] + (bounds[3]-bounds[2])/2 + 0.1, 
                f"{actor_data['ShortActorName']}\n({label_suffix})", color=color, 
                ha='center', va='bottom', fontsize=8, fontweight='bold')

    # Plot other actors for context if enabled
    if SHOW_OTHER_ACTORS:
        for _, other_actor_row in df_actors.iterrows():
            if other_actor_row['ActorName'] not in actors_in_plan_visual_info:
                other_bounds = get_actor_bounds(other_actor_row)
                all_plot_bounds.append(other_bounds)
                plot_2d_rect(ax, other_bounds, 'gray', f"{other_actor_row['ShortActorName']}\n(Context)", alpha=0.05, edge_alpha=0.3)
    
    # Set plot limits based on all actors plotted
    if all_plot_bounds:
        min_coords = np.min([[b[0], b[2]] for b in all_plot_bounds], axis=0)
        max_coords = np.max([[b[1], b[3]] for b in all_plot_bounds], axis=0)
        
        data_range_x = max_coords[0] - min_coords[0]
        data_range_y = max_coords[1] - min_coords[1]
        padding = max(data_range_x, data_range_y) * 0.15 # Increased padding slightly
        if padding == 0: padding = 1.0

        ax.set_xlim(min_coords[0] - padding, max_coords[0] + padding)
        ax.set_ylim(min_coords[1] - padding, max_coords[1] + padding)
    else:
        ax.set_xlim(-5, 5); ax.set_ylim(-5, 5)

    # Plot Robot OBBs at key positions
    plot_robot_obb(ax, robot_begin_pos_2d, robot_initial_orientation_rad, ROBOT_WIDTH, ROBOT_DEPTH, 'green', 'Robot @ Begin', alpha=0.5)
    plot_robot_obb(ax, robot_goto_pos_2d, robot_goto_orientation_rad, ROBOT_WIDTH, ROBOT_DEPTH, 'darkorange', 'Robot @ GoTo', alpha=0.5)
    plot_robot_obb(ax, robot_end_pos_2d, robot_end_orientation_rad, ROBOT_WIDTH, ROBOT_DEPTH, 'darkred', 'Robot @ End', alpha=0.5)

    # Plot path segments between ROBOT positions
    ax.plot([robot_begin_pos_2d[0], robot_goto_pos_2d[0]], [robot_begin_pos_2d[1], robot_goto_pos_2d[1]],
            'g--o', linewidth=2, markersize=5, label='Path: Robot Begin -> Robot GoTo')
    ax.plot([robot_goto_pos_2d[0], robot_end_pos_2d[0]], [robot_goto_pos_2d[1], robot_end_pos_2d[1]],
            'm--o', linewidth=2, markersize=5, label='Path: Robot GoTo -> Robot EndAt')

    # Annotate turn actions (positions might need adjustment based on robot OBBs)
    action1_text = f"1. {plan_data['AnswerAction1']}"
    # Place annotation near the robot's begin position
    ax.text(robot_begin_pos_2d[0], robot_begin_pos_2d[1] + 0.2, action1_text, color='black', 
            fontsize=9, bbox=dict(facecolor='yellow', alpha=0.5, pad=0.2))

    action2_text = f"3. {plan_data['AnswerAction2']}"
    # Place annotation near the robot's goto position
    ax.text(robot_goto_pos_2d[0], robot_goto_pos_2d[1] + 0.2, action2_text, color='black', 
            fontsize=9, bbox=dict(facecolor='yellow', alpha=0.5, pad=0.2))

    ax.set_xlabel('World X (m)')
    ax.set_ylabel('World Y (m)')
    ax.set_title(f"Route Plan Visualization (Possibility ID: {possibility_id})\n"
                 f"{plan_data['BeginAt_Display']} (Robot Start) -> {plan_data['GoTo_Display']} (Robot Mid) -> {plan_data['EndAt_Display']} (Robot End)", fontsize=10)
    ax.legend(fontsize=7) # Adjusted legend font size
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()

    filename = f'route_plan_visual.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300)
    print(f"Visualization saved to {filepath}")
    plt.show() # Comment out if running in a headless environment or for batch processing
    plt.close(fig)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir) # Assumes c_route_plan_tool is one level down
    output_visual_dir = ensure_output_directory(script_dir)

    actors_csv_path = os.path.join(project_root, '0_data_cleanup_tool', 'output', 'ranked_unique_actor_anno.csv')
    route_plans_csv_path = os.path.join(script_dir, 'output', 'route_plan_all.csv')

    try:
        df_actors = pd.read_csv(actors_csv_path)
        df_route_plans = pd.read_csv(route_plans_csv_path)
    except FileNotFoundError as e:
        print(f"Error loading CSV files: {e}")
        return

    if df_route_plans.empty:
        print("Route plan CSV is empty. No data to visualize.")
        return

    # Get data for the specified possibility ID
    plan_to_visualize = df_route_plans[df_route_plans['Possibility'] == POSSIBILITY_ID]

    if plan_to_visualize.empty:
        print(f"No data found for Possibility ID: {POSSIBILITY_ID}. Max ID is {df_route_plans['Possibility'].max()}")
        return
    
    visualize_route_plan(plan_to_visualize.iloc[0], df_actors, POSSIBILITY_ID, output_visual_dir)

if __name__ == "__main__":
    main()