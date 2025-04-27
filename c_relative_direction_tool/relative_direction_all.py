import pandas as pd
import numpy as np
import os

# Configuration
ENABLE_VISUALIZATION = False  # Toggle to control visualization generation

def determine_quadrant(df, standing_at_name, facing_at_name, locate_at_name):
    # Get world coordinates for each point from the DataFrame
    standing_at = df[df['ActorName'] == standing_at_name][['WorldX', 'WorldY']].values[0]
    facing_at = df[df['ActorName'] == facing_at_name][['WorldX', 'WorldY']].values[0]
    locate_at = df[df['ActorName'] == locate_at_name][['WorldX', 'WorldY']].values[0]
    
    # Create a relative coordinate system where:
    # - Origin is at standing_at point
    # - Y-axis points from standing_at to facing_at (normalized)
    y_axis = facing_at - standing_at
    y_axis = y_axis / np.linalg.norm(y_axis)
    
    # X-axis is perpendicular to y_axis (rotate 90 degrees clockwise)
    # This ensures x-axis is to the right when facing along positive y-axis
    x_axis = np.array([y_axis[1], -y_axis[0]])
    
    # Convert locate_at point to relative coordinates
    locate_at_rel = locate_at - standing_at
    
    # Project locate_at onto the new coordinate axes
    x_coord = np.dot(locate_at_rel, x_axis)  # Right (+) or Left (-)
    y_coord = np.dot(locate_at_rel, y_axis)  # Front (+) or Back (-)
    
    # Calculate angle between locate_at and Y-axis for direction determination
    angle = np.degrees(np.arctan2(x_coord, y_coord))
    
    # Determine direction in three different granularities:
    # 1. Hard: Precise quadrant with front/back + left/right
    # 2. Medium: Uses 135° sectors for back, otherwise left/right
    # 3. Easy: Simple left/right determination
    
    # Hard version (quadrant-based)
    if x_coord >= 0 and y_coord >= 0:
        quadrant_hard = "front-right (quadrant I)"
        quadrant_easy = "right"
    elif x_coord < 0 and y_coord >= 0:
        quadrant_hard = "front-left (quadrant II)"
        quadrant_easy = "left"
    elif x_coord < 0 and y_coord < 0:
        quadrant_hard = "back-left (quadrant III)"
        quadrant_easy = "left"
    else:  # x_coord >= 0 and y_coord < 0
        quadrant_hard = "back-right (quadrant IV)"
        quadrant_easy = "right"
    
    # Medium version (135° sectors)
    if angle > 135 or angle < -135:  # Back sector
        quadrant_medium = "back"
    elif angle >= -135 and angle < 0:  # Left sector
        quadrant_medium = "left"
    else:  # Right sector (angle >= 0 and angle <= 135)
        quadrant_medium = "right"
    
    return {
        'quadrant': quadrant_hard,
        'quadrant_medium': quadrant_medium,
        'quadrant_easy': quadrant_easy,
        'new_coords': (x_coord, y_coord),  # Coordinates in relative system
        'standing_at': standing_at,        # World coordinates
        'facing_at': facing_at,           # World coordinates
        'locate_at': locate_at,           # World coordinates
        'x_axis': x_axis,                 # Unit vector
        'y_axis': y_axis                  # Unit vector
    }

def ensure_output_directory():
    # Create main output directory
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def main():
    # Load actor positions from CSV
    df = pd.read_csv('/Users/bytedance/Desktop/SynVSI_anno_gen/anno_cleanup_tool/output/ranked_unique_actor_anno.csv')
    actor_names = df['ActorName'].unique()
    output_dir = ensure_output_directory()
    
    # Process all valid combinations of three different actors
    all_results = []
    possibility_counter = 1
    
    for standing_at in actor_names:
        for facing_at in actor_names:
            for locate_at in actor_names:
                # Skip invalid combinations where points are the same
                if standing_at != facing_at and standing_at != locate_at and facing_at != locate_at:
                    try:
                        # Calculate relative directions
                        result = determine_quadrant(df, standing_at, facing_at, locate_at)
                        
                        # Store actor names for current combination
                        points = {
                            'standing_at': standing_at,  # Origin point
                            'facing_at': facing_at,      # Defines forward direction
                            'locate_at': locate_at       # Target point to describe
                        }
                        
                        # Record results for each point in the combination
                        for point_type, actor_name in points.items():
                            all_results.append({
                                'Possibility': possibility_counter,  # Unique ID for this combination
                                'ActorName': actor_name,
                                'WorldX': result[point_type][0],
                                'WorldY': result[point_type][1],
                                'PointType': point_type,
                                'RelativeDirectionHard': result['quadrant'] if point_type == 'locate_at' else None,
                                'RelativeDirectionMedium': result['quadrant_medium'] if point_type == 'locate_at' else None,
                                'RelativeDirectionEasy': result['quadrant_easy'] if point_type == 'locate_at' else None
                            })
                        
                        possibility_counter += 1
                        
                    except Exception as e:
                        print(f"Error processing combination {possibility_counter}: {standing_at}, {facing_at}, {locate_at}")
                        print(f"Error message: {str(e)}")
                        continue
    
    # Save all results to CSV
    if all_results:
        output_df = pd.DataFrame(all_results)
        output_df.to_csv(os.path.join(output_dir, 'relative_direction_all.csv'), index=False)
        print(f"Successfully processed {len(all_results)//3} combinations")

if __name__ == "__main__":
    main()