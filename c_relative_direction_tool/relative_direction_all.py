import pandas as pd
import numpy as np
import os

def determine_quadrant(df, standing_at_name, facing_at_name, locate_at_name):
    """
    Calculate relative direction between three points in 2D space.
    Args:
        df: DataFrame containing actor positions
        standing_at_name: Name of standing position actor
        facing_at_name: Name of facing direction actor
        locate_at_name: Name of object to locate     
    Returns:
        Dictionary containing direction information in three difficulty levels
        and coordinate system transformation details
    """
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
        quadrant_hard = "front-right"
        quadrant_num = "quadrant I"
        quadrant_easy = "right"
    elif x_coord < 0 and y_coord >= 0:
        quadrant_hard = "front-left"
        quadrant_num = "quadrant II"
        quadrant_easy = "left"
    elif x_coord < 0 and y_coord < 0:
        quadrant_hard = "back-left"
        quadrant_num = "quadrant III"
        quadrant_easy = "left"
    else:  # x_coord >= 0 and y_coord < 0
        quadrant_hard = "back-right"
        quadrant_num = "quadrant IV"
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
        'quadrant_num': quadrant_num,
        'quadrant_medium': quadrant_medium,
        'quadrant_easy': quadrant_easy,
        'new_coords': (x_coord, y_coord),
        'standing_at': standing_at,
        'facing_at': facing_at,
        'locate_at': locate_at,
        'x_axis': x_axis,
        'y_axis': y_axis
    }

def ensure_output_directory():
    # Create main output directory
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir
def main():
    """
    Main processing function that:
    1. Loads actor position data
    2. Generates all valid combinations of standing/facing/locate points
    3. Calculates relative directions for each combination
    4. Outputs results to CSV with questions and answers at three difficulty levels
    """
    # Determine the project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir) # Assumes this script is in a subdirectory of the project root

    # Load actor positions from CSV using a relative path
    csv_path = os.path.join(project_root, '0_data_cleanup_tool', 'output', 'ranked_unique_actor_anno.csv')
    df = pd.read_csv(csv_path)
    actor_names = df['ActorName'].unique()
    output_dir = ensure_output_directory()
    
    # Process all valid combinations of three different actors
    all_results = []
    possibility_counter = 1
    
    for standing_at in actor_names:
        for facing_at in actor_names:
            for locate_at in actor_names:
                if standing_at != facing_at and standing_at != locate_at and facing_at != locate_at:
                    try:
                        result = determine_quadrant(df, standing_at, facing_at, locate_at)
                        
                        # Get cleaned names from the DataFrame
                        clean_standing = df[df['ActorName'] == standing_at]['ShortActorName'].iloc[0]
                        clean_facing = df[df['ActorName'] == facing_at]['ShortActorName'].iloc[0]
                        clean_locate = df[df['ActorName'] == locate_at]['ShortActorName'].iloc[0]
                        
                        hard_question = f"""If I am standing by the {clean_standing} and facing the {clean_facing}, is the {clean_locate} to my front-left, front-right, back-left, or back-right? The directions refer to the quadrants of a Cartesian plane (if I am standing at the origin and facing along the positive y-axis)."""
                        
                        medium_question = f"""If I am standing by the {clean_standing} and facing the {clean_facing}, is the {clean_locate} to my left, right, or back? An object is to my back if I would have to turn at least 135 degrees in order to face it."""
                        
                        easy_question = f"""If I am standing by the {clean_standing} and facing the {clean_facing}, is the {clean_locate} to the left or the right of the stove?"""
                        
                        # Combine all data into one row
                        all_results.append({
                            'Possibility': possibility_counter,
                            'standing_at': standing_at,
                            'standing_at_x': result['standing_at'][0],
                            'standing_at_y': result['standing_at'][1],
                            'facing_at': facing_at,
                            'facing_at_x': result['facing_at'][0],
                            'facing_at_y': result['facing_at'][1],
                            'locate_at': locate_at,
                            'locate_at_x': result['locate_at'][0],
                            'locate_at_y': result['locate_at'][1],
                            'QuadrantNumber': result['quadrant_num'],
                            'QuestionHard': hard_question,
                            'AnswerHard': result['quadrant'],
                            'QuestionMedium': medium_question,
                            'AnswerMedium': result['quadrant_medium'],
                            'QuestionEasy': easy_question,
                            'AnswerEasy': result['quadrant_easy']
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