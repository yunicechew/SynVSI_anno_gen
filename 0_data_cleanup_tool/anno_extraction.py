import pandas as pd
import os

# Configuration variables
<<<<<<< HEAD
INPUT_CSV = "/Users/bytedance/Desktop/SynVSI_anno_gen/0_data_cleanup_tool/output/frame_extract_meta.csv"
=======
script_dir = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(script_dir, "output", "frame_extract_meta.csv")

>>>>>>> temp-branch
MIN_FRAME_COUNT = 10  # Minimum number of frames an actor must appear in
MIN_VOLUME = 0.005    # Minimum volume in cubic meters

def clean_actor_name(name):
    """
    Extract simplified name from actor identifier string, keeping everything after last '_To_'
    and replacing underscores with spaces.
    Example: 
    - 'BP_To_sink5_2_SM_To_sink_door_L' -> 'sink door L'
    - 'SM_To_chair1_To_chair2' -> 'chair2'
    Args: 
        name: Full actor name string
    Returns: 
        Simplified name with underscores replaced by spaces
    """
    # Split on last occurrence of '_To_'
    parts = name.rsplit('_To_', 1)
    if len(parts) > 1:
        # Take everything after last '_To_' and replace underscores with spaces
        return parts[1].replace('_', ' ')
    return name

def extract_ranked_actor_info(min_frame_count=MIN_FRAME_COUNT, min_volume=MIN_VOLUME):
    """
    Extract unique actor information with first appearance frame and world data
    Args: 
        min_frame_count: Minimum number of frames an actor must appear in
        min_volume: Minimum volume in cubic meters
    """
    # Read the input CSV file using absolute path
    df = pd.read_csv(INPUT_CSV)

    # Count frame appearances for each actor
    frame_counts = df.groupby(['ActorName', 'ActorClass'])['FrameNumber'].nunique().reset_index()
    frame_counts = frame_counts.rename(columns={'FrameNumber': 'FrameCount'})
    
    # Filter actors that appear in more than min_frame_count frames
    qualified_actors = frame_counts[frame_counts['FrameCount'] >= min_frame_count]

    # Get first appearance frame for qualified actors
    first_appearance = df[df['ActorName'].isin(qualified_actors['ActorName'])]
    first_appearance = first_appearance.groupby(['ActorName', 'ActorClass'])['FrameNumber'].min().reset_index()
    first_appearance = first_appearance.rename(columns={'FrameNumber': 'FirstFrame'})

    # Get unique actor information for qualified actors
    selected_columns = ['ActorName', 'ActorClass', 'WorldX', 'WorldY', 'WorldZ', 
                       'WorldSizeX', 'WorldSizeY', 'WorldSizeZ']
    actor_info = df[df['ActorName'].isin(qualified_actors['ActorName'])][selected_columns].drop_duplicates()

    # Convert world coordinates and sizes from centimeters to meters
    world_columns = ['WorldX', 'WorldY', 'WorldZ', 'WorldSizeX', 'WorldSizeY', 'WorldSizeZ']
    actor_info[world_columns] = actor_info[world_columns] / 100.0

    # Calculate volume in cubic meters
    actor_info['Volume'] = actor_info['WorldSizeX'] * actor_info['WorldSizeY'] * actor_info['WorldSizeZ']
    
    # Filter by minimum volume
    actor_info = actor_info[actor_info['Volume'] >= min_volume]

    # Clean actor names
    actor_info['CleanedActorName'] = actor_info['ActorName'].apply(clean_actor_name)

    # Merge the information and sort by first appearance
    merged_info = pd.merge(first_appearance, actor_info, on=['ActorName', 'ActorClass'])
    # Add frame count information
    merged_info = pd.merge(merged_info, qualified_actors[['ActorName', 'FrameCount']], on='ActorName')
    merged_info = merged_info.sort_values('FirstFrame')
    
    # Ensure column order with FirstFrame, FrameCount and Volume
    column_order = ['FirstFrame', 'FrameCount', 'ActorName', 'CleanedActorName', 'ActorClass', 
                    'WorldX', 'WorldY', 'WorldZ', 'WorldSizeX', 'WorldSizeY', 'WorldSizeZ', 'Volume']
    merged_info = merged_info[column_order]

    # Create output directory if it doesn't exist
<<<<<<< HEAD
    output_dir = "0_data_cleanup_tool/output"
=======
    output_dir = os.path.join(script_dir, "output") # New path relative to this script
>>>>>>> temp-branch
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save to output CSV
    output_csv = os.path.join(output_dir, "ranked_unique_actor_anno.csv")
    merged_info.to_csv(output_csv, index=False)

    print(f"Ranked actor information has been saved to {output_csv}")
    print(f"Only included actors that:")
    print(f"1. Appear in {min_frame_count} or more frames")
    print(f"2. Have a volume greater than {min_volume} cubic meters")
    print("Note: All world coordinates and sizes have been converted from centimeters to meters")

if __name__ == "__main__":
    extract_ranked_actor_info()  # Using default values from configuration variables