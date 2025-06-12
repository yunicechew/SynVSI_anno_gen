import pandas as pd
import os
import argparse
import re # Added import

# Configuration variables
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)  # Navigate up to SynVSI_anno_gen

# Default values
DEFAULT_DATA_SUBDIRECTORY = "20250527-145925"
MIN_FRAME_COUNT = 5  # Minimum number of frames an actor must appear in
MIN_VOLUME = 0.005   # Minimum volume in cubic meters

# Input/Output paths
INPUT_DATA_ROOT = os.path.join(project_root, "0_original_ue_anno")  # Path to original UE anno directory
OUTPUT_DATA_ROOT = os.path.join(project_root, "0_data_cleanup_tool", "output")  # Keep output in cleanup tool

# Removed clean_actor_name function

# --- Add the new function here ---
def _determine_short_actor_name(actor_name: str, cleaned_actor_name: str) -> str:
    """Determines a more short short actor name for VLM prompting."""
    short_actor_name = actor_name  # Default
    parts = actor_name.split('_')
    if len(parts) > 3:
        raw_name_parts = []
        for i in range(2, len(parts)):
            if parts[i].endswith('v') and parts[i][:-1].isdigit():
                break
            raw_name_parts.append(parts[i])
        if raw_name_parts:
            joined_name = "".join(raw_name_parts)
            short_actor_name = re.sub(r'(?<!^)(?=[A-Z])', ' ', joined_name).lower()
    elif len(parts) == 3:
        if parts[1].lower() == 'to':
            raw_name = parts[2]
            short_actor_name = re.sub(r'(?<!^)(?=[A-Z])', ' ', raw_name).lower()
        else:
            raw_name = parts[2]
            short_actor_name = re.sub(r'(?<!^)(?=[A-Z])', ' ', raw_name).lower()
    else:
        # Use cleaned_actor_name if provided and different, otherwise actor_name
        # For this script, cleaned_actor_name will be the same as actor_name
        short_actor_name = cleaned_actor_name.lower() if cleaned_actor_name else actor_name.lower()
    return short_actor_name
# --- End of new function ---

def extract_ranked_actor_info(data_subdir=DEFAULT_DATA_SUBDIRECTORY, min_frame_count=MIN_FRAME_COUNT, min_volume=MIN_VOLUME):
    """
    Extract unique actor information with first appearance frame and world data
    Args:
        data_subdir: Data subdirectory name containing the input data
        min_frame_count: Minimum number of frames an actor must appear in
        min_volume: Minimum volume in cubic meters
    """
    # Construct input path using the data subdirectory
    input_csv = os.path.join(INPUT_DATA_ROOT, data_subdir, "Screenshot_summary.csv")

    if not os.path.exists(input_csv):
        print(f"Error: Input CSV file not found at {input_csv}")
        return

    # Read the input CSV file
    df = pd.read_csv(input_csv)

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
    # Apply transformation: X -> -X for WorldX
    actor_info['WorldX'] = - (actor_info['WorldX'] / 100.0)
    # actor_info['WorldX'] = (actor_info['WorldX'] / 100.0)
    actor_info['WorldY'] = actor_info['WorldY'] / 100.0
    actor_info['WorldZ'] = actor_info['WorldZ'] / 100.0
    
    size_columns = ['WorldSizeX', 'WorldSizeY', 'WorldSizeZ']
    actor_info[size_columns] = actor_info[size_columns] / 100.0

    # Calculate volume in cubic meters
    actor_info['Volume'] = actor_info['WorldSizeX'] * actor_info['WorldSizeY'] * actor_info['WorldSizeZ']
    
    # Filter by minimum volume
    actor_info = actor_info[actor_info['Volume'] >= min_volume]

    # Merge the information and sort by first appearance
    merged_info = pd.merge(first_appearance, actor_info, on=['ActorName', 'ActorClass'])
    # Add frame count information
    merged_info = pd.merge(merged_info, qualified_actors[['ActorName', 'FrameCount']], on='ActorName')
    
    # Generate ShortActorName
    # Pass actor_name for both arguments as CleanedActorName is not separately generated here.
    merged_info['ShortActorName'] = merged_info.apply(
        lambda row: _determine_short_actor_name(row['ActorName'], row['ActorName']), axis=1
    )

    # Extract camera information for each actor's first appearance frame
    camera_columns = ['CamX', 'CamY', 'CamZ', 'CamPitch', 'CamYaw', 'CamRoll']
    
    # Create a dictionary to store camera data for each actor
    camera_data = {}
    
    # For each actor, get the camera data from its first appearance frame
    for _, row in merged_info.iterrows():
        actor_name = row['ActorName']
        first_frame = row['FirstFrame']
        
        # Get the camera data from the first frame this actor appears in
        frame_data = df[(df['ActorName'] == actor_name) & (df['FrameNumber'] == first_frame)]
        
        if not frame_data.empty and all(col in frame_data.columns for col in camera_columns):
            camera_row = frame_data.iloc[0]
            camera_data[actor_name] = {
                'CamX': - (camera_row['CamX'] / 100.0),  # Convert to meters and apply transformation X -> -X
                'CamY': camera_row['CamY'] / 100.0,   # Convert to meters
                'CamZ': camera_row['CamZ'] / 100.0,   # Convert to meters
                'CamPitch': camera_row['CamPitch'],
                'CamYaw': camera_row['CamYaw'],
                'CamRoll': camera_row['CamRoll']
            }
        else:
            # If camera data is missing, use NaN values
            camera_data[actor_name] = {col: float('nan') for col in camera_columns}
    
    # Add camera data to the merged info dataframe
    for col in camera_columns:
        merged_info[col] = merged_info['ActorName'].apply(lambda x: camera_data[x][col])
    
    # Sort by first appearance
    merged_info = merged_info.sort_values('FirstFrame')
    
    # Ensure column order with FirstFrame, FrameCount, Volume and Camera data
    column_order = ['FirstFrame', 'FrameCount', 'ActorName', 'ActorClass', 'ShortActorName',
                    'WorldX', 'WorldY', 'WorldZ', 'WorldSizeX', 'WorldSizeY', 'WorldSizeZ', 'Volume',
                    'CamX', 'CamY', 'CamZ', 'CamPitch', 'CamYaw', 'CamRoll']
    merged_info = merged_info[column_order]

    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DATA_ROOT):
        os.makedirs(OUTPUT_DATA_ROOT)

    # Save to output CSV in the cleanup tool's output directory
    output_csv = os.path.join(OUTPUT_DATA_ROOT, "ranked_unique_actor_anno.csv")
    merged_info.to_csv(output_csv, index=False)

    # print(f"Ranked actor information has been saved to {output_csv}")
    print(f"Unique actors:")
    print(f"1. Appear in {min_frame_count} or more frames")
    print(f"2. Have a volume greater than {min_volume} cubic meters")
    # print("Note: All world coordinates and sizes have been converted from centimeters to meters")
    # print("Note: Camera position data (CamX, CamY, CamZ) has also been converted to meters")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract and rank actor information from Screenshot_summary.csv')
    parser.add_argument('--data_subdir', type=str, default=DEFAULT_DATA_SUBDIRECTORY,
                      help='Data subdirectory name containing the input data')
    parser.add_argument('--min_frames', type=int, default=MIN_FRAME_COUNT,
                      help='Minimum number of frames an actor must appear in')
    parser.add_argument('--min_volume', type=float, default=MIN_VOLUME,
                      help='Minimum volume in cubic meters')
    
    args = parser.parse_args()
    
    extract_ranked_actor_info(
        data_subdir=args.data_subdir,
        min_frame_count=args.min_frames,
        min_volume=args.min_volume
    )