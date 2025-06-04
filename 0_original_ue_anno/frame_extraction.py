import os
import subprocess
import glob
import re
import pandas as pd
import shutil

# Configuration variables
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)  # Navigates up from 0_original_ue_anno to SynVSI_anno_gen

# Define the specific data subdirectory name (e.g., timestamped folder)
# This can be changed if you process a different dataset
DATA_SUBDIRECTORY_NAME = "20250527-145925"

# Construct the base path to the input data using the project root
INPUT_DATA_ROOT = os.path.join(script_dir, DATA_SUBDIRECTORY_NAME)

ANNOTATION_DIR = os.path.join(INPUT_DATA_ROOT, "annotation")
ORIGINAL_DIR = os.path.join(INPUT_DATA_ROOT, "original")
SUMMARY_CSV = os.path.join(INPUT_DATA_ROOT, "Screenshot_summary.csv")
FPS = 2  # Frames per second - fixed rate

def natural_sort_key(s):
    """Natural sort key function for Screenshot_{count} pattern"""
    return [int(text) if text.isdigit() else text.lower() 
            for text in re.split('([0-9]+)', s)]

def create_video_from_frames(input_dir, output_path):
    """
    Create a video from frames in the input directory using ffmpeg
    Args:
        input_dir: Directory containing the frame images
        output_path: Path where the output video will be saved
    Returns:
        List of tuples containing (original_frame_path, video_timestamp)
    """
    # Ensure input directory exists
    if not os.path.exists(input_dir):
        print(f"Input directory not found: {input_dir}")
        return None
    
    # Get all frame files and sort them naturally
    all_frames = sorted(glob.glob(os.path.join(input_dir, 'Screenshot_*.png')), key=natural_sort_key)
    total_frames = len(all_frames)
    
    if total_frames == 0:
        print(f"No frames found in {input_dir}")
        return None
        
    # Create a temporary directory for frames
    temp_dir = os.path.join(os.path.dirname(output_path), 'temp_frames')
    os.makedirs(temp_dir, exist_ok=True)
    
    frame_metadata = []
    
    try:
        # Use all frames in sequence
        for i, source in enumerate(all_frames):
            target = os.path.join(temp_dir, f'frame_{i:04d}.png')
            if os.path.exists(target):
                os.remove(target)
            os.symlink(source, target)
            
            # Calculate timestamp in seconds
            timestamp = i / FPS
            frame_metadata.append((source, timestamp))
        
        # Calculate video duration based on frame count and FPS
        video_duration = total_frames / FPS
        
        # Construct ffmpeg command
        ffmpeg_cmd = [
            'ffmpeg',
            '-y',  # Overwrite output file if exists
            '-framerate', str(FPS),  # Input framerate
            '-i', os.path.join(temp_dir, 'frame_%04d.png'),  # Input pattern
            '-vf', 'pad=width=ceil(iw/2)*2:height=ceil(ih/2)*2',  # Pad to even dimensions
            '-c:v', 'libx264',  # Use H.264 codec
            '-pix_fmt', 'yuv420p',  # Pixel format for better compatibility
            '-r', str(FPS),  # Output framerate
            output_path
        ]
        
        # Run ffmpeg command
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
        print(f"Video created: {output_path}")
        print(f"Used all {total_frames} frames at {FPS} FPS for {video_duration:.2f} seconds")
        
        return frame_metadata
        
    except subprocess.CalledProcessError as e:
        print(f"Error creating video: {e.stderr.decode()}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
    finally:
        # Clean up temporary directory
        for file in glob.glob(os.path.join(temp_dir, '*.png')):
            os.remove(file)
        os.rmdir(temp_dir)
    
    return None

def save_frame_metadata(frame_metadata, output_csv_path):
    """Save metadata for only the selected frames from Screenshot_summary.csv"""
    if not os.path.exists(SUMMARY_CSV):
        print(f"Summary CSV not found: {SUMMARY_CSV}")
        return
        
    # Read the summary CSV
    df = pd.read_csv(SUMMARY_CSV)
    
    # Extract frame numbers from selected frames
    selected_frame_numbers = []
    for frame_path, _ in frame_metadata:
        frame_num = int(re.search(r'Screenshot_(\d+)\.png', frame_path).group(1))
        selected_frame_numbers.append(frame_num)
    
    # Filter the dataframe to only include selected frames
    selected_df = df[df['FrameNumber'].isin(selected_frame_numbers)]
    
    # Save to output location
    selected_df.to_csv(output_csv_path, index=False)
    print(f"Selected frame metadata saved to: {output_csv_path}")

def main():
    # Save output directly in the DATA_SUBDIRECTORY_NAME folder instead of creating a new output folder
    output_dir = INPUT_DATA_ROOT
    
    # Process annotated frames and get metadata
    annotated_output = os.path.join(output_dir, "annotation_video.mp4")
    anno_metadata = create_video_from_frames(ANNOTATION_DIR, annotated_output)
    
    # Process original frames
    original_output = os.path.join(output_dir, "original_video.mp4")
    orig_metadata = create_video_from_frames(ORIGINAL_DIR, original_output)
    
    # Metadata saving is now commented out - only create videos
    # if anno_metadata:
    #     metadata_output = os.path.join(output_dir, "frame_extract_meta.csv")
    #     save_frame_metadata(anno_metadata, metadata_output)

if __name__ == "__main__":
    main()