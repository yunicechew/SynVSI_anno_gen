import os
import subprocess
import sys

# Toggle flags for scripts
ENABLE_VISUALIZATIONS = True  # Set to False to skip all visualization scripts
ENABLE_FRAME_EXTRACTION = True  # Set to False to skip frame extraction script
ENABLE_INFERENCE_SCRIPTS = False # Set to False to skip inference scripts

def run_script(script_path):
    """Run a Python script and check for errors"""
    print(f"\nRunning: {os.path.basename(script_path)}")
    print("-" * 50)
    result = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running {script_path}:")
        print(result.stderr)
        return False
    
    print(result.stdout)
    return True

def main():
    # Base directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Frame extraction script (optional)
    frame_extraction_script = "0_original_ue_anno/frame_extraction.py"
    
    # Organize scripts by their categories
    data_cleanup_scripts = [
        "0_data_cleanup_tool/anno_extraction.py",
        "0_data_cleanup_tool/actor_visual_description.py",
    ]
    
    # Merged list of main processing scripts
    main_processing_scripts = [
        # Measurement scripts
        "m_absolute_distance_tool/absolute_distance_all.py",
        "m_object_size_tool/object_size_all.py",
        "m_room_size_tool/room_size_all.py",
        # Configuration scripts
        "c_object_count_tool/object_count_all.py",
        "c_relative_direction_tool/relative_direction_all.py",
        "c_relative_distance_tool/relative_distance_all.py",
        # Spatiotemporal scripts
        "s_appearance_order_tool/appearance_order_all.py"
    ]
    
    # Combine all data processing scripts in execution order
    data_scripts = (
        data_cleanup_scripts +
        main_processing_scripts + # Use the new merged list
        ["0_infer_and_score/qa_all.py"] # Add qa_all.py to the end of data processing
    )
    
    visualization_scripts = [
        # Data cleanup visualizations
        "0_data_cleanup_tool/2d_anno_visualization.py",
        "0_data_cleanup_tool/3d_anno_visualization.py",
        # Measurement visualizations
        "m_absolute_distance_tool/absolute_distance_visual.py",
        "m_object_size_tool/object_size_visual.py",
        # Configuration visualizations # Updated comment
        "c_relative_direction_tool/relative_direction_visual.py",
        "c_relative_distance_tool/relative_distance_visual.py",
        # Spatiotemporal visualizations # Updated comment
        "s_appearance_order_tool/appearance_order_visual.py"
    ]
    
    inference_scripts = [
        "0_infer_and_score/infer_all.py" # Only infer_all.py remains for dedicated inference step
    ]
    
    print("Starting to run data processing scripts...")
    
    # Run frame extraction script if enabled
    if ENABLE_FRAME_EXTRACTION:
        frame_extraction_path = os.path.join(base_dir, frame_extraction_script)
        if not os.path.exists(frame_extraction_path):
            print(f"Error: Frame extraction script not found: {frame_extraction_path}")
        else:
            print("\nRunning frame extraction script...")
            success = run_script(frame_extraction_path)
            if not success:
                print(f"\nExecution stopped due to error in {frame_extraction_script}")
                return
    else:
        print("\nSkipping frame extraction script (ENABLE_FRAME_EXTRACTION is False)")
    
    # Run data processing scripts
    for script in data_scripts:
        script_path = os.path.join(base_dir, script)
        if not os.path.exists(script_path):
            print(f"Error: Script not found: {script_path}")
            continue
            
        success = run_script(script_path)
        if not success:
            print(f"\nExecution stopped due to error in {script}")
            return
    
    # Run visualization scripts if enabled
    if ENABLE_VISUALIZATIONS:
        print("\nStarting to run visualization scripts...")
        for script in visualization_scripts:
            script_path = os.path.join(base_dir, script)
            if not os.path.exists(script_path):
                print(f"Error: Script not found: {script_path}")
                continue
                
            success = run_script(script_path)
            if not success:
                print(f"\nExecution stopped due to error in {script}")
                return
    else:
        print("\nSkipping visualization scripts (ENABLE_VISUALIZATIONS is False)")
    
    # Run inference scripts if enabled
    if ENABLE_INFERENCE_SCRIPTS:
        print("\nStarting to run inference scripts...")
        for script in inference_scripts:
            script_path = os.path.join(base_dir, script)
            if not os.path.exists(script_path):
                print(f"Error: Script not found: {script_path}")
                continue
            
            success = run_script(script_path)
            if not success:
                print(f"\nExecution stopped due to error in {script}")
                return
    else:
        print("\nSkipping inference scripts (ENABLE_INFERENCE_SCRIPTS is False)")
    
    print("\nAll scripts execution completed!")

if __name__ == "__main__":
    main()