import os
import subprocess
import sys

<<<<<<< HEAD
# Toggle flags for visualization scripts
ENABLE_VISUALIZATIONS = False  # Set to False to skip all visualization scripts
=======
# Toggle flags for scripts
ENABLE_VISUALIZATIONS = True  # Set to False to skip all visualization scripts
ENABLE_FRAME_EXTRACTION = True  # Set to False to skip frame extraction script
>>>>>>> temp-branch

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
    
<<<<<<< HEAD
    # Separate data processing and visualization scripts
    data_scripts = [
        "anno_cleanup_tool/anno_extraction.py",
        "c_relative_direction_tool/relative_direction_all.py",
        "m_absolute_distance_tool/absolute_distance_all.py",
    ]
    
    visualization_scripts = [
        "anno_cleanup_tool/2d_anno_visualization.py",
        "anno_cleanup_tool/3d_anno_visualization.py",
        "c_relative_direction_tool/relative_direction_visual.py",
        "m_absolute_distance_tool/absolute_distance_visual.py"
=======
    # Frame extraction script (optional)
    frame_extraction_script = "0_data_cleanup_tool/frame_extraction.py"
    
    # Separate data processing and visualization scripts
    data_scripts = [
        "0_data_cleanup_tool/anno_extraction.py",
        "c_relative_direction_tool/relative_direction_all.py",
        "m_absolute_distance_tool/absolute_distance_all.py",
        "c_relative_distance_tool/relative_distance_all.py"
    ]
    
    visualization_scripts = [
        "0_data_cleanup_tool/2d_anno_visualization.py",
        "0_data_cleanup_tool/3d_anno_visualization.py",
        "c_relative_direction_tool/relative_direction_visual.py",
        "m_absolute_distance_tool/absolute_distance_visual.py",
        "c_relative_distance_tool/relative_distance_visual.py"
>>>>>>> temp-branch
    ]
    
    print("Starting to run data processing scripts...")
    
<<<<<<< HEAD
=======
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
    
>>>>>>> temp-branch
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
    
    print("\nAll scripts execution completed!")

if __name__ == "__main__":
    main()