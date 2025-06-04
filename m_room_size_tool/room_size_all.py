import pandas as pd
import os
import json

def ensure_output_directory():
    """Create output directory if it doesn't exist and return its path."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    return output_dir

def main():
    """
    Main function to extract room dimensions, calculate area, generate QA pairs,
    and save them to a CSV file.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir) # Navigates to SynVSI_anno_gen

    # Path to the JSON file containing room dimensions
    # This path is hardcoded as per the request, assuming it's a specific input.
    json_file_path = os.path.join(project_root, '0_original_ue_anno', '20250527-145925', 'result_Actor_BP_HDAGenenrator_C_UAID_18C04D8F75B6AC6E02_1312812927_1748996305_1748997426.json')

    output_dir = ensure_output_directory()
    all_results = []
    possibility_counter = 1

    if not os.path.exists(json_file_path):
        print(f"Error: JSON file not found at {json_file_path}")
        return

    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)

        room_bounds_cm = data.get('room1_bound')
        if room_bounds_cm is None or len(room_bounds_cm) != 2:
            print(f"Error: 'room1_bound' not found or invalid in {json_file_path}")
            return

        # Convert dimensions from centimeters to meters
        # 1 meter = 100 centimeters
        length_m = room_bounds_cm[0] / 100.0
        width_m = room_bounds_cm[1] / 100.0

        # Calculate area in square meters
        area_sq_m = length_m * width_m

        question = "What is the size of this room (in square meters)? If multiple rooms are shown, estimate the size of the combined space."
        answer = round(area_sq_m, 2) # Round to 2 decimal places for readability

        all_results.append({
            'Possibility': possibility_counter,
            'RoomLength_cm': room_bounds_cm[0],
            'RoomWidth_cm': room_bounds_cm[1],
            'RoomLength_m': length_m,
            'RoomWidth_m': width_m,
            'Question': question,
            'Answer': answer
        })

        # Save all results to CSV
        output_csv_path = os.path.join(output_dir, 'room_size_all.csv')
        output_df = pd.DataFrame(all_results)
        output_df.to_csv(output_csv_path, index=False)
        print(f"Successfully processed room size and saved QA pair to {output_csv_path}")

        # Print example
        print("\nExample question and answer:")
        print(f"Q: {question}")
        print(f"A: {answer} sq m")

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {json_file_path}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()