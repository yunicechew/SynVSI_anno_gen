import os
import csv
import pandas as pd
from itertools import combinations
import numpy as np
import random

def ensure_output_dir():
    """Create output directory if it doesn't exist"""
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def load_unique_objects(file_path):
    """Load unique objects from CSV file"""
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error loading unique objects: {e}")
        return None

def load_absolute_distances(file_path):
    """Load absolute distances from CSV file"""
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error loading absolute distances: {e}")
        return None

def create_distance_dict(abs_distances_df):
    """Create a dictionary of distances between object pairs"""
    distance_dict = {}
    
    for _, row in abs_distances_df.iterrows():
        obj1 = row['Actor1']
        obj2 = row['Actor2']
        distance = row['Answer']
        
        # Store distance in both directions
        distance_dict[(obj1, obj2)] = distance
        distance_dict[(obj2, obj1)] = distance
    
    return distance_dict

def get_ambiguity_threshold(room_size_csv_path):
    """
    Reads the room size from the CSV and returns the appropriate ambiguity threshold.
    0.3m for room size > 40 sq m, 0.15m otherwise.
    """
    try:
        room_size_df = pd.read_csv(room_size_csv_path)
        if not room_size_df.empty and 'Answer' in room_size_df.columns:
            room_area_sq_m = room_size_df['Answer'].iloc[0]
            if room_area_sq_m > 40:
                return 0.3
            else:
                return 0.15
        else:
            print(f"Warning: 'Answer' column not found or CSV is empty in {room_size_csv_path}. Using default threshold 0.15m.")
            return 0.15
    except FileNotFoundError:
        print(f"Warning: Room size CSV not found at {room_size_csv_path}. Using default threshold 0.15m.")
        return 0.15
    except Exception as e:
        print(f"Error reading room size CSV: {e}. Using default threshold 0.15m.")
        return 0.15

def generate_relative_distance_combinations(unique_objects_df, distance_dict, ambiguity_threshold):
    """Generate all combinations of one primary object and four option objects"""
    results = []
    possibility_counter = 1

    # --- Preprocessing for option selection ---
    # Create a map from ShortActorName to a list of its ActorName instances
    short_name_to_actor_names_map = unique_objects_df.groupby('ShortActorName')['ActorName'].apply(list).to_dict()
    all_available_short_names = list(short_name_to_actor_names_map.keys())

    # --- PrimaryObject selection (must have a unique ShortActorName) ---
    short_name_counts = unique_objects_df['ShortActorName'].value_counts()
    unique_primary_short_names = short_name_counts[short_name_counts == 1].index
    primary_object_candidates = unique_objects_df[
        unique_objects_df['ShortActorName'].isin(unique_primary_short_names)
    ]['ActorName'].tolist()

    # --- Debugging: Print PrimaryObjects NOT used (those with count > 1) ---
    multiple_short_names = short_name_counts[short_name_counts > 1].index
    not_used_primary_objects = unique_objects_df[
        unique_objects_df['ShortActorName'].isin(multiple_short_names)
    ]['ActorName'].tolist()
    
    # if not_used_primary_objects:
    #     print("--- Debug: PrimaryObjects NOT used (ShortActorName count > 1) ---")
    #     for obj_actor_name in not_used_primary_objects:
    #         # Ensure we get a single ShortActorName, even if theoretically it could be multiple (should not happen here)
    #         short_name_series = unique_objects_df[unique_objects_df['ActorName'] == obj_actor_name]['ShortActorName']
    #         if not short_name_series.empty:
    #             short_name = short_name_series.iloc[0]
    #             count = short_name_counts.get(short_name, 0)
    #             print(f"  ActorName: {obj_actor_name}, ShortActorName: {short_name}, Count: {count}")
    #         else:
    #             print(f"  ActorName: {obj_actor_name} (ShortActorName not found in unique_objects_df)")
    #     print("-----------------------------------------------------------------")
    # else:
    #     print("--- Debug: No PrimaryObjects found with ShortActorName count > 1 ---")
    
    for primary_obj_actor_name in primary_object_candidates:
        primary_obj_short_name = unique_objects_df[unique_objects_df['ActorName'] == primary_obj_actor_name]['ShortActorName'].iloc[0]
        
        # Option ShortActorNames must be different from the primary object's ShortActorName
        candidate_option_short_names = [s_name for s_name in all_available_short_names if s_name != primary_obj_short_name]

        if len(candidate_option_short_names) < 4:
            continue # Not enough unique option types to form a question

        for option_short_name_tuple in combinations(candidate_option_short_names, 4):
            current_options_details = [] # To store (representative_actor_name, short_name, min_distance)

            for selected_option_short_name in option_short_name_tuple:
                instances_for_this_option_short_name = short_name_to_actor_names_map[selected_option_short_name]
                
                min_dist_for_selected_option = float('inf')
                representative_actor_name = None

                for instance_actor_name in instances_for_this_option_short_name:
                    # Distance from the current primary object to this specific instance of the option type
                    dist = distance_dict.get((primary_obj_actor_name, instance_actor_name), float('inf'))
                    if dist < min_dist_for_selected_option:
                        min_dist_for_selected_option = dist
                        representative_actor_name = instance_actor_name
                
                if representative_actor_name is not None: # Found a valid instance
                    current_options_details.append((representative_actor_name, selected_option_short_name, min_dist_for_selected_option))
                else:
                    # This case means no instance of this option_short_name had a valid distance (e.g., all were inf)
                    # We'll add a placeholder to ensure 4 items, then filter
                    current_options_details.append((None, selected_option_short_name, float('inf')))

            # Ensure all 4 options have a valid representative actor (not None)
            if len(current_options_details) != 4 or any(detail[0] is None for detail in current_options_details):
                # print(f"Warning: Skipping combination for primary {primary_obj_short_name} due to missing representative for options: {option_short_name_tuple}")
                continue
            
            # Sort options by their minimum distance to the primary object
            current_options_details.sort(key=lambda x: x[2])
            
            # --- AMBIGUITY CHECKS ---
            # Check 1: If the closest option object is under the ambiguity_threshold (too close to primary)
            if current_options_details[0][2] < ambiguity_threshold:
                # Optional: print a message for debugging
                # print(f"Skipping (Closest option too near primary): Primary '{primary_obj_short_name}', Closest Option '{current_options_details[0][1]}' (dist: {current_options_details[0][2]:.2f}m) < threshold ({ambiguity_threshold}m)")
                continue # Discard this possibility
            
            # Check 2: Original ambiguity check (difference between option distances)
            is_ambiguous_between_options = False
            for i in range(len(current_options_details) - 1):
                dist1 = current_options_details[i][2]
                dist2 = current_options_details[i+1][2]
                if abs(dist1 - dist2) < ambiguity_threshold:
                    is_ambiguous_between_options = True
                    break
            
            if is_ambiguous_between_options:
                # print(f"Skipping (Ambiguous distances between options): Primary '{primary_obj_short_name}', Distances: {[round(d[2], 2) for d in current_options_details]}, Threshold: {ambiguity_threshold}m")
                continue # Discard ambiguous question
            # --- End Ambiguity Checks ---

            # Get the ShortActorName of the correct answer (closest object)
            correct_answer_short_name = current_options_details[0][1]

            # Get all option ShortActorNames that will be presented in the question
            all_option_short_names_for_display = [detail[1] for detail in current_options_details]
            
            # Shuffle these display options
            shuffled_display_options_short_names = random.sample(all_option_short_names_for_display, len(all_option_short_names_for_display))
            
            # Format options as A., B., C., D.
            formatted_options = [f"{chr(65+i)}. {opt}" for i, opt in enumerate(shuffled_display_options_short_names)]
            
            # Determine the answer letter
            answer_letter = ""
            try:
                answer_letter = chr(65 + shuffled_display_options_short_names.index(correct_answer_short_name))
            except ValueError:
                # This should ideally not happen if correct_answer_short_name is always in all_option_short_names
                print(f"Warning: Correct answer '{correct_answer_short_name}' not found in shuffled options for primary {primary_obj_short_name}. Options: {shuffled_display_options_short_names}")
                continue # Skip this possibility if answer cannot be found

            # The question will list the options
            # Extract just the names for the question string: "objectA, objectB, objectC, or objectD"
            option_names_for_question_text = [opt.split('. ')[1] for opt in formatted_options]
            question = f"Measuring from the closest point of each object, which of these objects ({', '.join(option_names_for_question_text[:-1])}, or {option_names_for_question_text[-1]}) is the closest to the {primary_obj_short_name}?"
            
            result = {
                'Possibility': possibility_counter,
                'PrimaryObject': primary_obj_actor_name, # Actual ActorName of the primary
                'OptionObject1': current_options_details[0][0], # Representative ActorName for option 1 (closest)
                'OptionObject2': current_options_details[1][0], # Representative ActorName for option 2
                'OptionObject3': current_options_details[2][0], # Representative ActorName for option 3
                'OptionObject4': current_options_details[3][0], # Representative ActorName for option 4
                'Distance1': current_options_details[0][2],
                'Distance2': current_options_details[1][2],
                'Distance3': current_options_details[2][2],
                'Distance4': current_options_details[3][2],
                'Question': question,
                'Answer': answer_letter, # The letter (A, B, C, D)
                'Options': formatted_options # The list of formatted options ['A. actor', 'B. actor', ...]
            }
            results.append(result)
            possibility_counter += 1
            
    return results

def save_relative_distances(results, output_path):
    """Save relative distance results to CSV"""
    try:
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
        # print(f"Relative distances saved to {output_path}")
        return True
    except Exception as e:
        print(f"Error saving relative distances: {e}")
        return False

def main():
    # File paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    unique_objects_path = os.path.join(base_dir, "0_data_cleanup_tool", "output", "ranked_unique_actor_anno.csv")
    absolute_distances_path = os.path.join(base_dir, "m_absolute_distance_tool", "output", "absolute_distances_all.csv")
    room_size_csv_path = os.path.join(base_dir, "m_room_size_tool", "output", "room_size_all.csv")
    
    # Create output directory
    output_dir = ensure_output_dir()
    output_path = os.path.join(output_dir, "relative_distance_all.csv")
    
    # Load data
    # print(f"Loading unique objects from {unique_objects_path}")
    unique_objects_df = load_unique_objects(unique_objects_path)
    if unique_objects_df is None:
        return
    
    # print(f"Loading absolute distances from {absolute_distances_path}")
    abs_distances_df = load_absolute_distances(absolute_distances_path)
    if abs_distances_df is None:
        return
    
    # Determine ambiguity threshold
    ambiguity_threshold = get_ambiguity_threshold(room_size_csv_path)
    print(f"Using ambiguity threshold: {ambiguity_threshold}m")

    # Create distance dictionary
    distance_dict = create_distance_dict(abs_distances_df)
    
    # Generate relative distance combinations
    # print("Generating relative distance combinations...")
    results = generate_relative_distance_combinations(unique_objects_df, distance_dict, ambiguity_threshold)
    
    # Save results
    save_relative_distances(results, output_path)
    
    print(f"Successfully processed {len(results)} possibility")

if __name__ == "__main__":
    main()