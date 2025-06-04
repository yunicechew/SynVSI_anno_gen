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

def generate_relative_distance_combinations(unique_objects_df, distance_dict):
    """Generate all combinations of one primary object and four option objects"""
    all_objects = unique_objects_df['ActorName'].tolist()
    results = []
    possibility_counter = 1
    
    # For each object as primary
    for primary_obj in all_objects:
        # Get primary object's cleaned name
        primary_obj_name = unique_objects_df[unique_objects_df['ActorName'] == primary_obj]['ShortActorName'].values[0]
        
        # Get all other objects
        other_objects = [obj for obj in all_objects if obj != primary_obj]
        
        # Generate all combinations of 4 objects from the other objects
        for option_objects in combinations(other_objects, min(4, len(other_objects))):
            # Skip if we don't have exactly 4 option objects
            if len(option_objects) != 4:
                continue
                
            # Get distances from primary to each option
            distances = []
            for option_obj in option_objects:
                key = (primary_obj, option_obj)
                option_obj_name = unique_objects_df[unique_objects_df['ActorName'] == option_obj]['ShortActorName'].values[0]
                if key in distance_dict:
                    distances.append((option_obj, option_obj_name, distance_dict[key]))
                else:
                    print(f"Warning: No distance found for {primary_obj} to {option_obj}")
                    distances.append((option_obj, option_obj_name, float('inf')))
            
            # Sort options by distance (closest to furthest)
            distances.sort(key=lambda x: x[2])
            
            # Create shuffled indices for the question
            shuffled_indices = list(range(4))
            random.shuffle(shuffled_indices)
            
            # Create the question with shuffled options
            option_names = [distances[i][1] for i in shuffled_indices]
            question = f"Measuring from the closest point of each object, which of these objects ({option_names[0]}, {option_names[1]}, {option_names[2]}, {option_names[3]}) is the closest to the {primary_obj_name}?"
            
            # The answer is always the first option (closest)
            answer = distances[0][1]
            
            # Create result entry
            result = {
                'Possibility': possibility_counter,
                'PrimaryObject': primary_obj,
                'OptionObject1': distances[0][0],
                'OptionObject2': distances[1][0],
                'OptionObject3': distances[2][0],
                'OptionObject4': distances[3][0],
                'Distance1': distances[0][2],
                'Distance2': distances[1][2],
                'Distance3': distances[2][2],
                'Distance4': distances[3][2],
                'Question': question,
                'Answer': answer
            }
            
            results.append(result)
            possibility_counter += 1
    
    return results

def save_relative_distances(results, output_path):
    """Save relative distance results to CSV"""
    try:
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
        print(f"Relative distances saved to {output_path}")
        return True
    except Exception as e:
        print(f"Error saving relative distances: {e}")
        return False

def main():
    # File paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    unique_objects_path = os.path.join(base_dir, "0_data_cleanup_tool", "output", "ranked_unique_actor_anno.csv")
    absolute_distances_path = os.path.join(base_dir, "m_absolute_distance_tool", "output", "absolute_distances_all.csv")
    
    # Create output directory
    output_dir = ensure_output_dir()
    output_path = os.path.join(output_dir, "relative_distance_all.csv")
    
    # Load data
    print(f"Loading unique objects from {unique_objects_path}")
    unique_objects_df = load_unique_objects(unique_objects_path)
    if unique_objects_df is None:
        return
    
    print(f"Loading absolute distances from {absolute_distances_path}")
    abs_distances_df = load_absolute_distances(absolute_distances_path)
    if abs_distances_df is None:
        return
    
    # Create distance dictionary
    distance_dict = create_distance_dict(abs_distances_df)
    
    # Generate relative distance combinations
    print("Generating relative distance combinations...")
    results = generate_relative_distance_combinations(unique_objects_df, distance_dict)
    
    # Save results
    save_relative_distances(results, output_path)
    
    print(f"Relative distance processing completed! Generated {len(results)} combinations.")

if __name__ == "__main__":
    main()