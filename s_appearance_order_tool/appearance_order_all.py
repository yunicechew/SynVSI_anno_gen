import pandas as pd
import os
import random
from itertools import combinations

def ensure_output_directory():
    """Create output directory if it doesn't exist"""
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def main():
    # Read the CSV file using relative path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # Assumes this script is one level down from project root
    input_csv_path = os.path.join(project_root, '0_data_cleanup_tool', 'output', 'ranked_unique_actor_anno.csv')
    df = pd.read_csv(input_csv_path)

    # Get unique actor names
    actor_names = df['ActorName'].unique()
    output_dir = ensure_output_directory()
    
    # Process all combinations of 4 different actors
    all_results = []
    possibility_counter = 1
    
    # Generate all combinations of 4 actors (order doesn't matter)
    for actor_combo in combinations(actor_names, 4):
        try:
            # Get the FirstFrame for each actor
            actor_data = []
            for actor_name in actor_combo:
                actor_row = df[df['ActorName'] == actor_name].iloc[0]
                actor_data.append({
                    'ActorName': actor_name,
                    'ShortActorName': actor_row['ShortActorName'],
                    'FirstFrame': int(actor_row['FirstFrame'])
                })
            
            # Sort actors by FirstFrame to determine appearance order
            actor_data.sort(key=lambda x: x['FirstFrame'])
            
            # Create randomized order for question (different from correct order)
            question_order = actor_data.copy()
            while question_order == actor_data:  # Ensure the order is different
                random.shuffle(question_order)
            
            # Create question with randomized order
            question_names = [actor['ShortActorName'] for actor in question_order]
            question = f"What will be the first-time appearance order of the following categories in the video: {', '.join(question_names)}?"
            
            # Create answer with correct order
            answer = f"{', '.join([actor['ShortActorName'] for actor in actor_data])}"
            
            # Record results
            all_results.append({
                'Possibility': possibility_counter,
                'Actor1': actor_combo[0],
                'Actor2': actor_combo[1],
                'Actor3': actor_combo[2],
                'Actor4': actor_combo[3],
                'Actor1_FirstFrame': df[df['ActorName'] == actor_combo[0]]['FirstFrame'].iloc[0],
                'Actor2_FirstFrame': df[df['ActorName'] == actor_combo[1]]['FirstFrame'].iloc[0],
                'Actor3_FirstFrame': df[df['ActorName'] == actor_combo[2]]['FirstFrame'].iloc[0],
                'Actor4_FirstFrame': df[df['ActorName'] == actor_combo[3]]['FirstFrame'].iloc[0],
                'Question': question,
                'Answer': answer
            })
            
            possibility_counter += 1
            
        except Exception as e:
            print(f"Error processing combination {possibility_counter}: {actor_combo}")
            print(f"Error message: {str(e)}")
            continue
    
    # Save all results to CSV
    if all_results:
        output_df = pd.DataFrame(all_results)
        output_df.to_csv(os.path.join(output_dir, 'appearance_order_all.csv'), index=False)
        print(f"Successfully processed {len(all_results)} combinations of 4 actors")

if __name__ == "__main__":
    main()