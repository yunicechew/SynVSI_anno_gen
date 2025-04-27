import pandas as pd
import os

def extract_ranked_actor_info():
    """Extract unique actor information with first appearance frame and world data"""
    # Read the input CSV file using absolute path
    input_csv = "/Users/bytedance/Desktop/SynVSI_anno_gen/Screenshot_summary.csv"
    df = pd.read_csv(input_csv)

    # Get first appearance frame for each actor
    first_appearance = df.groupby(['ActorName', 'ActorClass'])['FrameNumber'].min().reset_index()
    first_appearance = first_appearance.rename(columns={'FrameNumber': 'FirstFrame'})

    # Get unique actor information
    selected_columns = ['ActorName', 'ActorClass', 'WorldX', 'WorldY', 'WorldZ', 
                       'WorldSizeX', 'WorldSizeY', 'WorldSizeZ']
    actor_info = df[selected_columns].drop_duplicates()

    # Merge the information and sort by first appearance
    merged_info = pd.merge(first_appearance, actor_info, on=['ActorName', 'ActorClass'])
    merged_info = merged_info.sort_values('FirstFrame')
    
    # Ensure column order with FirstFrame first
    column_order = ['FirstFrame', 'ActorName', 'ActorClass', 'WorldX', 'WorldY', 'WorldZ',
                    'WorldSizeX', 'WorldSizeY', 'WorldSizeZ']
    merged_info = merged_info[column_order]

    # Create output directory if it doesn't exist
    output_dir = "anno_cleanup_tool/output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save to output CSV
    output_csv = os.path.join(output_dir, "ranked_unique_actor_anno.csv")
    merged_info.to_csv(output_csv, index=False)

    print(f"Ranked actor information has been saved to {output_csv}")

if __name__ == "__main__":
    extract_ranked_actor_info()