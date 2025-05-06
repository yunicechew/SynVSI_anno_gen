import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Configuration variables
POSSIBILITY_ID = 1  # Change this to visualize different combinations

def visualize_result(result_data, standing_at_name, facing_at_name, locate_at_name):
    plt.figure(figsize=(10, 10))
    
    # Extract points from the result data (updated to match new CSV structure)
    points = {
        'standing_at': {'point': np.array([result_data['standing_at_x'].iloc[0],
                                         result_data['standing_at_y'].iloc[0]]),
                       'name': standing_at_name, 'color': 'green', 'label': 'Standing At (Origin)'},
        'facing_at': {'point': np.array([result_data['facing_at_x'].iloc[0],
                                       result_data['facing_at_y'].iloc[0]]),
                     'name': facing_at_name, 'color': 'orange', 'label': 'Facing At (Y-axis)'},
        'locate_at': {'point': np.array([result_data['locate_at_x'].iloc[0],
                                       result_data['locate_at_y'].iloc[0]]),
                     'name': locate_at_name, 'color': 'red', 'label': 'Locate At'}
    }
    
    # Calculate axes for visualization
    standing_at = points['standing_at']['point']
    facing_at = points['facing_at']['point']
    
    # Create coordinate system
    y_axis = facing_at - standing_at
    y_axis = y_axis / np.linalg.norm(y_axis)
    x_axis = np.array([y_axis[1], -y_axis[0]])
    
    # Plot all points and their labels
    for data in points.values():
        plt.scatter(data['point'][0], data['point'][1], color=data['color'], s=100, label=data['label'])
        plt.annotate(data['name'], 
                    (data['point'][0], data['point'][1]),
                    xytext=(0, -15),
                    textcoords='offset points',
                    ha='center',
                    fontsize=7,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
    
    # Draw relative coordinate system
    arrow_length = np.linalg.norm(facing_at - standing_at)
    for axis, vector, color, label in [('x_axis', x_axis, 'blue', 'Relative X-axis'), 
                                     ('y_axis', y_axis, 'magenta', 'Relative Y-axis')]:
        # Calculate positive and negative axis lengths
        pos = vector * arrow_length * (0.5 if axis == 'x_axis' else 1)
        neg = vector * (-arrow_length) * (0.5 if axis == 'x_axis' else 0.5)
        
        # Draw negative axis (without arrow)
        plt.plot([standing_at[0], standing_at[0] + neg[0]], 
                [standing_at[1], standing_at[1] + neg[1]], 
                color=color, label=label if axis == 'x_axis' else None)
        
        # Draw positive axis (with arrow)
        plt.arrow(standing_at[0], standing_at[1], 
                pos[0], pos[1], head_width=0.03, head_length=0.03, fc=color, ec=color,
                label=label if axis == 'y_axis' else None)
    
    # Draw dashed line from origin to target point
    plt.plot([points['standing_at']['point'][0], points['locate_at']['point'][0]], 
             [points['standing_at']['point'][1], points['locate_at']['point'][1]], 'r--')
    
    # Get directions from the data (updated to show all three difficulty levels)
    direction_hard = result_data['AnswerHard'].iloc[0]
    direction_medium = result_data['AnswerMedium'].iloc[0]
    direction_easy = result_data['AnswerEasy'].iloc[0]
    
    # Configure plot appearance with all direction levels
    plt.title(f'Relative Directions (Possibility {result_data["Possibility"].iloc[0]})\n'
              f'Hard: {direction_hard} | Medium: {direction_medium} | Easy: {direction_easy}')
    plt.xlabel('World X (fixed)')
    plt.ylabel('World Y (fixed)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axis('equal')
    plt.legend()
    
    plt.tight_layout()
    
    # Save visualization and display
    output_dir = ensure_output_directory()
    plt.savefig(os.path.join(output_dir, f'relative_direction_visual.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def ensure_output_directory():
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def main():
    # Read the pre-calculated results
    df_all = pd.read_csv(os.path.join(os.path.dirname(__file__), 'output', 'relative_direction_all.csv'))
    
    # Get data for the specified possibility
    result_data = df_all[df_all['Possibility'] == POSSIBILITY_ID]
    
    if len(result_data) == 0:
        print(f"No data found for possibility {POSSIBILITY_ID}")
        return
    
    # Get actor names for this possibility
    standing_at_name = result_data['standing_at'].iloc[0]
    facing_at_name = result_data['facing_at'].iloc[0]
    locate_at_name = result_data['locate_at'].iloc[0]
    
    # Generate visualization
    visualize_result(result_data, standing_at_name, facing_at_name, locate_at_name)
    
    print(f"Visualizing possibility {POSSIBILITY_ID}:") 
    print(f"Standing at: {standing_at_name}")
    print(f"Facing at: {facing_at_name}")
    print(f"Locating: {locate_at_name}")
    print(f"Directions:")
    print(f"  Hard: {result_data['AnswerHard'].iloc[0]}")
    print(f"  Medium: {result_data['AnswerMedium'].iloc[0]}")
    print(f"  Easy: {result_data['AnswerEasy'].iloc[0]}")

if __name__ == "__main__":
    main()