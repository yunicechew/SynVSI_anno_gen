# SynVSI Annotation Generation Tools

This repository contains several tools for processing and visualizing spatial annotation data from virtual scenes.

## Tools Overview

### 1. Annotation Cleanup Tool
Location: `anno_cleanup_tool/`

#### Key Files:
- `anno_extraction.py`: Extracts unique actor information from raw annotation data
- `2d_anno_visualization.py`: Creates 2D visualizations of actor positions and bounding boxes
- `3d_anno_visualization.py`: Generates 3D visualizations of actor positions with bounding boxes

#### Outputs:
- `ranked_unique_actor_anno.csv`: Processed CSV with actor information
- 2D and 3D visualization images

### 2. Absolute Distance Tool
Location: `m_absolute_distance_tool/`

#### Key Files:
- `absolute_distance_all.py`: Calculates minimum distances between all actor pairs
- `absolute_distance_visual.py`: Visualizes 3D distances between specific actor pairs

#### Outputs:
- `absolute_distances_all.csv`: CSV with all pairwise distance calculations
- 3D visualization images showing distance relationships

### 3. Relative Direction Tool
Location: `c_relative_direction_tool/`

#### Key Files:
- `relative_direction_all.py`: Calculates relative directions between actor triplets
- `relative_direction_visual.py`: Visualizes relative direction relationships

#### Outputs:
- `relative_direction_all.csv`: CSV with all relative direction calculations
- Visualization images showing direction relationships