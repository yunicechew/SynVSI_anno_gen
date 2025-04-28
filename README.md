# SynVSI Annotation Generation Tools

This repository contains a comprehensive suite of tools for processing and visualizing spatial annotation data from virtual scenes. The tools are designed to analyze object positions, calculate distances, and determine relative spatial relationships between objects in 3D environments.

## Tools Overview

### 1. Annotation Cleanup Tool
Location: `anno_cleanup_tool/`

#### Purpose
Processes raw annotation data to extract and organize object information, creating clean datasets and visualizations for further analysis.

#### Key Files
- `anno_extraction.py`: Processes raw annotation data to extract unique actor information
  - Cleans actor names for better readability
  - Records first appearance frame for each actor
  - Preserves world coordinates and dimensions
  - Outputs organized CSV data

- `2d_anno_visualization.py`: Creates top-down 2D visualizations
  - Plots actor positions on X-Y plane
  - Draws bounding boxes for each object
  - Labels objects with names and first appearance frames
  - Includes grid and axis labels for reference

- `3d_anno_visualization.py`: Generates detailed 3D scene visualizations
  - Creates 3D bounding boxes for all objects
  - Shows spatial relationships in all dimensions
  - Provides interactive 3D view
  - Includes object labels and frame information

#### Outputs
- `ranked_unique_actor_anno.csv`: Comprehensive CSV containing:
  - Actor names and cleaned identifiers
  - First appearance frame numbers
  - World coordinates (X, Y, Z)
  - Object dimensions
- 2D and 3D visualization images for scene understanding

### 2. Absolute Distance Tool
Location: `m_absolute_distance_tool/`

#### Purpose
Calculates and visualizes precise distances between pairs of objects in the 3D environment.

#### Key Files
- `absolute_distance_all.py`: Distance calculation engine
  - Computes minimum distances between all possible object pairs
  - Considers object dimensions and bounding boxes
  - Generates natural language questions about distances
  - Handles edge cases and overlapping objects

- `absolute_distance_visual.py`: 3D distance visualization
  - Creates detailed 3D visualizations of object pairs
  - Shows minimum distance paths
  - Includes translucent bounding boxes
  - Provides distance measurements in meters

#### Outputs
- `absolute_distances_all.csv`: Comprehensive distance data
  - All possible object pair combinations
  - Precise distance measurements in meters
  - Natural language questions for each pair
  - Systematic possibility IDs for reference
- 3D visualization images showing distance relationships

### 3. Relative Direction Tool
Location: `c_relative_direction_tool/`

#### Purpose
Analyzes and describes relative spatial relationships between objects using directional terms.

#### Key Files
- `relative_direction_all.py`: Direction analysis engine
  - Calculates relative directions between object triplets
  - Supports multiple difficulty levels
  - Generates natural language questions
  - Uses quadrant-based direction system

- `relative_direction_visual.py`: Direction visualization
  - Creates visual representations of directional relationships
  - Shows reference frames and orientations
  - Highlights spatial relationships
  - Supports multiple perspective views

#### Outputs
- `relative_direction_all.csv`: Detailed directional data
  - Complete set of object triplet combinations
  - Questions at three difficulty levels
  - Precise directional answers
  - Quadrant-based classifications
- Visualization images showing direction relationships

## Usage
Each tool can be run independently from its respective directory. The tools process data from the input CSV files and generate both numerical results and visual representations for analysis.

## Dependencies
- Python 3.x
- pandas
- numpy
- matplotlib
- mpl_toolkits