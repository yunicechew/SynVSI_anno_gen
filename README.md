# SynVSI Spatial Annotation Toolkit

A comprehensive suite of tools for processing, analyzing, and visualizing spatial annotation data from virtual 3D environments. The toolkit enables researchers and developers to extract meaningful spatial relationships between objects in complex scenes.

## Key Features

- **Data Processing**: Clean and organize raw spatial annotation data
- **Distance Analysis**: Calculate precise distances between object pairs
- **Direction Analysis**: Determine relative spatial relationships
- **Visualization**: Generate 2D and 3D representations of spatial data

## Tool Suite

### 1. Data Cleanup & Extraction
Location: `0_data_cleanup_tool/`

#### Core Functionality
- Processes raw annotation frames into structured datasets
- Extracts and cleans object metadata (names, positions, dimensions)
- Filters objects based on visibility and size thresholds
- Converts spatial measurements to consistent units

#### Key Components
- `anno_extraction.py`: Main data processing pipeline
- `2d_anno_visualization.py`: Top-down 2D scene plots
- `3d_anno_visualization.py`: Interactive 3D scene views

#### Outputs
- `ranked_unique_actor_anno.csv`: Cleaned object metadata
- Visualizations showing object positions and relationships

### 2. Absolute Distance Analysis
Location: `m_absolute_distance_tool/`

#### Core Functionality
- Computes minimum distances between all object pairs
- Generates natural language questions about spatial relationships
- Handles complex cases like overlapping objects

#### Key Components
- `absolute_distance_all.py`: Distance calculation engine
- `absolute_distance_visual.py`: 3D distance visualization

#### Outputs
- `absolute_distances_all.csv`: Comprehensive distance measurements
- Visualizations highlighting distance relationships

### 3. Relative Direction Analysis
Location: `c_relative_direction_tool/`

#### Core Functionality
- Analyzes directional relationships between object triplets
- Supports multiple difficulty levels for question generation
- Uses quadrant-based direction classification

#### Key Components
- `relative_direction_all.py`: Direction analysis engine
- `relative_direction_visual.py`: Direction visualization

#### Outputs
- `relative_direction_all.csv`: Directional relationship data
- Visualizations showing reference frames and orientations

## Getting Started

### Prerequisites
- Python 3.x
- Required packages:
  ```bash
  pip install pandas numpy matplotlib mpl_toolkits
  ```