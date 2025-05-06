# SynVSI Spatial Annotation Toolkit

<<<<<<< HEAD
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
=======
Tools for processing, analyzing, and visualizing 3D spatial annotation data. Extracts spatial relationships, generates Q&A, and creates visualizations.

## Key Features

-   **Data Processing**: Cleans and standardizes spatial data.
-   **Absolute Distance**: Calculates distances between object pairs.
-   **Relative Direction**: Determines directional relationships (e.g., object A is to the left of object B from C's view).
-   **Relative Distance**: Finds the closest object to a primary one from a set of options.
-   **Q&A Generation**: Creates questions and answers for spatial reasoning.
-   **Visualization**: Offers 2D and 3D views of objects and relationships.

## Directory Structure

-   `0_data_cleanup_tool/`: Initial data processing and visualization.
-   `m_absolute_distance_tool/`: Absolute distance calculations and visuals.
-   `c_relative_direction_tool/`: Relative direction analysis and visuals.
-   `c_relative_distance_tool/`: Relative distance comparisons and visuals.
-   `run_all.py`: Script to run the entire pipeline (verify its specific use).

Each tool directory has an `output/` folder for generated CSVs and images.

## Tool Suite

### 1. Data Cleanup & Extraction (`0_data_cleanup_tool/`)

-   **Functionality**: Processes raw frame data, cleans names, filters objects, converts units, and ranks objects.
-   **Key Scripts**:
    -   `frame_extraction.py`: Extracts and processes raw frame metadata.
    -   `anno_extraction.py`: Cleans, filters, converts units, and ranks unique actors.
    -   `2d_anno_visualization.py`: Generates 2D top-down visualizations.
    -   `3d_anno_visualization.py`: Generates 3D visualizations.
-   **Inputs**:
    -   Initial data (e.g., `frame_extract_meta.csv` in `0_data_cleanup_tool/output/`).
-   **Outputs** (in `0_data_cleanup_tool/output/`):
    -   `ranked_unique_actor_anno.csv`: Cleaned and ranked actor metadata.
    -   `2d_anno_visualization.png`: 2D actor distribution plot.
    -   `3d_anno_visualization.png`: 3D actor distribution plot.

### 2. Absolute Distance Analysis (`m_absolute_distance_tool/`)

-   **Functionality**: Calculates minimum distances between all unique object pairs and generates related Q&A.
-   **Key Scripts**:
    -   `absolute_distance_all.py`: Computes distances and generates Q&A.
    -   `absolute_distance_visual.py`: Visualizes 3D bounding boxes and distance for a selected pair (set `POSSIBILITY_ID` in script).
-   **Inputs**:
    -   `0_data_cleanup_tool/output/ranked_unique_actor_anno.csv`
-   **Outputs** (in `m_absolute_distance_tool/output/`):
    -   `absolute_distances_all.csv`: All calculated distances, pairs, and Q&A.
    -   `absolute_distance_visual.png`: 3D visualization for a specific pair.

### 3. Relative Direction Analysis (`c_relative_direction_tool/`)

-   **Functionality**: Determines relative direction of an object from an observer's viewpoint (standing at one object, facing another) and generates Q&A.
-   **Key Scripts**:
    -   `relative_direction_all.py`: Calculates directions for actor triplets and generates Q&A.
    -   `relative_direction_visual.py`: Visualizes a selected triplet (set `POSSIBILITY_ID` in script).
-   **Inputs**:
    -   `0_data_cleanup_tool/output/ranked_unique_actor_anno.csv`
-   **Outputs** (in `c_relative_direction_tool/output/`):
    -   `relative_direction_all.csv`: Relative direction data, actors, and Q&A.
    -   `relative_direction_visual.png` (if visualizer run).

### 4. Relative Distance Analysis (`c_relative_distance_tool/`)

-   **Functionality**: For a primary object and several options, identifies the closest option object and generates Q&A.
-   **Key Scripts**:
    -   `relative_distance_all.py`: Identifies closest object and generates Q&A using pre-calculated absolute distances.
    -   `relative_distance_visual.py`: Visualizes primary and option objects in 3D, highlighting the closest (set `POSSIBILITY_ID` in script).
-   **Inputs**:
    -   `0_data_cleanup_tool/output/ranked_unique_actor_anno.csv`
    -   `m_absolute_distance_tool/output/absolute_distances_all.csv`
-   **Outputs** (in `c_relative_distance_tool/output/`):
    -   `relative_distance_all.csv`: Relative distance comparison data and Q&A.
    -   `relative_distance_visual.png`: 3D visualization for a specific scenario.
>>>>>>> temp-branch

## Getting Started

### Prerequisites
<<<<<<< HEAD
- Python 3.x
- Required packages:
  ```bash
  pip install pandas numpy matplotlib mpl_toolkits
  ```
=======

-   Python 3.x
-   Required Python packages: `pandas`, `numpy`, `matplotlib`.

### Installation

1.  Clone the repository.
2.  Install packages:
    ```bash
    pip install pandas numpy matplotlib
    ```

### Running the Tools

1.  **Input Data**: Place initial data (e.g., `frame_extract_meta.csv`) in `0_data_cleanup_tool/output/` or as configured.
2.  **File Paths**: Scripts use relative paths for portability.
3.  **Execution Order**:
    -   Run `0_data_cleanup_tool/` scripts first (e.g., `anno_extraction.py`).
    -   Then, `m_absolute_distance_tool/` scripts (e.g., `absolute_distance_all.py`).
    -   Finally, `c_relative_direction_tool/` and `c_relative_distance_tool/` scripts.
    -   Check `run_all.py` for automated execution.
4.  **Running Individual Scripts**:
    Navigate to the script's directory and run:
    ```bash
    python3 script_name.py
    ```
    Example:
    ```bash
    cd 0_data_cleanup_tool
    python3 anno_extraction.py
    ```
5.  **Visualization Scripts** (`_visual.py`):
    -   These visualize specific scenarios.
    -   Modify `POSSIBILITY_ID` (or similar) at the top of the script to select data from the corresponding `_all.csv`.
    -   Outputs plots and image files (e.g., `.png`) in the tool's `output/` directory.
6.  **Outputs**: Check `output/` subdirectories in each tool's folder for results.
>>>>>>> temp-branch
