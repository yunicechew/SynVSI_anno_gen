# SynVSI Annotation Toolkit for VQA Data Generation

## Project Overview

This project, SynVSI Annotation Toolkit, is designed to transform raw 3D spatial data, collected from Unreal Engine, into a structured Video Question Answering (VQA) dataset. The primary goal is to generate high-quality Q&A pairs suitable for training and evaluating multi-modal large language models (MLLMs).

The generation of these Q&A pairs is guided by a taxonomy derived from the VSI-Bench benchmark, focusing on spatial and spatiotemporal understanding. This taxonomy categorizes visual-spatial inquiries into several key areas:

*   **Configuration (47.6%)**: Understanding the spatial arrangement of objects, including:
    *   Relative Direction (18.9%)
    *   Relative Distance (13.8%)
    *   Object Count (11.1%)
    *   Route Plan (3.8%)
*   **Measurement (40.3%)**: Quantifying attributes of objects and spaces, such as:
    *   Object Size (18.5%)
    *   Room Size (5.6%)
    *   Absolute Distance (16.2%)
*   **Spatiotemporal (12.1%)**: Analyzing events and object states over time, for instance:
    *   Appearance Order (12.1%)

This toolkit provides a suite of scripts to process raw data, extract relevant spatial and temporal information, generate corresponding questions and answers based on the VSI-Bench taxonomy, and optionally visualize the scenarios.

## Directory Structure

-   `.gitignore`: Specifies intentionally untracked files that Git should ignore.
-   `0_data_cleanup_tool/`: Initial data processing, enrichment, and visualization.
-   `0_original_ue_anno/`: Stores original Unreal Engine annotation data and related processing scripts (e.g., frame extraction).
-   `m_absolute_distance_tool/`: Absolute distance calculations and visuals.
-   `m_object_size_tool/`: Object size calculations and visuals.
-   `m_room_size_tool/`: Room size calculations and Q&A generation.
-   `c_object_count_tool/`: Object counting and Q&A generation.
-   `c_relative_direction_tool/`: Relative direction analysis and visuals.
-   `c_relative_distance_tool/`: Relative distance comparisons and visuals.
-   `s_appearance_order_tool/`: Appearance order analysis and visuals.
-   `0_infer_and_score/`: Placeholder for Q&A consolidation, inference, and scoring.
-   `run_all.py`: Script to run the entire pipeline.

Each tool directory typically has an `output/` folder for generated CSVs and images.

## Tool Suite

The toolkit is organized into several modules, each targeting specific aspects of the VSI-Bench taxonomy to generate diverse Q&A pairs. The tools process data sequentially, with outputs from earlier stages often serving as inputs for later ones.

### 1. Data Cleanup & Extraction (`0_data_cleanup_tool/`)

-   **Functionality**: Processes raw frame data, cleans names, filters objects, converts units, and ranks objects.
-   **Key Scripts**:
    -   `anno_extraction.py`: Cleans, filters, converts units, and ranks unique actors.
    -   `2d_anno_visualization.py`: Generates 2D top-down visualizations of actor distributions.
    -   `3d_anno_visualization.py`: Generates 3D visualizations of actor distributions.
    -   *(Optional)* `0_original_ue_anno/frame_extraction.py`: Extracts and processes raw frame metadata from UE output (if available in `0_original_ue_anno/`).
-   **Inputs**:
    -   Raw UE output (e.g., `Screenshot_summary.csv`) for `anno_extraction.py`.
    -   `0_data_cleanup_tool/output/ranked_unique_actor_anno.csv` for visualizers.
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
    -   `absolute_distance_visual.png` (if visualizer run).

### 3. Object Size Analysis (`m_object_size_tool/`)

-   **Functionality**: Calculates the size of objects (e.g., longest dimension, volume) and generates related Q&A.
-   **Key Scripts**:
    -   `object_size_all.py`: Computes object sizes and generates Q&A.
    -   `object_size_visual.py`: Visualizes a selected object and its dimensions (set `POSSIBILITY_ID` in script).
-   **Inputs**:
    -   `0_data_cleanup_tool/output/ranked_unique_actor_anno.csv`
-   **Outputs** (in `m_object_size_tool/output/`):
    -   `object_size_all.csv`: Calculated object sizes and Q&A.
    -   `object_size_visual.png` (if visualizer run).

### 4. Room Size Analysis (`m_room_size_tool/`)

-   **Functionality**: Extracts room dimensions from JSON data, calculates room area, and generates Q&A about room size.
-   **Key Scripts**:
    -   `room_size_all.py`: Extracts room dimensions, calculates area, and generates Q&A.
-   **Inputs**:
    -   JSON file containing room boundary data (e.g., `result_Actor_BP_HDAGenenrator_C_UAID_*.json` from `0_original_ue_anno/`).
-   **Outputs** (in `m_room_size_tool/output/`):
    -   `room_size_all.csv`: Room dimensions, area, and Q&A.

### 5. Object Count Analysis (`c_object_count_tool/`)

-   **Functionality**: Counts objects based on specified criteria (e.g., type, location) and generates related Q&A.
-   **Key Scripts**:
    -   `object_count_all.py`: Performs counting and generates Q&A.
-   **Inputs**:
    -   `0_data_cleanup_tool/output/ranked_unique_actor_anno.csv`
-   **Outputs** (in `c_object_count_tool/output/`):
    -   `object_count_all.csv`: Object counts and Q&A.
    -   *(Visualization script might be added here if developed)*

### 6. Relative Direction Analysis (`c_relative_direction_tool/`)

-   **Functionality**: Determines relative direction of an object from an observer's viewpoint (standing at one object, facing another) and generates Q&A.
-   **Key Scripts**:
    -   `relative_direction_all.py`: Calculates directions for actor triplets and generates Q&A.
    -   `relative_direction_visual.py`: Visualizes a selected triplet (set `POSSIBILITY_ID` in script).
-   **Inputs**:
    -   `0_data_cleanup_tool/output/ranked_unique_actor_anno.csv`
-   **Outputs** (in `c_relative_direction_tool/output/`):
    -   `relative_direction_all.csv`: Relative direction data, actors, and Q&A.
    -   `relative_direction_visual.png` (if visualizer run).

### 7. Relative Distance Analysis (`c_relative_distance_tool/`)

-   **Functionality**: For a primary object and several options, identifies the closest option object and generates Q&A.
-   **Key Scripts**:
    -   `relative_distance_all.py`: Identifies closest object and generates Q&A using pre-calculated absolute distances.
    -   `relative_distance_visual.py`: Visualizes primary and option objects in 3D, highlighting the closest (set `POSSIBILITY_ID` in script).
-   **Inputs**:
    -   `0_data_cleanup_tool/output/ranked_unique_actor_anno.csv`
    -   `m_absolute_distance_tool/output/absolute_distances_all.csv`
-   **Outputs** (in `c_relative_distance_tool/output/`):
    -   `relative_distance_all.csv`: Relative distance comparison data and Q&A.
    -   `relative_distance_visual.png` (if visualizer run).

### 8. Appearance Order Analysis (`s_appearance_order_tool/`)

-   **Functionality**: Determines the order in which objects appear based on their first frame of appearance and generates Q&A about this sequence.
-   **Key Scripts**:
    -   `appearance_order_all.py`: Determines appearance order and generates Q&A.
    -   `appearance_order_visual.py`: Visualizes the timeline or sequence of object appearances (set `POSSIBILITY_ID` if applicable).
-   **Inputs**:
    -   `0_data_cleanup_tool/output/ranked_unique_actor_anno.csv`
-   **Outputs** (in `s_appearance_order_tool/output/`):
    -   `appearance_order_all.csv`: Object appearance order data and Q&A.
    -   `appearance_order_visual.png` (if visualizer run).

### 9. Inference and Scoring (`0_infer_and_score/`)

-   **Functionality**: (Planned) Consolidates Q&A from all generation tools, runs them through an inference model, and scores the results.
-   **Key Scripts**: (Planned)
    -   `qa_all.py`: Gathers and consolidates Q&A from various `_all.csv` files into a unified format.
    -   `infer_all.py`: Takes the consolidated Q&A, runs inference (e.g., using a pre-trained model), and saves results with scores.
-   **Inputs**:
    -   Various `*_all.csv` files from other tool directories (e.g., `absolute_distances_all.csv`, `relative_direction_all.csv`).
    -   Consolidated Q&A file (e.g., `all_qa.csv`) for `infer_all.py`.
-   **Outputs** (in `0_infer_and_score/output/`):
    -   `all_qa.csv` (or similar): Consolidated Q&A data.
    -   `inference_results.csv` (or similar): Inference outputs and scores.

## Getting Started
### Prerequisites

-   Python 3.x
-   Required Python packages: `pandas`, `numpy`, `matplotlib`.
-   Additional packages may be required for `0_infer_and_score/` (e.g., `transformers`, `torch`) if implemented. Check specific script requirements.

### Installation

1.  Clone the repository.
2.  Install core packages:
    ```bash
    pip install pandas numpy matplotlib
    ```
3.  Install additional packages for inference if needed (refer to `0_infer_and_score/` script imports once implemented).

### Running the Tools

1.  **Input Data**:
    -   For initial processing with <mcfile name="anno_extraction.py" path="0_data_cleanup_tool/anno_extraction.py"></mcfile>, ensure your raw data (e.g., `Screenshot_summary.csv` from UE) is correctly placed, typically in a subdirectory within <mcfolder name="0_original_ue_anno" path="0_original_ue_anno/"></mcfolder> and specify this subdirectory when running <mcfile name="anno_extraction.py" path="0_data_cleanup_tool/anno_extraction.py"></mcfile> (e.g., via command-line argument).
    -   If using the optional <mcfile name="frame_extraction.py" path="0_original_ue_anno/frame_extraction.py"></mcfile>, ensure raw UE frame metadata is in <mcfolder name="0_original_ue_anno" path="0_original_ue_anno/"></mcfolder>.
    -   Subsequent scripts generally consume outputs from previous stages, primarily <mcfile name="ranked_unique_actor_anno.csv" path="0_data_cleanup_tool/output/ranked_unique_actor_anno.csv"></mcfile>.

2.  **File Paths**: Scripts use relative paths for portability within the project structure.

3.  **Execution Order (`run_all.py`)**:
    -   The `run_all.py` script provides an automated way to execute the entire pipeline. It typically runs:
        1.  Data Cleanup & Enrichment (`0_data_cleanup_tool/`)
        2.  Measurement Tools (`m_*/`)
        3.  Comparative Tools (`c_*/`)
        4.  Sequential Tools (`s_*/`)
        5.  Visualization Scripts (optional, controlled by `ENABLE_VISUALIZATIONS` in `run_all.py`)
        6.  Inference Scripts (optional, controlled by `ENABLE_INFERENCE_SCRIPTS` in `run_all.py`)
    -   To run the entire pipeline:
        ```bash
        python3 run_all.py
        ```
    -   Modify boolean flags at the top of `run_all.py` (e.g., `ENABLE_VISUALIZATIONS`, `ENABLE_INFERENCE_SCRIPTS`, `ENABLE_FRAME_EXTRACTION`) to control optional steps.

4.  **Running Individual Scripts**:
    -   Navigate to the script's directory and run it using Python 3.
        ```bash
        cd <tool_directory>
        python3 script_name.py
        ```
    -   Example:
        ```bash
        cd 0_data_cleanup_tool
        python3 anno_extraction.py --data_subdir <your_data_subdirectory_name>
        ```
    -   Some scripts might accept command-line arguments (e.g., `--data_subdir`, `--min_frames`). Check individual scripts or run with `-h` or `--help` if `argparse` is used.

5.  **Visualization Scripts (`*_visual.py`)**:
    -   These scripts typically visualize a specific scenario or data point from an `*_all.csv` file generated by a corresponding data processing script.
    -   You may need to modify a `POSSIBILITY_ID` (or similar variable) at the top of the visualization script to select which entry from the CSV to visualize.
    -   Outputs are usually image files (e.g., `.png`) saved in the tool's `output/` directory and often displayed on screen.

6.  **Outputs**:
    -   Check the `output/` subdirectory within each tool's folder for generated CSV files (data and Q&A) and image files (visualizations).
