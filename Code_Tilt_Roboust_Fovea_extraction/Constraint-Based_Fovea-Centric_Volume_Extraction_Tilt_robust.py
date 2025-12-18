import os
import shutil
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# --- Configuration ---
SOURCE_ROOT_DIR = r"Constraint-Based_Fovea-Centrtic_Test_Data"
DESTINATION_ROOT = r"Constraint-Based_Fovea-Centrtic_33_Extracted_data"
NUM_ADJACENT_SLICES = 16 
TOTAL_SLICES = 128

# --- CONSTRAINT ---
SEARCH_START_SLICE = 59
SEARCH_END_SLICE = 69

# --- CRITICAL FIX CONSTANTS ---
PENALTY_VALUE = 200.0 

# --- Image Processing Constants (Optimized for Retinal Tissue Detection) ---
CENTER_WIDTH_PERCENT = 0.35
GAUSSIAN_KERNEL_SIZE = (7, 7)

# --- Core Image Processing Logic: Tilt-Robust A-Scan Analysis ---

# This function is entirely written to be robust to image rotation/tilt.
# It now finds the minimum (highest) centroid of all individual vertical
# columns (A-scans) instead of one global centroid.
def find_foveal_pit_metric(img: np.ndarray) -> float:
    """
    Calculates a tilt-robust metric for the foveal pit.
    
    This works by:
    1. Segmenting the central retinal tissue.
    2. Calculating the centroid_y for EACH vertical column (A-scan).
    3. Returning the MINIMUM (highest in the image) of these column-wise
       centroids.
       
    This value corresponds to the thinnest point of the retina (the pit),
    even if the entire scan is tilted.
    """
    height, width = img.shape
    
    start_col = int(width * (0.5 - CENTER_WIDTH_PERCENT / 2))
    end_col = int(width * (0.5 + CENTER_WIDTH_PERCENT / 2))
    central_img = img[:, start_col:end_col]
    
    if central_img.size == 0:
        return float(height) # Return bad score if crop is invalid

    # 1. Blur and Adaptive Threshold (same as before)
    blurred_img = cv2.GaussianBlur(central_img, GAUSSIAN_KERNEL_SIZE, 0)
    _, segmented_tissue = cv2.threshold(
        blurred_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    
    # 2. Column-wise (A-Scan) Centroid Analysis (NumPy optimized)
    
    # Convert to float64 for precision in moment calculations
    segmented_tissue = segmented_tissue.astype(np.float64) / 255.0

    # m00 (sum of pixels) for each column
    column_sums = np.sum(segmented_tissue, axis=0)
    
    # Create a [y] vector (0, 1, 2... height-1) for weighted sum
    y_indices = np.arange(height, dtype=np.float64).reshape(-1, 1)
    
    # m01 (weighted sum) for each column
    # This multiplies each column by the y_indices vector and sums it up.
    column_weighted_sums = np.sum(segmented_tissue * y_indices, axis=0)

    # Initialize all column centroids to a "bad" score (max height)
    column_centroids = np.full_like(column_sums, float(height))
    
    # Find columns where tissue was actually found (m00 > 0)
    valid_columns = column_sums > 1e-6 # Use a small epsilon for float comparison
    
    # 3. Calculate centroid_y ONLY for valid columns
    column_centroids[valid_columns] = column_weighted_sums[valid_columns] / column_sums[valid_columns]

    # 4. The final metric is the *minimum* (highest) centroid_y found
    # This represents the "peak" of the foveal pit.
    if column_centroids.size > 0:
        min_centroid_y = np.min(column_centroids)
        return min_centroid_y
    else:
        # No tissue found at all in the central strip
        return float(height)


def calculate_centroid_metric(folder_path: Path, penalty: float) -> dict:
    """
    Calculates the penalized foveal pit metric for ALL 128 slices.
    """
    slice_metrics = {}
    
    # Loop over ALL 128 slices
    for i in range(1, TOTAL_SLICES + 1):
        
        file_name = f"oct_c_{i:03d}.bmp"
        file_path = folder_path / file_name
        
        if not file_path.exists():
            continue

        img = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
        
        if img is not None:
            
            # Call the tilt-robust function
            pit_metric = find_foveal_pit_metric(img)
            
            effective_metric = pit_metric
            
            # --- APPLY PENALTY ---
            if not (SEARCH_START_SLICE <= i <= SEARCH_END_SLICE):
                effective_metric += penalty
            
            slice_metrics[i] = effective_metric
            
    return slice_metrics


def find_foveal_slice_index(slice_metrics: dict) -> int:
    """
    Finds the slice index corresponding to the minimum penalized metric.
    """
    if not slice_metrics:
        return (SEARCH_START_SLICE + SEARCH_END_SLICE) // 2 

    fovea_center_slice = min(slice_metrics, key=slice_metrics.get)
    
    return fovea_center_slice


def process_all_folders(source_dir, destination_dir, adjacent_count, penalty_value):
    """
    Orchestrates the entire process: finds the foveal slice and copies the 33 slices.
    """
    print(f"--- Starting Tilt-Robust Fovea Search Process ---")
    print(f"**Critical Constraint:** Prioritizing search in slice range {SEARCH_START_SLICE} to {SEARCH_END_SLICE}")
    print(f"**Metric:** Penalized Minimum A-Scan (Column) Centroid")
    
    Path(destination_dir).mkdir(parents=True, exist_ok=True)
    
    patient_folders = [f for f in os.listdir(source_dir) if Path(source_dir, f).is_dir()]
    
    if not patient_folders:
        print(f"Error: No patient folders found in {source_dir}")
        return

    for folder_name in tqdm(patient_folders, desc="Processing Patient Folders"):
        source_folder_path = Path(source_dir) / folder_name
        dest_folder_path = Path(destination_dir) / folder_name
        
        metric_data = calculate_centroid_metric(source_folder_path, penalty_value)
        
        if not metric_data:
            tqdm.write(f"\n-> Folder '{folder_name}': SKIPPED (No valid images found)")
            continue
            
        central_slice_index = find_foveal_slice_index(metric_data)
        
        start_index = central_slice_index - adjacent_count
        end_index = central_slice_index + adjacent_count
        
        # Boundary adjustments
        if end_index > TOTAL_SLICES:
            start_index = max(1, start_index - (end_index - TOTAL_SLICES))
            end_index = TOTAL_SLICES
        if start_index < 1:
            end_index = min(TOTAL_SLICES, end_index + (1 - start_index))
            start_index = 1
            
        
        tqdm.write(f"\n-> Folder '{folder_name}': Fovea Center (Min Penalized Metric) located at slice {central_slice_index}")
        tqdm.write(f"-> Copying slices: {start_index} to {end_index} (Total {end_index - start_index + 1} slices)")
        
        dest_folder_path.mkdir(parents=True, exist_ok=True)

        copied_count = 0
        for i in range(start_index, end_index + 1):
            source_file_name = f"oct_c_{i:03d}.bmp"
            source_file_path = source_folder_path / source_file_name
            dest_file_path = dest_folder_path / source_file_name

            if source_file_path.exists():
                shutil.copy2(source_file_path, dest_file_path)
                copied_count += 1
            
        tqdm.write(f"-> Copy complete. {copied_count} files copied.")
        
    print(f"\nTilt-Robust Macula Slice Selection complete. Output is constrained to prioritize the {SEARCH_START_SLICE}-{SEARCH_END_SLICE} range.")


# --- Execution ---
if __name__ == "__main__":
    process_all_folders(SOURCE_ROOT_DIR, DESTINATION_ROOT, NUM_ADJACENT_SLICES, PENALTY_VALUE)