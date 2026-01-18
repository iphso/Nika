"""
Visualize reconstruction errors for Nika models.

This script analyzes the quality of video reconstruction by a trained NikaBlock
neural implicit representation (INR) model. It generates two types of visualizations:

1. ERROR OVER TIME PLOT:
   Shows how reconstruction quality varies across frames. This helps identify:
   - Temporal regions where the model struggles (e.g., high-motion scenes)
   - Whether error is consistent or varies significantly across the video
   - The relationship between MSE and PSNR (they're inversely related)

2. SPATIAL ERROR HEATMAP:
   Shows average reconstruction error at each pixel position across all frames.
   This reveals:
   - Spatial regions the model finds difficult (e.g., edges, textures)
   - Whether errors concentrate in specific areas (e.g., corners, center)
   - Potential model capacity issues (uniform high error = underfitting)

BACKGROUND ON METRICS:
- MSE (Mean Squared Error): Average of squared differences between predicted
  and ground truth pixel values. Lower is better. Range: [0, 1] for normalized images.
- PSNR (Peak Signal-to-Noise Ratio): Logarithmic measure of reconstruction quality.
  PSNR = 10 * log10(1 / MSE). Higher is better. Typical range: 20-40 dB.
  - 30 dB: Acceptable quality
  - 35 dB: Good quality
  - 40+ dB: Excellent quality (often indistinguishable from original)

USAGE:
    python visualize_errors.py <model_path> [--output-dir OUTPUT_DIR] [--batch-size N]

EXAMPLE:
    python visualize_errors.py models/small-beauty-epoch1992-psnr33.31.torch

The model filename encodes metadata: {config}-{video}-epoch{N}-psnr{X.XX}.torch
- config: Model size configuration (xxs, xs, small, medium, large)
- video: Name of the video the model was trained on
- epoch: Training epoch when checkpoint was saved
- psnr: PSNR achieved at that epoch (for quick reference)
"""

# =============================================================================
# IMPORTS
# =============================================================================

import argparse  # CLI argument parsing
import os        # File path operations
import re        # Regular expressions for filename parsing

import matplotlib.pyplot as plt  # Plotting library for visualizations
import torch                     # PyTorch for tensor operations and model loading
import torch.nn.functional as F  # Functional API for loss computation (mse_loss)

# Project-specific imports:
# - REFERENCES: Dictionary mapping config names to model hyperparameters
#   (grid_ranks, tucker_ranks, conv_hidden, etc.)
# - load_video_frames: Utility to load PNG frames from a directory into a tensor
# - NikaBlock: The neural implicit representation model architecture
from configs import REFERENCES
from load_data import load_video_frames
from nika import NikaBlock


# =============================================================================
# FILENAME PARSING
# =============================================================================

def parse_model_filename(path: str) -> tuple[str, str]:
    """
    Extract config name and video name from model filename.

    The Nika training pipeline saves models with a standardized filename format:
        {config}-{video}-epoch{N}-psnr{X.XX}.torch

    This function parses that format to automatically determine:
    1. Which model configuration to use (affects architecture hyperparameters)
    2. Which video to load for comparison (must match training video)

    Args:
        path: Full or relative path to the .torch checkpoint file
              Example: "models/small-beauty-epoch1992-psnr33.31.torch"

    Returns:
        Tuple of (config, video_name):
        - config: One of "xxs", "xs", "small", "medium", "large"
        - video_name: Name of the video directory (e.g., "beauty", "bunny")

    Raises:
        ValueError: If filename doesn't match expected pattern

    Example:
        >>> parse_model_filename("models/small-beauty-epoch1992-psnr33.31.torch")
        ('small', 'beauty')
    """
    # Extract just the filename, ignoring directory path
    basename = os.path.basename(path)

    # Regex breakdown:
    # ^(\w+)     - Start of string, capture config name (word characters)
    # -(\w+)     - Literal dash, capture video name (word characters)
    # -epoch\d+  - Literal "-epoch" followed by digits (not captured)
    # -psnr[\d.]+- Literal "-psnr" followed by digits/decimals (not captured)
    # \.torch$   - Literal ".torch" at end of string
    match = re.match(r"^(\w+)-(\w+)-epoch\d+-psnr[\d.]+\.torch$", basename)

    if not match:
        raise ValueError(
            f"Could not parse model filename: {basename}. "
            "Expected pattern: {config}-{video}-epoch{N}-psnr{X.XX}.torch"
        )

    # match.group(1) = first captured group (config)
    # match.group(2) = second captured group (video name)
    return match.group(1), match.group(2)


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model(path: str, vid_shape: tuple, config: str, device: str) -> NikaBlock:
    """
    Load a trained NikaBlock model from a checkpoint file.

    NikaBlock is a neural implicit representation that learns to map:
        (x, y, t) coordinates -> RGB pixel values

    The model architecture is determined by the config name, which selects
    hyperparameters from the REFERENCES dictionary in configs.py. Different
    configs trade off between model size and reconstruction quality.

    IMPORTANT: The model must be initialized with the same target_shape it was
    trained with. This is why we need the video shape - the internal coordinate
    grids are sized to match the video dimensions.

    Args:
        path: Path to .torch checkpoint file containing model state_dict
        vid_shape: Shape of video tensor as (T, C, H, W) where:
                   - T = number of frames (temporal dimension)
                   - C = channels (always 3 for RGB)
                   - H = height in pixels
                   - W = width in pixels
        config: Config name determining model architecture hyperparameters.
                Must be one of: "xxs", "xs", "small", "medium", "large"
        device: PyTorch device string, e.g., "cuda:0" or "cpu"

    Returns:
        NikaBlock model loaded with trained weights, set to eval mode

    Raises:
        ValueError: If config name is not recognized

    Note on target_shape format:
        NikaBlock expects [k, H, W, T] where k=4 is a fixed hyperparameter
        related to the internal feature grid structure. This differs from
        the standard PyTorch video format (T, C, H, W).
    """
    # Validate config name before attempting to use it
    if config not in REFERENCES:
        raise ValueError(f"Unknown config: {config}. Valid: {list(REFERENCES.keys())}")

    # Unpack video dimensions
    # T = temporal (frames), C = channels, H = height, W = width
    T, C, H, W = vid_shape

    # Initialize model architecture
    # - target_shape: [k, H, W, T] format required by NikaBlock internals
    # - k=4: Fixed hyperparameter for feature grid structure
    # - **REFERENCES[config]: Unpacks architecture hyperparameters:
    #   - grid_ranks: Rank of the learned feature grid tensor decomposition
    #   - real_tucker_ranks: Ranks for real-valued Tucker decomposition
    #   - complex_tucker_ranks: Ranks for complex-valued Tucker decomposition
    #   - conv_hidden: Hidden dimension for upsampling CNN
    # - out_channels=3: RGB output
    model = NikaBlock(
        target_shape=[4, H, W, T],
        k=4,
        **REFERENCES[config],
        out_channels=3,
        device=device,
    )

    # Load trained weights from checkpoint
    # map_location ensures weights are loaded to the correct device
    # (important when loading GPU-trained model on CPU or different GPU)
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)

    # Set to evaluation mode:
    # - Disables dropout (if any)
    # - Sets batch normalization to use running statistics (if any)
    # - Required for deterministic inference
    model.eval()

    return model


# =============================================================================
# ERROR COMPUTATION
# =============================================================================

def compute_errors(
    model: NikaBlock,
    video: torch.Tensor,
    device: str,
    batch_size: int = 10,
) -> tuple[list[float], list[float], torch.Tensor]:
    """
    Compute per-frame and per-pixel reconstruction errors.

    This function is the core of the error analysis. It:
    1. Iterates through all video frames in batches (for memory efficiency)
    2. Generates model predictions for each frame
    3. Compares predictions to ground truth using MSE and PSNR
    4. Accumulates spatial error statistics across all frames

    BATCHING RATIONALE:
    Processing all frames at once would require too much GPU memory for
    high-resolution videos. Batching allows processing videos of any length
    while keeping memory usage bounded.

    Args:
        model: Trained NikaBlock model (should be in eval mode)
        video: Ground truth video tensor with shape (T, C, H, W)
               - Values should be uint8 in range [0, 255]
               - Will be normalized to [0, 1] for comparison
        device: PyTorch device for computation
        batch_size: Number of frames to process simultaneously
                    - Larger = faster but more memory
                    - Default 10 works well for 1080p on 8GB GPU

    Returns:
        Tuple of (per_frame_mse, per_frame_psnr, spatial_mse):

        per_frame_mse: List[float] of length T
            MSE value for each frame. Use this to identify which frames
            have high reconstruction error.

        per_frame_psnr: List[float] of length T
            PSNR value for each frame in dB. Logarithmic scale makes it
            easier to compare quality across different error magnitudes.

        spatial_mse: Tensor of shape (H, W)
            Average MSE at each pixel position across all frames.
            High values indicate spatial regions the model struggles with.

    Implementation Notes:
        - torch.no_grad() disables gradient computation for efficiency
        - Predictions are clamped to [0, 1] to match valid pixel range
        - Spatial MSE is averaged over channels before accumulation
    """
    # Extract dimensions from video tensor
    num_frames = video.shape[0]  # T dimension
    H, W = video.shape[2], video.shape[3]  # Spatial dimensions

    # Initialize accumulators for per-frame metrics
    per_frame_mse = []   # Will have num_frames entries
    per_frame_psnr = []  # Will have num_frames entries

    # Accumulator for spatial error heatmap
    # We sum MSE at each pixel across all frames, then divide by num_frames
    spatial_mse_sum = torch.zeros((H, W), device=device, dtype=torch.float32)

    # Calculate number of batches needed (ceiling division)
    num_batches = (num_frames + batch_size - 1) // batch_size

    # Disable gradient computation - we're only doing inference
    # This saves memory and computation time
    with torch.no_grad():
        for batch_idx in range(num_batches):
            # Calculate frame indices for this batch
            min_t = batch_idx * batch_size          # First frame index (inclusive)
            max_t = min((batch_idx + 1) * batch_size, num_frames)  # Last frame (exclusive)

            # Load and normalize ground truth frames for this batch
            # - video[min_t:max_t]: Slice batch of frames, shape (batch, C, H, W)
            # - .to(torch.float32): Convert from uint8 to float for computation
            # - / 255.0: Normalize from [0, 255] to [0, 1] range
            batch_gt = video[min_t:max_t].to(torch.float32) / 255.0

            # Create temporal indices for model input
            # NikaBlock takes frame indices and returns corresponding RGB frames
            # - torch.arange creates [min_t, min_t+1, ..., max_t-1]
            # - dtype=int64 required by model's internal coordinate encoding
            t_batch = torch.arange(min_t, max_t, device=device, dtype=torch.int64)

            # Generate predictions from model
            # - model(t_batch): Forward pass, returns shape (batch, C, H, W)
            # - .clamp(0, 1): Ensure predictions are in valid pixel range
            #   (network output might slightly exceed [0,1] due to activations)
            prediction = model(t_batch).clamp(0, 1)

            # Process each frame in the batch individually
            # (needed for per-frame metrics and spatial accumulation)
            for i in range(prediction.shape[0]):
                frame_idx = min_t + i  # Absolute frame index in video
                pred_frame = prediction[i]  # Shape: (C, H, W)
                gt_frame = batch_gt[i]      # Shape: (C, H, W)

                # Compute Mean Squared Error for this frame
                # F.mse_loss computes: mean((pred - gt)^2) over all elements
                # Result is a scalar tensor
                mse = F.mse_loss(pred_frame, gt_frame)

                # Convert MSE to PSNR (Peak Signal-to-Noise Ratio)
                # Formula: PSNR = 10 * log10(MAX^2 / MSE)
                # For normalized images, MAX = 1, so: PSNR = 10 * log10(1 / MSE)
                # The 1e-8 prevents log(0) when MSE is very small
                psnr = 10 * torch.log10(1 / (mse + 1e-8))

                # Store scalar values (convert from tensor to Python float)
                per_frame_mse.append(mse.item())
                per_frame_psnr.append(psnr.item())

                # Compute per-pixel MSE for spatial heatmap
                # - (pred_frame - gt_frame) ** 2: Squared error at each pixel, shape (C, H, W)
                # - .mean(dim=0): Average over channels, resulting shape (H, W)
                # This gives us the average squared error at each spatial position
                pixel_mse = ((pred_frame - gt_frame) ** 2).mean(dim=0)

                # Accumulate into spatial sum (will divide by num_frames later)
                spatial_mse_sum += pixel_mse

            # Progress indicator every 100 frames
            if min_t % 100 == 0:
                print(f"Processed frames {min_t}-{max_t - 1}")

    # Compute average spatial MSE across all frames
    # spatial_mse_sum contains sum of per-pixel MSE; divide by frame count for average
    spatial_mse = spatial_mse_sum / num_frames

    return per_frame_mse, per_frame_psnr, spatial_mse


# =============================================================================
# VISUALIZATION: ERROR OVER TIME
# =============================================================================

def plot_error_over_time(
    per_frame_mse: list[float],
    per_frame_psnr: list[float],
    output_path: str,
    title: str,
) -> None:
    """
    Create a dual-axis plot showing MSE and PSNR over frame index.

    This visualization helps identify temporal patterns in reconstruction quality:
    - Spikes indicate frames the model struggles with (often high-motion)
    - Trends reveal whether quality degrades over time
    - Comparison of MSE and PSNR shows how logarithmic scaling affects perception

    WHY DUAL AXES:
    MSE and PSNR have inverse relationship and different scales:
    - MSE: Linear scale, lower is better, range ~[0.0001, 0.01]
    - PSNR: Logarithmic scale, higher is better, range ~[20, 50] dB
    Plotting both on same axes would make one unreadable.

    Args:
        per_frame_mse: List of MSE values, one per frame
        per_frame_psnr: List of PSNR values in dB, one per frame
        output_path: Where to save the PNG file
        title: Plot title (typically includes config and video name)

    Output:
        Saves a PNG file with:
        - Red line: MSE values (left y-axis)
        - Blue line: PSNR values (right y-axis)
        - Dashed horizontal line: Average PSNR with annotation
    """
    # Create figure with specified size (12 inches wide, 6 inches tall)
    # Returns figure and primary axes objects
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # X-axis: frame indices (0, 1, 2, ..., num_frames-1)
    frames = list(range(len(per_frame_mse)))

    # --- LEFT Y-AXIS: MSE (red) ---
    color_mse = "tab:red"  # Matplotlib's red from default color cycle
    ax1.set_xlabel("Frame Index")
    ax1.set_ylabel("MSE", color=color_mse)
    ax1.plot(
        frames,
        per_frame_mse,
        color=color_mse,
        alpha=0.7,       # Slight transparency for visual appeal
        linewidth=0.8    # Thin line to show detail without overwhelming
    )
    ax1.tick_params(axis="y", labelcolor=color_mse)  # Color y-tick labels to match

    # --- RIGHT Y-AXIS: PSNR (blue) ---
    # twinx() creates a second y-axis sharing the same x-axis
    ax2 = ax1.twinx()
    color_psnr = "tab:blue"
    ax2.set_ylabel("PSNR (dB)", color=color_psnr)
    ax2.plot(
        frames,
        per_frame_psnr,
        color=color_psnr,
        alpha=0.7,
        linewidth=0.8
    )
    ax2.tick_params(axis="y", labelcolor=color_psnr)

    # --- AVERAGE PSNR REFERENCE LINE ---
    # Horizontal dashed line helps viewer quickly assess overall quality
    avg_psnr = sum(per_frame_psnr) / len(per_frame_psnr)
    ax2.axhline(
        y=avg_psnr,
        color=color_psnr,
        linestyle="--",  # Dashed line
        alpha=0.5        # Semi-transparent to not dominate
    )

    # Add text annotation showing the average value
    ax2.annotate(
        f"Avg: {avg_psnr:.2f} dB",
        xy=(len(frames) * 0.95, avg_psnr),  # Position near right side
        ha="right",    # Horizontal alignment
        va="bottom",   # Vertical alignment (above the line)
        color=color_psnr,
    )

    # --- FINALIZE AND SAVE ---
    plt.title(title)
    fig.tight_layout()  # Adjust spacing to prevent label overlap
    plt.savefig(output_path, dpi=150)  # 150 DPI = good balance of quality/size
    plt.close()  # Release memory

    print(f"Saved error-over-time plot to {output_path}")


# =============================================================================
# VISUALIZATION: SPATIAL ERROR HEATMAP
# =============================================================================

def plot_spatial_error_heatmap(
    spatial_mse: torch.Tensor,
    output_path: str,
    title: str,
) -> None:
    """
    Create a heatmap showing average MSE at each pixel position.

    This visualization reveals spatial patterns in reconstruction error:
    - Bright regions (high error) often correspond to:
      - High-frequency textures (grass, hair, fabric patterns)
      - Motion boundaries and edges
      - Fine details the model lacks capacity to represent
    - Dark regions (low error) indicate areas the model handles well:
      - Smooth gradients
      - Static backgrounds
      - Large uniform regions

    INTERPRETATION TIPS:
    - Uniform brightness = model capacity is the limiting factor
    - Bright edges = model struggles with sharp transitions
    - Bright in motion areas = temporal modeling limitations

    Args:
        spatial_mse: Tensor of shape (H, W) containing average MSE at each pixel
                     Values are typically in range [0.0001, 0.01]
        output_path: Where to save the PNG file
        title: Plot title (typically includes config and video name)

    Output:
        Saves a PNG file with:
        - Heatmap where brightness indicates error magnitude
        - "Hot" colormap: black (low) -> red -> yellow -> white (high)
        - Colorbar showing MSE value scale
    """
    # Convert from PyTorch tensor to NumPy array for matplotlib
    # .cpu() moves tensor to CPU if it was on GPU
    # .numpy() converts to NumPy array
    spatial_mse_np = spatial_mse.cpu().numpy()

    # Create figure - wider than tall to match typical video aspect ratio
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create heatmap visualization
    # - imshow displays 2D array as an image
    # - cmap="hot": Black -> Red -> Yellow -> White colormap
    #   (intuitive: hotter = higher error)
    # - aspect="auto": Stretch to fill axes (vs "equal" which preserves pixel aspect)
    im = ax.imshow(spatial_mse_np, cmap="hot", aspect="auto")

    # Add colorbar to show MSE value scale
    # - Positioned to the right of the main plot
    # - Label explains what the colors represent
    cbar = fig.colorbar(im, ax=ax, label="Mean Squared Error")

    # Add labels
    ax.set_title(title)
    ax.set_xlabel("Width (pixels)")
    ax.set_ylabel("Height (pixels)")

    # Finalize and save
    fig.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Saved spatial heatmap to {output_path}")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """
    Main entry point for the error visualization script.

    This function orchestrates the entire pipeline:
    1. Parse command-line arguments
    2. Set up PyTorch device and optimizations
    3. Parse model filename to determine config and video
    4. Load video frames from disk
    5. Load trained model
    6. Compute reconstruction errors
    7. Generate and save visualizations

    The script is designed to work within a Docker container where:
    - Video frames are mounted at /app/static/benchmarks/{video_name}/
    - Models are stored at /app/models/
    - Output plots go to /app/error_plots/
    """

    # -------------------------------------------------------------------------
    # ARGUMENT PARSING
    # -------------------------------------------------------------------------

    parser = argparse.ArgumentParser(
        description="Visualize reconstruction errors for Nika models"
    )

    # Required: path to model checkpoint
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to model checkpoint (.torch file)",
    )

    # Optional: output directory for plots
    parser.add_argument(
        "--output-dir",
        type=str,
        default="error_plots",
        help="Directory to save output plots (default: error_plots)",
    )

    # Optional: batch size for memory management
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size for processing frames (default: 10)",
    )

    # Optional: limit number of frames (useful for quick tests)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=600,
        help="Maximum number of frames to process (default: 600)",
    )

    # Optional: video directory (for non-standard setups)
    parser.add_argument(
        "--video-dir",
        type=str,
        default="static/benchmarks",
        help="Base directory for video frames (default: static/benchmarks)",
    )

    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # DEVICE SETUP AND OPTIMIZATIONS
    # -------------------------------------------------------------------------

    # Select GPU if available, otherwise fall back to CPU
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Enable TensorFloat-32 (TF32) for faster matrix operations on Ampere+ GPUs
    # TF32 uses 19-bit precision (vs FP32's 23-bit) for ~3x speedup
    # Accuracy impact is negligible for inference
    torch.set_float32_matmul_precision("high")
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # -------------------------------------------------------------------------
    # PARSE MODEL FILENAME
    # -------------------------------------------------------------------------

    # Extract config (model size) and video name from filename
    # This determines which architecture to instantiate and which video to load
    config, video_name = parse_model_filename(args.model_path)
    print(f"Config: {config}, Video: {video_name}")

    # -------------------------------------------------------------------------
    # LOAD VIDEO
    # -------------------------------------------------------------------------

    # Construct path to video frames directory
    # Videos are stored as directories of PNG frames: {video_dir}/{video_name}/*.png
    video_path = os.path.join(args.video_dir, video_name)
    print(f"Loading video from {video_path}...")

    # Load video frames into a tensor
    # - device: Load directly to GPU if available
    # - max_frames: Limit frames for memory/time constraints
    # - dtype=uint8: Keep as bytes until needed (memory efficient)
    # - normalize=False: We'll normalize during error computation
    video = load_video_frames(
        video_path,
        device,
        max_frames=args.max_frames,
        dtype=torch.uint8,
        normalize=False,
    )
    print(f"Video shape: {video.shape}")  # Expected: (T, 3, H, W)

    # -------------------------------------------------------------------------
    # LOAD MODEL
    # -------------------------------------------------------------------------

    print(f"Loading model from {args.model_path}...")

    # Initialize model architecture and load trained weights
    # Model architecture is determined by config; video shape sets coordinate grid size
    model = load_model(args.model_path, video.shape, config, device)

    # -------------------------------------------------------------------------
    # COMPUTE ERRORS
    # -------------------------------------------------------------------------

    print("Computing errors...")

    # Run inference on all frames and compute error metrics
    per_frame_mse, per_frame_psnr, spatial_mse = compute_errors(
        model, video, device, args.batch_size
    )

    # Print summary statistics
    avg_psnr = sum(per_frame_psnr) / len(per_frame_psnr)
    avg_mse = sum(per_frame_mse) / len(per_frame_mse)
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    print(f"Average MSE: {avg_mse:.6f}")

    # -------------------------------------------------------------------------
    # GENERATE VISUALIZATIONS
    # -------------------------------------------------------------------------

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Error Over Time Plot ---
    # Filename format: {config}_{video}_error_over_time.png
    time_plot_path = os.path.join(
        args.output_dir, f"{config}_{video_name}_error_over_time.png"
    )
    plot_error_over_time(
        per_frame_mse,
        per_frame_psnr,
        time_plot_path,
        f"Reconstruction Error Over Time - {config}/{video_name}",
    )

    # --- Spatial Error Heatmap ---
    # Filename format: {config}_{video}_spatial_heatmap.png
    heatmap_path = os.path.join(
        args.output_dir, f"{config}_{video_name}_spatial_heatmap.png"
    )
    plot_spatial_error_heatmap(
        spatial_mse,
        heatmap_path,
        f"Spatial Error Heatmap - {config}/{video_name}",
    )

    print("Done!")


# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================

# Standard Python idiom: only run main() if this file is executed directly
# (not when imported as a module)
if __name__ == "__main__":
    main()
