"""
Generate residual videos for Nika models.

This script creates a side-by-side comparison video showing:
- Ground Truth: Original video frames
- Prediction: Model reconstruction
- Residual: Absolute difference (amplified for visibility)

The residual visualization helps identify:
- Temporal patterns in reconstruction error
- Which parts of the frame are hardest to reconstruct
- Correlation with motion, textures, or edges

USAGE:
    python generate_residual_video.py <model_path> [--output-dir OUTPUT_DIR] [--fps FPS]

EXAMPLE:
    python generate_residual_video.py models/small-beauty-epoch1992-psnr33.31.torch
"""

import argparse
import os

import imageio
import numpy as np
import torch

from configs import REFERENCES
from load_data import load_video_frames
from nika import NikaBlock
from visualize_errors import parse_model_filename


def load_model(path: str, vid_shape: tuple, config: str, device: str) -> NikaBlock:
    """
    Load a trained NikaBlock model from a checkpoint file.

    Args:
        path: Path to .torch checkpoint file
        vid_shape: Shape of video tensor as (T, C, H, W)
        config: Config name (xxs, xs, small, medium, large)
        device: PyTorch device string

    Returns:
        NikaBlock model loaded with trained weights, in eval mode
    """
    if config not in REFERENCES:
        raise ValueError(f"Unknown config: {config}. Valid: {list(REFERENCES.keys())}")

    T, C, H, W = vid_shape

    model = NikaBlock(
        target_shape=[4, H, W, T],
        k=4,
        **REFERENCES[config],
        out_channels=3,
        device=device,
    )

    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    return model


def add_label_to_frame(frame: np.ndarray, labels: list[str]) -> np.ndarray:
    """
    Add text labels to the top of each panel of a single frame.

    Args:
        frame: Single frame as numpy array (H, W, C), uint8
        labels: List of label strings for each panel

    Returns:
        Frame with labels overlaid
    """
    H, W, C = frame.shape
    panel_width = W // len(labels)
    label_height = 30
    font_scale = 0.8

    try:
        import cv2

        for i, label in enumerate(labels):
            x_start = i * panel_width
            # Draw semi-transparent background rectangle
            overlay = frame.copy()
            cv2.rectangle(
                overlay,
                (x_start, 0),
                (x_start + panel_width, label_height),
                (0, 0, 0),
                -1,
            )
            frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

            # Draw text
            text_x = x_start + 10
            text_y = 22
            cv2.putText(
                frame,
                label,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                2,
            )
    except ImportError:
        pass

    return frame


def generate_and_save_video(
    model: NikaBlock,
    video: torch.Tensor,
    device: str,
    output_path: str,
    fps: int = 30,
    batch_size: int = 10,
    amplification: float = 5.0,
) -> None:
    """
    Generate comparison frames and write directly to video file.

    Streams frames to disk to avoid memory issues with large videos.

    Args:
        model: Trained NikaBlock model
        video: Ground truth video tensor (T, C, H, W), uint8 [0, 255]
        device: PyTorch device
        output_path: Path to save the video
        fps: Frames per second
        batch_size: Frames to process at once
        amplification: Factor to amplify residuals for visibility
    """
    num_frames = video.shape[0]
    C, H, W = video.shape[1], video.shape[2], video.shape[3]

    labels = ["Ground Truth", "Prediction", f"Residual ({int(amplification)}x)"]
    num_batches = (num_frames + batch_size - 1) // batch_size

    # Open video writer (using imageio's ffmpeg backend)
    writer = imageio.get_writer(output_path, fps=fps, codec="libx264")

    try:
        with torch.no_grad():
            for batch_idx in range(num_batches):
                min_t = batch_idx * batch_size
                max_t = min((batch_idx + 1) * batch_size, num_frames)

                # Normalize ground truth to [0, 1]
                batch_gt = video[min_t:max_t].to(torch.float32) / 255.0

                # Generate predictions
                t_batch = torch.arange(min_t, max_t, device=device, dtype=torch.int64)
                prediction = model(t_batch).clamp(0, 1)

                # Compute residual (absolute difference, amplified)
                residual = (prediction - batch_gt).abs() * amplification
                residual = residual.clamp(0, 1)

                # Process each frame in batch
                for i in range(prediction.shape[0]):
                    gt_frame = batch_gt[i].cpu().numpy()
                    pred_frame = prediction[i].cpu().numpy()
                    res_frame = residual[i].cpu().numpy()

                    # Concatenate horizontally: (C, H, W) -> (C, H, W*3)
                    combined = np.concatenate([gt_frame, pred_frame, res_frame], axis=2)

                    # Convert (C, H, W) float [0,1] -> (H, W, C) uint8 [0,255]
                    frame = (combined * 255).astype(np.uint8)
                    frame = frame.transpose(1, 2, 0)

                    # Add labels
                    frame = add_label_to_frame(frame, labels)

                    # Write frame
                    writer.append_data(frame)

                if min_t % 100 == 0:
                    print(f"Processed frames {min_t}-{max_t - 1}")
    finally:
        writer.close()

    print(f"Saved video to {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate residual comparison video for Nika models"
    )

    parser.add_argument(
        "model_path",
        type=str,
        help="Path to model checkpoint (.torch file)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="visuals",
        help="Directory to save output video (default: visuals)",
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second for output video (default: 30)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size for processing frames (default: 10)",
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=600,
        help="Maximum number of frames to process (default: 600)",
    )

    parser.add_argument(
        "--video-dir",
        type=str,
        default="static/benchmarks",
        help="Base directory for video frames (default: static/benchmarks)",
    )

    parser.add_argument(
        "--amplification",
        type=float,
        default=5.0,
        help="Factor to amplify residuals for visibility (default: 5.0)",
    )

    args = parser.parse_args()

    # Device setup
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    torch.set_float32_matmul_precision("high")
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Parse model filename
    config, video_name = parse_model_filename(args.model_path)
    print(f"Config: {config}, Video: {video_name}")

    # Load video
    video_path = os.path.join(args.video_dir, video_name)
    print(f"Loading video from {video_path}...")

    video = load_video_frames(
        video_path,
        device,
        max_frames=args.max_frames,
        dtype=torch.uint8,
        normalize=False,
    )
    print(f"Video shape: {video.shape}")

    # Load model
    print(f"Loading model from {args.model_path}...")
    model = load_model(args.model_path, video.shape, config, device)

    # Generate and save video (streaming to avoid memory issues)
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"{config}_{video_name}_comparison.mp4")

    print("Generating comparison video...")
    generate_and_save_video(
        model,
        video,
        device,
        output_path,
        fps=args.fps,
        batch_size=args.batch_size,
        amplification=args.amplification,
    )

    print("Done!")


if __name__ == "__main__":
    main()
