import glob
import torch
import imageio.v3 as iio


def load_video_frames(
    dir_path,
    device="cuda",
    dtype=torch.float32,
    max_frames=100,
    normalize=True,

):
    if str(device).startswith("cuda"):
        torch.cuda.set_device(device)
    all_paths = sorted(glob.glob(f"{dir_path}/*.png"))
    if not all_paths:
        raise RuntimeError(f"No frames found in {dir_path}")
    if len(all_paths) <= max_frames:
        paths = all_paths
    else:
        paths = all_paths[:max_frames]
    first = iio.imread(paths[0], plugin="pillow")
    H, W = first.shape[:2]

    # Preallocate CPU tensor (pinned for faster HtoD)
    vid_cpu = torch.empty((len(paths), 3, H, W), dtype=dtype, pin_memory=True)

    # Fill preallocated buffer
    for t, p in enumerate(paths):
        if t % 50 == 0:
            print(f"Loading frame {t}/{len(paths)}")
        if normalize:
            arr = iio.imread(p, plugin="pillow").astype("float32") / 255.0
        else:
            arr = iio.imread(p, plugin="pillow")
        arr = torch.from_numpy(arr)
        lin = arr.permute(2, 0, 1).contiguous()         # 3,H,W
        vid_cpu[t].copy_(lin)

    return vid_cpu.to(device=device, dtype=dtype, non_blocking=True)
