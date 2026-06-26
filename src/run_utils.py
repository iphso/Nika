import argparse
import re
import torch


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--device", default="cuda:0", help="Device, e.g. cuda:0 or 0")
    parser.add_argument("--config", default="small", help="Config name from configs.REFERENCES")
    parser.add_argument("--name", default="bunny", help="Video name or run identifier")
    parser.add_argument("--log_dir", default=None, help="Directory for run logs")
    parser.add_argument("--basedir", default="static/benchmarks", help="Base video directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Initial training batch size")
    parser.add_argument("--operator_steps", type=int, default=None, help="Number of operator steps (overrides config)")
    return parser


def resolve_device(device: str) -> str:
    if re.fullmatch(r"\d+", device):
        return f"cuda:{device}"
    if re.fullmatch(r"cuda\d+", device):
        return device.replace("cuda", "cuda:")
    return device


def setup_torch(seed: int = 42) -> None:
    torch.manual_seed(seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
