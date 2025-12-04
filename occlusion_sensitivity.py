import argparse
import os
from collections import defaultdict

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose, CenterCrop
from tqdm import tqdm

import face_alignment

from data.transforms import NormalizeVideo, ToTensorVideo
from models.spatiotemporal_net import get_model
from preprocessing.crop_mouths import crop_video


# -------------------------
# Video reading (OpenCV)
# -------------------------

def read_video_cv2(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # BGR → RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 25.0  # fallback

    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"No frames read from video: {path}")

    video = np.stack(frames, axis=0)  # (T, H, W, C)
    return video, fps


# -------------------------
# Occlusion helpers
# -------------------------

@torch.no_grad()
def compute_clip_sensitivity(
    model,
    clip,              # [1, T, 1, H, W] on device
    base_logit,        # scalar float
    patch_size,
    patch_stride,
    device,
    frames_per_clip,
):
    """
    Per-clip spatial occlusion sensitivity.

    Returns:
        heatmap: [T, H, W] tensor on CPU
    """
    _, T, _, H, W = clip.shape
    assert T == frames_per_clip, f"Expected {frames_per_clip} frames, got {T}"

    # Accumulate importance + counts for averaging overlapping patches
    heat = torch.zeros((T, H, W), device=device)
    counts = torch.zeros((T, H, W), device=device)

    for t in tqdm(range(T)):
        for y in range(0, H - patch_size + 1, patch_stride):
            for x in range(0, W - patch_size + 1, patch_stride):
                occluded = clip.clone()
                occluded[0, t, 0, y:y+patch_size, x:x+patch_size] = 0.0

                logit_occ = model(occluded.permute(0, 2, 1, 3, 4), lengths=[frames_per_clip])[0].item()
                delta = base_logit - logit_occ

                heat[t, y:y+patch_size, x:x+patch_size] += delta
                counts[t, y:y+patch_size, x:x+patch_size] += 1

    counts = torch.where(counts == 0, torch.ones_like(counts), counts)
    heat = heat / counts
    heat = torch.clamp(heat, min=0)  # keep positive contributions

    return heat.detach().cpu()


def normalize_heatmap_per_video(heat):
    """
    heat: [T, H, W] tensor on CPU
    Returns: [T, H, W] normalized to [0, 1] per whole video
    """
    heat_np = heat.numpy().astype(np.float32)
    max_val = heat_np.max()
    min_val = heat_np.min()
    if max_val > min_val:
        norm = (heat_np - min_val) / (max_val - min_val + 1e-8)
    else:
        norm = np.zeros_like(heat_np, dtype=np.float32)
    return norm


def overlay_heatmap_on_frames(frames_gray, heatmaps, alpha=0.6):
    """
    frames_gray: [T, H, W] numpy in [0,1] (mouth crops, grayscale)
    heatmaps:    [T, H, W] numpy in [0,1]
    Returns: list of [H, W, 3] uint8 frames
    """
    T, H, W = frames_gray.shape
    cmap = plt.get_cmap("jet")

    out_frames = []
    for t in range(T):
        frame = frames_gray[t]
        hm = heatmaps[t]

        hm_color = cmap(hm)[:, :, :3]  # [H, W, 3], float [0,1]
        frame_rgb = np.stack([frame, frame, frame], axis=-1)  # [H, W, 3]

        overlay = (1 - alpha) * frame_rgb + alpha * hm_color
        overlay = np.clip(overlay * 255.0, 0, 255).astype(np.uint8)
        out_frames.append(overlay)

    return out_frames



def save_video(frames_rgb, fps, output_path):
    """
    frames_rgb: list or array of RGB frames (H, W, 3)
    fps: float (e.g. 30.0)
    output_path: str, e.g. 'occlusion_overlay.mp4'
    """
    if not frames_rgb:
        raise ValueError("No frames provided to save_video.")

    h, w, _ = frames_rgb[0].shape
    # OpenCV expects BGR frames and a codec specification
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for frame in frames_rgb:
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    out.release()
    print(f"✅ Saved video to {output_path} ({fps:.1f} fps, {w}x{h})")


# -------------------------
# Main
# -------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Occlusion sensitivity for LipForensics")

    parser.add_argument("--video", type=str, required=True,
                        help="Path to input video (full face)")
    parser.add_argument("--frames_per_clip", type=int, default=25,
                        help="Number of frames per clip (must match model)")
    parser.add_argument("--stride", type=int, default=25,
                        help="Temporal stride between clips (use 25 for non-overlapping)")
    parser.add_argument("--patch_size", type=int, default=11,
                        help="Spatial occlusion patch size (square)")
    parser.add_argument("--patch_stride", type=int, default=5,
                        help="Stride between occlusion patches")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device for the model, e.g. 'cuda:0' or 'cpu'")
    parser.add_argument("--weights_forgery_path", type=str,
                        default="./models/weights/lipforensics_ff.pth",
                        help="Path to pretrained forgery detection weights")

    return parser.parse_args()


def main():
    args = parse_args()

    # ---- Device selection ----
    if "cuda" in args.device:
        if torch.cuda.is_available():
            device = torch.device(args.device)
        else:
            print("CUDA requested but not available, falling back to CPU.")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # ---- Load model ----
    model = get_model(weights_forgery_path=args.weights_forgery_path, device=device)
    model.eval()
    print("Face forgery weights loaded.")

    # ---- Face alignment ----
    fa_device = "cuda" if ("cuda" in args.device and torch.cuda.is_available()) else "cpu"
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D,
                                      device=fa_device,
                                      flip_input=False)

    # ---- Transforms (same as eval) ----
    mean = (0.421,)
    std = (0.165,)
    transform = Compose([
        ToTensorVideo(),
        CenterCrop((88, 88)),
        NormalizeVideo(mean, std),
    ])

    # ---- Read video with OpenCV ----
    print(f"Reading video (OpenCV): {args.video}")
    video_np, fps = read_video_cv2(args.video)  # [T, H, W, C], uint8
    T_total = video_np.shape[0]
    frames_per_clip = args.frames_per_clip

    adjusted_length = (T_total // frames_per_clip) * frames_per_clip
    if adjusted_length == 0:
        raise ValueError(f"Video too short: {T_total} frames, need at least {frames_per_clip}")

    video_np = video_np[:adjusted_length]  # [T_adj, H, W, C]
    T_adj = video_np.shape[0]

    # ---- IMPORTANT: match original crop_mouths expectations ----
    # crop_video originally received a torch tensor from torchvision.io.read_video
    # with shape [T, H, W, C]. So we mimic that exactly.
    video_torch = torch.from_numpy(video_np)  # [T, H, W, C], uint8
    print("Cropping mouths with face_alignment...")
    vid_cropped = crop_video(video_torch, fa)  # returns [T_adj, Hc, Wc, C] (numpy or tensor)

    # Ensure numpy before transforms
    if hasattr(vid_cropped, "numpy"):
        vid_cropped = vid_cropped.numpy()
    vid_cropped = vid_cropped.astype(np.uint8)

    # ---- Apply model transform (ToTensorVideo -> grayscale, crop, normalize) ----
    cropped_video = transform(torch.tensor(vid_cropped))  # [T_adj, 1, 88, 88]
    T_adj, C, H, W = cropped_video.shape
    assert C == 1 and H == 88 and W == 88, f"Unexpected mouth crop shape: {cropped_video.shape}"

    # Save unnormalized grayscale for visualization
    mean_t = torch.tensor(mean, dtype=torch.float32).view(1, 1, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32).view(1, 1, 1, 1)
    cropped_unnorm = (cropped_video * std_t) + mean_t
    cropped_unnorm = torch.clamp(cropped_unnorm, 0.0, 1.0)  # [T, 1, 88, 88]

    # ---- Build clips: [num_clips, T, C, H, W] ----
    num_clips = T_adj // frames_per_clip
    video_clips = cropped_video.view(num_clips, frames_per_clip, C, H, W).to(device)  # [N, 25, 1, 88, 88]

    # ---- Base logits per clip ----
    with torch.no_grad():
        base_logits = model(video_clips.permute(0, 2, 1, 3, 4), lengths=[frames_per_clip] * num_clips)  # [N]
        base_logits = base_logits.view(num_clips).cpu()

    print("Base logits per clip:", base_logits.numpy())
    print("Mean logit:", base_logits.mean().item())

    # ---- Occlusion sensitivity per clip ----
    all_heatmaps = []
    for i in range(num_clips):
        print(f"Computing occlusion for clip {i+1}/{num_clips}...")
        clip = video_clips[i:i+1]  # [1, T, 1, 88, 88]

        heat = compute_clip_sensitivity(
            model=model,
            clip=clip,
            base_logit=base_logits[i].item(),
            patch_size=args.patch_size,
            patch_stride=args.patch_stride,
            device=device,
            frames_per_clip=frames_per_clip,
        )  # [T, H, W] on CPU
        all_heatmaps.append(heat)

    # ---- Concatenate heatmaps over time to get full-video heatmap ----
    video_heatmap = torch.cat(all_heatmaps, dim=0)  # [T_adj, H, W] on CPU

    # Normalize per frame to [0,1]
    video_heatmap_norm = normalize_heatmap_per_video(video_heatmap)  # [T_adj, H, W]

    # Prepare grayscale frames for overlay
    frames_gray = cropped_unnorm.squeeze(1).cpu().numpy()  # [T_adj, 88, 88]

    # ---- Overlay heatmaps on grayscale mouth crops ----
    print("Creating overlay video...")
    frames_rgb = overlay_heatmap_on_frames(frames_gray, video_heatmap_norm, alpha=0.6)

    # ---- Save video ----
    output_path = args.video.split(".")[0] + "_occlusion_sensitivity" + "." + args.video.split(".")[1]
    print(f"Saving to output_path (fps={fps})")
    save_video(frames_rgb, fps=fps, output_path=output_path)
    print("Done.")


if __name__ == "__main__":
    main()
