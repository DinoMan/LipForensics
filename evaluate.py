"""Evaluate pre-trained LipForensics model on various face forgery datasets"""

import argparse
from collections import defaultdict
from xml.parsers.expat import model

import pandas as pd
from sklearn import metrics
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop
from tqdm import tqdm
import face_alignment
from data.transforms import NormalizeVideo, ToTensorVideo
from data.dataset_clips import ForensicsClips, CelebDFClips, DFDCClips
from data.samplers import ConsecutiveClipSampler
from models.spatiotemporal_net import get_model
from utils import get_files_from_split
from preprocessing.crop_mouths import crop_video


import torch
import torch.nn.functional as F


import torch
import numpy as np
import cv2

def save_cam_overlay_video(video_gray: torch.Tensor,
                           cam: torch.Tensor,
                           out_path: str,
                           fps: int = 30,
                           alpha: float = 0.6,
                           colormap: int = cv2.COLORMAP_JET):
    """
    video_gray: [T, H, W] tensor (values in [0,1] or [0,255])
    cam:        [T, H, W] tensor (can be any range, normalized per-frame)
    out_path:   output .mp4 filename
    fps:        frames per second
    alpha:      blending weight for grayscale frame (heatmap weight = 1-alpha)
    colormap:   OpenCV colormap, e.g. cv2.COLORMAP_JET
    """

    assert video_gray.shape == cam.shape, "video_gray and cam must be same shape [T,H,W]"

    T, H, W = video_gray.shape

    # Prepare video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))

    # Convert to numpy
    video_np = video_gray.detach().cpu().numpy()
    cam_np = cam.detach().cpu().numpy()

    for t in range(T):
        frame = video_np[t]
        cam_t = cam_np[t]

        # --- Normalize grayscale frame to [0,255] ---
        if frame.max() <= 1.0:
            frame_uint8 = (frame * 255).astype(np.uint8)
        else:
            frame_uint8 = frame.astype(np.uint8)

        frame_bgr = cv2.cvtColor(frame_uint8, cv2.COLOR_GRAY2BGR)

        # --- Normalize CAM per frame ---
        cam_t = cam_t - cam_t.min()
        if cam_t.max() > 0:
            cam_t = cam_t / cam_t.max()

        cam_uint8 = (cam_t * 255).astype(np.uint8)

        # --- Build heatmap ---
        heatmap = cv2.applyColorMap(cam_uint8, colormap)

        # --- Blend grayscale + heatmap ---
        overlay = cv2.addWeighted(frame_bgr, alpha, heatmap, 1 - alpha, 0)

        writer.write(overlay)

    writer.release()
    print(f"[OK] Saved CAM overlay video → {out_path}")


class GradCAM3D:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.A = None    # activations: [C, T', H', W']
        self.G = None    # gradients:    [C, T', H', W']

        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        # out: [C, T', H', W']
        self.A = out

    def _backward_hook(self, module, grad_input, grad_output):
        # grad_output[0]: [C, T', H', W']
        self.G = grad_output[0]

    def __call__(self, batch, lengths, time_agg="none"):            
        """
        x: [B, C, T, H, W]
        """
        cams = []
        for i, x in enumerate(batch):
            x = x.unsqueeze(0)  # [1, C, T, H, W]
            self.model.zero_grad()

            # Forward → [1, 1] single logit
            logit = self.model(x, lengths=[lengths[i]])[0, 0]
            logit.backward()

            A = self.A          # [C, T', H', W']
            G = self.G          # [C, T', H', W']

            # === 1. Compute channel weights α_c (GAP over T'H'W') ===
            # weights: [C, 1, 1, 1]
            weights = G.mean(dim=(1, 2, 3), keepdim=True)

            # === 2. Weighted sum over channels ===
            # A is [C, T', H', W'], weights is [C, 1, 1, 1]
            cam = (weights * A).sum(dim=0, keepdim=True)   # [1, T', H', W']
            cam = F.relu(cam)

            # === 3. Normalize to [0, 1] ===
            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-8)

            # === 4. Upsample to full video size ===
            _, _, T, H, W = x.shape      # input shape
            cam = F.interpolate(
                cam.unsqueeze(0),        # [1,1,T',H',W']
                size=(T, H, W),
                mode="trilinear",
                align_corners=False
            )[0, 0]                      # → [T, H, W]
            cams.append(cam)
            
        cam = torch.cat(cams, dim=0)  # [B, T, H, W]
        if time_agg == "mean":
            return cam, cam.mean(dim=0)   # [T,H,W], [H,W]
        elif time_agg == "max":
            return cam, cam.max(dim=0)[0] # [T,H,W], [H,W]
        else:
            return cam, None




def parse_args():
    parser = argparse.ArgumentParser(description="DeepFake detector evaluation")
    parser.add_argument(
        "--dataset",
        help="Dataset to evaluate on",
        type=str,
        choices=[
            "FaceForensics++",
            "Deepfakes",
            "FaceSwap",
            "Face2Face",
            "NeuralTextures",
            "FaceShifter",
            "DeeperForensics",
            "CelebDF",
            "DFDC",
        ],
        default="FaceForensics++",
    )
    parser.add_argument(
        "--compression",
        help="Video compression level for FaceForensics++",
        type=str,
        choices=["c0", "c23", "c40"],
        default="c23",
    )
    parser.add_argument("--grayscale", dest="grayscale", action="store_true")
    parser.add_argument("--rgb", dest="grayscale", action="store_false")
    parser.set_defaults(grayscale=True)
    parser.add_argument("--frames_per_clip", default=25, type=int)
    parser.add_argument("--stride", default=25, type=int)

    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--device", help="Device to put tensors on", type=str, default="cuda:0")
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument(
        "--weights_forgery_path",
        help="Path to pretrained weights for forgery detection",
        type=str,
        default="./models/weights/lipforensics_ff.pth"
    )
    parser.add_argument(
        "--split_path", help="Path to FF++ splits", type=str, default="./data/datasets/Forensics/splits/test.json"
    )
    parser.add_argument(
        "--dfdc_metadata_path", help="Path to DFDC metadata", type=str, default="./data/datasets/DFDC/metadata.json"
    )

    args = parser.parse_args()

    return args


def compute_video_level_auc(video_to_logits, video_to_labels):
    """ "
    Compute video-level area under ROC curve. Averages the logits across the video for non-overlapping clips.

    Parameters
    ----------
    video_to_logits : dict
        Maps video ids to list of logit values
    video_to_labels : dict
        Maps video ids to label
    """
    output_batch = torch.stack(
        [torch.mean(torch.stack(video_to_logits[video_id]), 0, keepdim=False) for video_id in video_to_logits.keys()]
    )
    output_labels = torch.stack([video_to_labels[video_id] for video_id in video_to_logits.keys()])

    fpr, tpr, _ = metrics.roc_curve(output_labels.cpu().numpy(), output_batch.cpu().numpy())
    return metrics.auc(fpr, tpr)


def validate_video_level(model, loader, args):
    """ "
    Evaluate model using video-level AUC score.

    Parameters
    ----------
    model : torch.nn.Module
        Model instance
    loader : torch.utils.data.DataLoader
        Loader for forgery data
    args
        Options for evaluation
    """
    model.eval()

    video_to_logits = defaultdict(list)
    video_to_labels = {}
    with torch.no_grad():
        for data in tqdm(loader):
            images, labels, video_indices = data
            images = images.to(args.device)
            labels = labels.to(args.device)

            # Forward
            logits = model(images, lengths=[args.frames_per_clip] * images.shape[0])

            # Get maps from video ids to list of logits (representing outputs for clips) as well as to label
            for i in range(len(video_indices)):
                video_id = video_indices[i].item()
                video_to_logits[video_id].append(logits[i])
                video_to_labels[video_id] = labels[i]

    auc_video = compute_video_level_auc(video_to_logits, video_to_labels)
    return auc_video


def save_video(x, output_path="out.mp4"):
    
    import torch
    import numpy as np
    import imageio

    video = x  # [T, 1, H, W], values 0–1 or 0–255

    # Convert to uint8 HxW for each frame
    frames = (video[:, 0] * 255).clamp(0, 255).byte().cpu().numpy()  # shape [T, H, W]

    imageio.mimsave(output_path, frames, fps=25, codec="libx264", quality=8, pixelformat="gray")


def main():
    args = parse_args()

    model = get_model(weights_forgery_path=args.weights_forgery_path, device="cuda:0").eval()

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device="cuda", flip_input=False)

    transform = Compose(
        [ToTensorVideo(), CenterCrop((88, 88)), NormalizeVideo((0.421,), (0.165,))]
    )
    
    target_layer = model.trunk.layer3
    cam_extractor = GradCAM3D(model, target_layer)
    for video_path in ["/home/dino/Documents/git/LipForensics/data/datasets/custom/015_with_audio.mp4"]:
        
        import torchvision.io as io

        # Returns video: (T, H, W, C), audio: (num_channels, num_samples)
        video, audio, info = io.read_video(video_path, pts_unit='sec')

        adjusted_length = (video.shape[0] // args.frames_per_clip) * args.frames_per_clip
        vid = crop_video(video[:adjusted_length], fa)

        save_video(vid/255, output_path="before.mp4")
        cropped_video = transform(vid)
        save_video(cropped_video, output_path="after.mp4")
        video_clips = cropped_video.unfold(dimension=0, size=args.frames_per_clip, step=args.stride).permute(0, 1, 4, 2, 3).cuda()
        
        cam = cam_extractor(video_clips, lengths=[args.frames_per_clip] * video_clips.shape[0])[0]
        
        save_cam_overlay_video(CenterCrop((88, 88))(vid).squeeze(), cam, fps=30, out_path="cam_overlay.mp4")
        
        with torch.no_grad():
            logits = model(video_clips, lengths=[args.frames_per_clip] * video_clips.shape[0])
        print("Logits:", logits)
        print(logits.mean().item())

if __name__ == "__main__":
    main()


