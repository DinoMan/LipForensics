"""Evaluate pre-trained LipForensics model on various face forgery datasets"""

import argparse
from collections import defaultdict

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

    for video_path in ["/home/dino/Documents/git/LipForensics/data/datasets/custom/nt_015_919_with_audio.mp4"]:
        
        import torchvision.io as io

        # Returns video: (T, H, W, C), audio: (num_channels, num_samples)
        video, audio, info = io.read_video(video_path, pts_unit='sec')

        adjusted_length = (video.shape[0] // args.frames_per_clip) * args.frames_per_clip
        
        vid = crop_video(video[:adjusted_length], fa)
        cropped_video = transform(vid)
        save_video(cropped_video, output_path="after.mp4")
        video_clips = cropped_video.unfold(dimension=0, size=args.frames_per_clip, step=args.stride).permute(0, 1, 4, 2, 3).cuda()
        with torch.no_grad():
            logits = model(video_clips, lengths=[args.frames_per_clip] * video_clips.shape[0])
        print("Logits:", logits)
        print(logits.mean().item())

if __name__ == "__main__":
    main()
