import sys
sys.path.insert(0, '.')

import argparse
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from model.trainer import Trainer
from model.utils import weight_init
import dataset.Transforms as myTransforms


PATCH_SIZE = 256
GRID = 4

def load_and_transform(pre_path, post_path, label_path, transform):
    pre = cv2.imread(pre_path)
    post = cv2.imread(post_path)
    label = cv2.imread(label_path, 0)
    img = np.concatenate((pre, post), axis=2)
    img_t, label_t = transform(img, label)
    return img_t, label_t, pre, post, label


def stitch_patches(patches, rows=4, cols=4):
    h, w = patches[0].shape[:2]
    if len(patches[0].shape) == 3:
        canvas = np.zeros((rows * h, cols * w, patches[0].shape[2]), dtype=patches[0].dtype)
        for idx, patch in enumerate(patches):
            r, c = idx // cols, idx % cols
            canvas[r*h:(r+1)*h, c*w:(c+1)*w, :] = patch
    else:
        canvas = np.zeros((rows * h, cols * w), dtype=patches[0].dtype)
        for idx, patch in enumerate(patches):
            r, c = idx // cols, idx % cols
            canvas[r*h:(r+1)*h, c*w:(c+1)*w] = patch
    return canvas


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id) if args.onGPU else ''

    model = Trainer(args.model_type, False).float()
    if args.onGPU:
        model = model.cuda()
    model.eval()

    checkpoint_path = os.path.join(args.savedir, 'best_model.pth')
    state_dict = torch.load(checkpoint_path, map_location='cpu' if not args.onGPU else None)
    model.load_state_dict(state_dict)
    print(f'Loaded checkpoint from {checkpoint_path}')

    mean = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    transform = myTransforms.Compose([
        myTransforms.Normalize(mean=mean, std=std),
        myTransforms.Scale(args.inWidth, args.inHeight),
        myTransforms.ToTensor()
    ])

    file_root = args.file_root
    image_name = args.image_name

    pre_patches_display = []
    post_patches_display = []
    label_patches_display = []
    pred_patches = []

    for patch_id in range(16):
        pre_path = os.path.join(file_root, 'test', 'A', f'{image_name}_{patch_id}.png')
        post_path = os.path.join(file_root, 'test', 'B', f'{image_name}_{patch_id}.png')
        label_path = os.path.join(file_root, 'test', 'label', f'{image_name}_{patch_id}.png')

        if not os.path.exists(pre_path):
            print(f'Warning: {pre_path} not found, skipping.')
            continue

        img_t, label_t, pre_rgb, post_rgb, label_raw = load_and_transform(
            pre_path, post_path, label_path, transform
        )

        img_t = img_t.unsqueeze(0)
        label_t = label_t.unsqueeze(0)

        if args.onGPU:
            img_t = img_t.cuda()

        pre_t = img_t[:, 0:3].float()
        post_t = img_t[:, 3:6].float()

        with torch.no_grad():
            output = model(pre_t, post_t)
            pred = torch.where(output > 0.5, torch.ones_like(output), torch.zeros_like(output))

        pred_np = pred.squeeze().cpu().numpy().astype(np.uint8)

        pre_rgb_display = cv2.cvtColor(pre_rgb, cv2.COLOR_BGR2RGB)
        post_rgb_display = cv2.cvtColor(post_rgb, cv2.COLOR_BGR2RGB)

        pre_patches_display.append(pre_rgb_display)
        post_patches_display.append(post_rgb_display)
        label_patches_display.append((label_raw > 0).astype(np.uint8))
        pred_patches.append(pred_np)

    pre_full = stitch_patches(pre_patches_display, GRID, GRID)
    post_full = stitch_patches(post_patches_display, GRID, GRID)
    label_full = stitch_patches(label_patches_display, GRID, GRID)
    pred_full = stitch_patches(pred_patches, GRID, GRID)

    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    titles = ['Before', 'After', 'Ground Truth', 'Prediction']

    axes[0, 0].imshow(pre_full)
    axes[0, 0].set_title('Before', fontsize=14)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(post_full)
    axes[0, 1].set_title('After', fontsize=14)
    axes[0, 1].axis('off')

    axes[1, 0].imshow(label_full, cmap='gray', vmin=0, vmax=1)
    axes[1, 0].set_title('Ground Truth', fontsize=14)
    axes[1, 0].axis('off')

    axes[1, 1].imshow(pred_full, cmap='gray', vmin=0, vmax=1)
    axes[1, 1].set_title('Prediction', fontsize=14)
    axes[1, 1].axis('off')

    plt.tight_layout()
    out_path = f'{image_name}_prediction.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Saved prediction figure to {out_path}')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_name', type=str, required=True,
                        help='Image identifier, e.g. test_1, test_100')
    parser.add_argument('--file_root', type=str, default='./levir_cd_256',
                        help='Dataset root directory')
    parser.add_argument('--savedir', type=str,
                        default='./results_LEVIR_iter_80000_lr_0.0002',
                        help='Directory with best_model.pth')
    parser.add_argument('--inWidth', type=int, default=256)
    parser.add_argument('--inHeight', type=int, default=256)
    parser.add_argument('--model_type', type=str, default='small',
                        help='tiny or small')
    parser.add_argument('--onGPU', type=lambda x: (str(x).lower() == 'true'),
                        default=True)
    parser.add_argument('--gpu_id', type=int, default=0)
    args = parser.parse_args()
    main(args)
