#!/usr/bin/python
# -*- encoding: utf-8 -*-

from tqdm import tqdm

import torch

import os
import numpy as np
import torch
import facer
import matplotlib.pyplot as plt


def identity(x):
    return x


def find_all_images(
    root_dir, extensions=(".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp")
):
    """Recursively find all image files in the directory."""
    image_paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(extensions):
                image_paths.append(os.path.join(dirpath, filename))
    return image_paths


def create_masks(respth="./res/test_res", dspth="./data", image_paths=[]):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if not os.path.exists(respth):
        os.makedirs(respth)
    atts = [
        "background",
        "neck",
        "face",
        "cloth",
        "rr",
        "lr",
        "rb",
        "lb",
        "re",
        "le",
        "nose",
        "imouth",
        "llip",
        "ulip",
        "hair",
        "eyeg",
        "hat",
        "earr",
        "neck_l",
    ]

    atts_map = {
        "background": "background",
        "neck": "neck",
        "face": "skin",
        "cloth": "cloth",
        "rr": "r_ear",
        "lr": "l_ear",
        "rb": "r_brow",
        "lb": "l_brow",
        "re": "r_eye",
        "le": "l_eye",
        "nose": "nose",
        "imouth": "mouth",
        "llip": "l_lip",
        "ulip": "u_lip",
        "hair": "hair",
        "eyeg": "eye_g",
        "hat": "hat",
        "earr": "ear_r",
        "neck_l": "neck_l",
    }

    face_parser = facer.face_parser("farl/celebm/448", device=device)
    facer.transform.get_quad = identity
    with torch.no_grad():
        if len(image_paths) == 0:
            image_paths = find_all_images(dspth)

        for image_path in tqdm(image_paths):
            image = facer.hwc2bchw(facer.read_hwc(image_path)).to(
                device=device
            )  # image: 1 x 3 x h x w

            corners = torch.tensor(
                [
                    [0.0, 0.0],
                    [image.shape[2], 0.0],
                    [image.shape[2], image.shape[2]],
                    [0.0, image.shape[2]],
                ],
                device=device,
            )

            faces2 = {
                "rects": torch.zeros((1, 4), device=device),
                "points": corners.unsqueeze(dim=0),
                "image_ids": torch.tensor(0, device=device).unsqueeze(dim=0),
                "scores": torch.tensor(1, device=device).unsqueeze(dim=0),
            }

            with torch.inference_mode():
                faces = face_parser(image, faces2)

            seg_logits = faces["seg"]["logits"]
            seg_probs = seg_logits.softmax(dim=1)  # nfaces x nclasses x h x w

            n_classes = seg_probs.size(1)

            vis_seg_probs = seg_probs.argmax(dim=1).float() / n_classes * 255

            vis_img = vis_seg_probs.sum(0, keepdim=True)

            for i in range(seg_probs.shape[1]):
                vis_seg_probs = seg_probs[:, i, :, :].unsqueeze(1) * 255
                vis_img = vis_seg_probs.sum(0, keepdim=True)
                vis_img[vis_img < 128] = 0
                vis_img[vis_img > 128] = 255
                # Assuming vis_img is a torch tensor
                vis_img_numpy = vis_img.squeeze().detach().cpu().numpy().astype(int)

                # Display the image using Matplotlib
                plt.imshow(
                    vis_img_numpy, cmap="gray"
                )  # You can choose the colormap based on your preference
                plt.axis("off")  # Turn off axis values

                output_path = image_path.replace(
                    dspth,
                    respth,
                )
                directory = os.path.dirname(output_path)
                if not os.path.exists(directory):
                    os.makedirs(directory)
                save_dir = os.path.splitext(output_path)[0]
                output_path = f"{save_dir}_{atts_map[atts[i]]}.png"
                # print(output_path)
                if np.any(vis_img_numpy != 0):
                    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)

                # plt.show()
                plt.close()


if __name__ == "__main__":

    create_masks(
        respth="./mask_results",
        dspth="/fssd4/user-data/gsarridis/rfw/images/test_aligned",
    )
