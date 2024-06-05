import os


import torch
import json


import numpy as np
import cv2
import torch.nn.functional as F
from matplotlib.offsetbox import OffsetImage


import torch
import numpy as np
from pytorch_grad_cam import GradCAM
from typing import List
import logging, random, os
import torch
import torch.backends.cudnn as cudnn
import numpy as np


def normalize_values(dictionary, number):
    divided_dict = {}
    for key, value in dictionary.items():
        if value > 0:
            divided_dict[key] = value / number[key]
        else:
            divided_dict[key] = 0
    return divided_dict


def scale_cam_image(cam, target_size=None):
    result = []
    result_norm = []
    for img in cam:
        n_img = img - np.min(img)
        n_img = n_img / (1e-7 + np.max(n_img))
        if target_size is not None:
            img = cv2.resize(img, target_size)
            n_img = cv2.resize(n_img, target_size)
        result.append(img)
        result_norm.append(n_img)
    result = np.float32(result)
    result_norm = np.float32(result_norm)
    return result, result_norm


class CustomGradCAM(GradCAM):
    def __init__(self, model, target_layers, use_cuda=False, reshape_transform=None):
        super(CustomGradCAM, self).__init__(
            model, target_layers, use_cuda, reshape_transform
        )

    def compute_cam_per_layer(
        self,
        input_tensor: torch.Tensor,
        targets: List[torch.nn.Module],
        eigen_smooth: bool,
    ) -> np.ndarray:
        activations_list = [
            a.cpu().data.numpy() for a in self.activations_and_grads.activations
        ]
        grads_list = [
            g.cpu().data.numpy() for g in self.activations_and_grads.gradients
        ]
        target_size = self.get_target_width_height(input_tensor)

        cam_per_target_layer = []
        cam_per_target_layer_norm = []
        # Loop over the saliency image from every layer
        for i in range(len(self.target_layers)):
            target_layer = self.target_layers[i]
            layer_activations = None
            layer_grads = None
            if i < len(activations_list):
                layer_activations = activations_list[i]
            if i < len(grads_list):
                layer_grads = grads_list[i]

            cam = self.get_cam_image(
                input_tensor,
                target_layer,
                targets,
                layer_activations,
                layer_grads,
                eigen_smooth,
            )
            cam = np.maximum(cam, 0)

            scaled, scaled_norm = scale_cam_image(cam, target_size)
            cam_per_target_layer.append(scaled[:, None, :])
            cam_per_target_layer_norm.append(scaled_norm[:, None, :])

        return cam_per_target_layer, cam_per_target_layer_norm

    def aggregate_multi_layers(self, cam_per_target_layer: np.ndarray) -> np.ndarray:
        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
        cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
        result = np.mean(cam_per_target_layer, axis=1)
        return scale_cam_image(result)


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


# Load configuration from JSON file
def load_config(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


def get_patches(tensor, patch_size, channels):
    """Divide a tensor into patches of a given size."""
    # unfold = torch.nn.Unfold(kernel_size=patch_size, stride=patch_size)
    # patches = unfold(tensor)
    size = patch_size  # patch size
    stride = patch_size  # patch stride
    patches = tensor.unfold(2, size, stride).unfold(3, size, stride)
    if channels > 1:
        return patches.reshape(1, channels, -1, patch_size, patch_size)
    else:
        return patches.reshape(1, -1, patch_size, patch_size)


def getImage(img, zoom=1):
    return OffsetImage(img, zoom=zoom)


def set_seed(seed):
    logging.info(f"----- Random Seed: {seed} -----")
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False  # set to False for final report


def save_model(epoch, net, optimizer, loss, save_path):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        save_path,
    )
