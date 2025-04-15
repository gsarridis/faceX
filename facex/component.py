import os
import torch
import torch.nn as nn
from tqdm import tqdm
from .dataset import get_dataloaders, get_dataloader_embeddings
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as F


import torch
import numpy as np
from torchvision.models import resnet
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import pytorch_grad_cam
from .nn_utils import (
    set_seed,
    UnNormalize,
    load_config,
    get_patches,
    CustomGradCAM,
    normalize_values,
)
from .plot_facex import plot
from io import BytesIO
import base64
from torchvision import transforms


def __mycall__(self, x):
    self.gradients = []
    self.activations = []
    outputs = self.model(x)
    # Return only the first output
    return outputs[0] if isinstance(outputs, tuple) else outputs


class SimilarityToConceptTarget:
    def __init__(self, features):
        self.features = features

    def __call__(self, model_output):
        cos = torch.nn.CosineSimilarity(dim=0)
        return cos(model_output, self.features)


class DifferenceFromConceptTarget:
    def __init__(self, features):
        self.features = features

    def __call__(self, model_output):
        cos = torch.nn.CosineSimilarity(dim=0)
        return 1 - cos(model_output, self.features)


def get_normalization_params(dataloader):
    # Get the dataset associated with the dataloader
    dataset = dataloader.dataset

    # Check if the dataset has the 'transforms' attribute
    if hasattr(dataset, "transform"):
        # If transform is a Compose, we need to iterate through it
        if isinstance(dataset.transform, transforms.Compose):
            for t in dataset.transform.transforms:
                # Look for Normalize transform
                if isinstance(t, transforms.Normalize):
                    mean = t.mean
                    std = t.std
                    return mean, std
        # If it's not a Compose, it may just be Normalize
        elif isinstance(dataset.transform, transforms.Normalize):
            return dataset.transform.mean, dataset.transform.std

    # If no Normalize transform is found, return None
    return None, None


def get_resize_size(data_transform):
    if hasattr(
        data_transform, "transforms"
    ):  # If the transformation is a Compose object
        for t in data_transform.transforms:
            if isinstance(t, transforms.Resize):
                return t.size  # Return the resize size
    elif isinstance(data_transform, transforms.Resize):
        return data_transform.size  # If it's a single Resize transform
    return None, None  # No Resize transform found


def save_plot_as_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    return img_base64


def get_target_layers(model: nn.Module, layer_name: str) -> list:
    """Retrieve the specified target layers from the model."""
    target_layers = []
    try:
        layer = dict(model.named_modules())[layer_name]
        target_layers.append(layer)
    except KeyError:
        raise ValueError(f"Layer '{layer_name}' not found in the model.")
    return target_layers


def process_data(
    imgs,
    att_map,
    att,
    pth,
    config,
    unorm,
    tp,
    tpv,
    tin,
    patch_size,
):
    for i in range(imgs.shape[0]):
        img = unorm(imgs[i])
        img_pth = pth[i]
        img_name = img_pth.split("/")[-1][:-4]
        img = img.unsqueeze(dim=0)
        attention = att_map[i].unsqueeze(dim=0)
        attention = F.resize(attention, config["img_size"])
        groundtruth_attention = att[i].reshape(
            1, 1, config["img_size"], config["img_size"]
        )

        img_patches = get_patches(img, patch_size, 3)
        attention_patches = get_patches(attention, patch_size, 1)
        groundtruth_attention_patches = get_patches(
            groundtruth_attention, patch_size, 1
        )

        mask = (groundtruth_attention_patches > 0).float()
        attention_sums = (
            (attention_patches * mask).view(1, -1, patch_size * patch_size).sum(-1)
        )

        max_index = attention_sums.argmax()
        selected_patch = img_patches[0, :, max_index]
        selected_attention_value = attention_sums[0, max_index]

        tp, tpv, tin = update_top_patches(
            selected_patch,
            selected_attention_value,
            tp,
            tpv,
            tin,
            img_name,
            config,
        )
    return tp, tpv, tin


def update_top_patches(
    selected_patch,
    selected_attention_value,
    tp,
    tpv,
    tin,
    img_name,
    config,
):
    if len(tp) < config["K_top_patches"] or selected_attention_value.item() > min(tpv):
        if selected_attention_value.item() > 0:
            if len(tp) >= config["K_top_patches"]:
                min_index = tpv.index(min(tpv))
                del tp[min_index]
                del tpv[min_index]
                del tin[min_index]
            tp.append(selected_patch)
            tpv.append(selected_attention_value.item())
            tin.append(img_name)
    return tp, tpv, tin


def global_focus(img1, att_maps, activations_frac_att, num_of_imgs):
    th = 0
    att_pixels = {}
    # Load the images

    pixels1 = np.array(img1) / 255

    th = np.sum(pixels1[pixels1 > 0]) / np.sum(pixels1 > 0)

    for region, att in att_maps.items():
        att = F.resize(att, 64)
        # print(torch.max(att))
        att_pixels[region] = np.array(att)  # / 255

    for att_dir in list(att_pixels.keys()):
        intersec = np.sum((pixels1 > th) & (att_pixels[att_dir] > 0))
        if np.sum(att_pixels[att_dir] > 0) > 0:
            activations_frac_att[att_dir] += intersec / np.sum(
                (att_pixels[att_dir] > 0)
            )
            num_of_imgs[att_dir] += 1
    return activations_frac_att, num_of_imgs


def facex(test_loader, model, config, r_target):
    model.eval()
    # Example usage with a dataloader
    mean, std = get_normalization_params(test_loader)
    if mean and std:
        # print(f"Mean: {mean}, Std: {std}")
        unorm = UnNormalize(mean=mean, std=std)
    # else:
    #     # print("No normalization found in the dataset's transforms.")
    # unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    top_patches = {key: [] for key in config["att_list"]}
    top_patch_values = {key: [] for key in config["att_list"]}
    top_img_names = {key: [] for key in config["att_list"]}
    target_layers = get_target_layers(model, config["target_layer"])
    target_class = [ClassifierOutputTarget(r_target[1])]
    activations_frac_att = {key: 0 for key in config["att_list"]}
    num_of_imgs = {key: 0 for key in config["att_list"]}
    use_cuda = config["device"] == torch.device("cuda")
    with CustomGradCAM(
        model=model, target_layers=target_layers, use_cuda=use_cuda
    ) as cam:
        cam.batch_size = config["bs"]
        for idx, (data, target, atts, pth) in enumerate(tqdm(test_loader)):

            data = data.to(config["device"])
            att_map, norm_att_map = cam(input_tensor=data, targets=target_class)
            gradcam = Image.fromarray((norm_att_map[0, 0] * 255).astype(np.uint8))
            gradcam = gradcam.resize((64, 64))
            activations_frac_att, num_of_imgs = global_focus(
                gradcam, atts, activations_frac_att, num_of_imgs
            )

            patch_size = 16
            att_map = torch.tensor(att_map)
            for region, att in atts.items():
                tp = top_patches[region]
                tpv = top_patch_values[region]
                tin = top_img_names[region]
                # print(data.sum().item())
                top_patches[region], top_patch_values[region], top_img_names[region] = (
                    process_data(
                        data.clone(),
                        att_map,
                        att,
                        pth,
                        config,
                        unorm,
                        tp,
                        tpv,
                        tin,
                        patch_size,
                    )
                )
        facex_patch_plots = {}
        for region in list(atts.keys()):
            sorted_indices = sorted(
                range(len(top_patch_values[region])),
                key=lambda i: top_patch_values[region][i],
                reverse=True,
            )
            sorted_images = [top_patches[region][i] for i in sorted_indices]

            # Take the top 20 images
            top_20_images = sorted_images[: config["K_top_patches"]]

            # Plot the images
            fig, axs = plt.subplots(1, config["K_top_patches"], figsize=(20, 1))

            for i, img in enumerate(top_20_images):
                img_array = img.detach().cpu().numpy()
                img_array = np.transpose(img_array, (1, 2, 0))

                axs[i].imshow(img_array)
                # axs[i].title(image_name)
            for i in range(config["K_top_patches"]):
                axs[i].axis("off")
            facex_patch_plots[region] = fig
            # plt.savefig("patches_" + region + ".png")
            # plt.show()
            # plt.close()

    activations_frac_att = normalize_values(activations_frac_att, num_of_imgs)

    facex_heatmap_plot = plot(
        config["face_prototype_dir"],
        config["hat_glasses_prototype_dir"],
        activations_frac_att,
    )

    # Save the heatmap plot as a base64 string
    heatmap_base64 = save_plot_as_base64(facex_heatmap_plot)

    # Save the patch plots as base64 strings
    patch_base64s = {}
    for region, fig in facex_patch_plots.items():
        patch_base64 = save_plot_as_base64(fig)
        patch_base64s[region] = patch_base64

    # Combine all images into one HTML document
    combined_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Facex Plots</title>
    </head>
    <body>
        <h1>Facex Heatmap</h1>
        <img src="data:image/png;base64,{}" alt="Heatmap Plot">
        <h1>High Impact Patches</h1>
    """.format(
        heatmap_base64
    )

    for region, patch_base64 in patch_base64s.items():
        combined_html += """
        <h2>Region: {}</h2>
        <img src="data:image/png;base64,{}" alt="Patch Plot for Region: {}">
        """.format(
            region, patch_base64, region
        )

    combined_html += """
    </body>
    </html>
    """
    # Save the combined HTML to a file
    # with open("facex_plots.html", "w") as f:
    #     f.write(combined_html)

    return facex_patch_plots, facex_heatmap_plot, combined_html


def facex_embeddings(test_loader, model, config, r_target):
    model.eval()
    mean, std = get_normalization_params(test_loader)
    if mean and std:
        # print(f"Mean: {mean}, Std: {std}")
        unorm = UnNormalize(mean=mean, std=std)
    top_patches = {key: [] for key in config["att_list"]}
    top_patch_values = {key: [] for key in config["att_list"]}
    top_img_names = {key: [] for key in config["att_list"]}
    target_layers = get_target_layers(model, config["target_layer"])

    activations_frac_att = {key: 0 for key in config["att_list"]}
    num_of_imgs = {key: 0 for key in config["att_list"]}
    use_cuda = config["device"] == torch.device("cuda")
    # Assign the new __call__ method to the instance to handle multiple model outputs (ie features, sth else)
    pytorch_grad_cam.activations_and_gradients.ActivationsAndGradients.__call__ = (
        __mycall__.__get__(
            None, pytorch_grad_cam.activations_and_gradients.ActivationsAndGradients
        )
    )
    for idx, (data0, data1, target, atts, pth) in enumerate(tqdm(test_loader)):
        with torch.no_grad():
            data0, data1 = data0.to(config["device"]), data1.to(config["device"])
            outputs = model(data0)
            # Check if the output is a tuple (i.e., multiple outputs)
            if isinstance(outputs, tuple):
                img0_concept_features = outputs[0].squeeze()  # Take the first output
            else:
                img0_concept_features = (
                    outputs.squeeze()
                )  # If it's not a tuple, just assign the output

        with CustomGradCAM(
            model=model, target_layers=target_layers, use_cuda=use_cuda
        ) as cam:
            cam.batch_size = config["bs"]
            data1 = data1.to(config["device"])
            if r_target[1] == 0:
                img0_target = [DifferenceFromConceptTarget(img0_concept_features)]
            elif r_target[1] == 1:
                img0_target = [SimilarityToConceptTarget(img0_concept_features)]
            else:
                raise ValueError(
                    f"Unexpected target value: {r_target[1]}. It should be either 0 (different person) or 1 (same person)"
                )

            att_map, norm_att_map = cam(input_tensor=data1, targets=img0_target)
            gradcam = Image.fromarray((norm_att_map[0, 0] * 255).astype(np.uint8))
            gradcam = gradcam.resize((64, 64))
            activations_frac_att, num_of_imgs = global_focus(
                gradcam, atts, activations_frac_att, num_of_imgs
            )

            patch_size = 16
            att_map = torch.tensor(att_map)
            for region, att in atts.items():
                tp = top_patches[region]
                tpv = top_patch_values[region]
                tin = top_img_names[region]
                # print(data.sum().item())
                top_patches[region], top_patch_values[region], top_img_names[region] = (
                    process_data(
                        data1.clone(),
                        att_map,
                        att,
                        pth,
                        config,
                        unorm,
                        tp,
                        tpv,
                        tin,
                        patch_size,
                    )
                )
        facex_patch_plots = {}
        for region in list(atts.keys()):
            sorted_indices = sorted(
                range(len(top_patch_values[region])),
                key=lambda i: top_patch_values[region][i],
                reverse=True,
            )
            sorted_images = [top_patches[region][i] for i in sorted_indices]

            # Take the top 20 images
            top_20_images = sorted_images[: config["K_top_patches"]]

            # Plot the images
            fig, axs = plt.subplots(1, config["K_top_patches"], figsize=(20, 1))

            for i, img in enumerate(top_20_images):
                img_array = img.detach().cpu().numpy()
                img_array = np.transpose(img_array, (1, 2, 0))

                axs[i].imshow(img_array)
                # axs[i].title(image_name)
            for i in range(config["K_top_patches"]):
                axs[i].axis("off")
            facex_patch_plots[region] = fig
            # plt.savefig("patches_" + region + ".png")
            # plt.show()
            # plt.close()

    activations_frac_att = normalize_values(activations_frac_att, num_of_imgs)

    facex_heatmap_plot = plot(
        config["face_prototype_dir"],
        config["hat_glasses_prototype_dir"],
        activations_frac_att,
    )

    # Save the heatmap plot as a base64 string
    heatmap_base64 = save_plot_as_base64(facex_heatmap_plot)

    # Save the patch plots as base64 strings
    patch_base64s = {}
    for region, fig in facex_patch_plots.items():
        patch_base64 = save_plot_as_base64(fig)
        patch_base64s[region] = patch_base64

    # Combine all images into one HTML document
    combined_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Facex Plots</title>
    </head>
    <body>
        <h1>Facex Heatmap</h1>
        <img src="data:image/png;base64,{}" alt="Heatmap Plot">
        <h1>High Impact Patches</h1>
    """.format(
        heatmap_base64
    )

    for region, patch_base64 in patch_base64s.items():
        combined_html += """
        <h2>Region: {}</h2>
        <img src="data:image/png;base64,{}" alt="Patch Plot for Region: {}">
        """.format(
            region, patch_base64, region
        )

    combined_html += """
    </body>
    </html>
    """
    # Save the combined HTML to a file
    # with open("facex_plots.html", "w") as f:
    #     f.write(combined_html)

    return facex_patch_plots, facex_heatmap_plot, combined_html


def run(
    target,
    protected,
    target_class,
    model,
    data_dir,
    csv_dir,
    target_layer,
    transform=None,
    img_size=512,
):
    # Parse command line arguments

    config = {}

    config["data_dir"] = data_dir
    config["att_dir"] = data_dir + "-mask-anno"
    config["csv_dir"] = csv_dir
    config["dataset"] = "bupt"

    config["target_layer"] = target_layer

    config["img_size"] = img_size
    config["att_list"] = [
        "skin",
        "u_lip",
        "l_lip",
        "hair",
        "l_ear",
        "r_ear",
        "nose",
        "mouth",
        "l_brow",
        "r_brow",
        "l_eye",
        "r_eye",
        "ear_r",
        "neck",
        "neck_l",
        "cloth",
        "background",
        "hat",
        "eye_g",
    ]
    config["seed"] = 1
    config["bs"] = 1
    config["nw"] = 1
    config["K_top_patches"] = 20
    # Get the directory of the current file
    current_directory = os.path.dirname(__file__)

    # Construct the paths relative to the current file's directory
    config["face_prototype_dir"] = os.path.join(current_directory, "face_model_v3.json")
    config["hat_glasses_prototype_dir"] = os.path.join(
        current_directory, "hat_glasses.json"
    )

    config["target"] = target
    config["target_class"] = target_class

    config["protected"] = protected
    config["transform"] = transform

    config["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set the random seed for reproducibility
    set_seed(config["seed"])
    # mode: (str, optional) Can be `"online"`, `"offline"` or `"disabled"`. Defaults to online.

    rt = config["target_class"]  # 0 or 1
    rt_name = str(rt) + config["target"]  # {0 or 1} + <e.g. male>
    # Create the CNN model
    model = model.to(config["device"])
    model.eval()
    test_loader = get_dataloaders(
        dataset=config["dataset"],
        task=config["target"],
        protected=config["protected"],
        data_dir=config["data_dir"],
        csv_dir=config["csv_dir"],
        att_dir=config["att_dir"],
        att_list=config["att_list"],
        img_size=config["img_size"],
        bs=config["bs"],
        nw=config["nw"],
        one_class=rt,
        transform=config["transform"],
    )

    patches, heatmap, combined_html = facex(test_loader, model, config, [rt_name, rt])
    return patches, heatmap, combined_html


def run_mammoth(dataset, protected, target_class, model, target_layer):
    # Parse command line arguments

    config = {}

    config["data_dir"] = dataset.root_dir
    config["att_dir"] = dataset.root_dir + "-mask-anno"
    config["csv_dir"] = dataset.path
    config["dataset"] = "bupt"

    config["target_layer"] = target_layer

    config["img_size"] = get_resize_size(dataset.data_transform)[0]
    config["att_list"] = [
        "skin",
        "u_lip",
        "l_lip",
        "hair",
        "l_ear",
        "r_ear",
        "nose",
        "mouth",
        "l_brow",
        "r_brow",
        "l_eye",
        "r_eye",
        "ear_r",
        "neck",
        "neck_l",
        "cloth",
        "background",
        "hat",
        "eye_g",
    ]
    config["seed"] = 1
    config["bs"] = 1
    config["nw"] = 1
    config["K_top_patches"] = 20
    # Get the directory of the current file
    current_directory = os.path.dirname(__file__)

    # Construct the paths relative to the current file's directory
    config["face_prototype_dir"] = os.path.join(current_directory, "face_model_v3.json")
    config["hat_glasses_prototype_dir"] = os.path.join(
        current_directory, "hat_glasses.json"
    )

    config["target"] = dataset.target
    config["target_class"] = target_class

    config["protected"] = protected
    config["transform"] = dataset.data_transform

    config["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set the random seed for reproducibility
    set_seed(config["seed"])
    # mode: (str, optional) Can be `"online"`, `"offline"` or `"disabled"`. Defaults to online.

    rt = config["target_class"]
    rt_name = str(rt) + config["target"]
    # Create the CNN model
    model = model.to(config["device"])
    model.eval()
    test_loader = get_dataloaders(
        dataset=config["dataset"],
        task=config["target"],
        protected=config["protected"],
        data_dir=config["data_dir"],
        csv_dir=config["csv_dir"],
        att_dir=config["att_dir"],
        att_list=config["att_list"],
        img_size=config["img_size"],
        bs=config["bs"],
        nw=config["nw"],
        one_class=rt,
        transform=config["transform"],
    )

    patches, heatmap, combined_html = facex(test_loader, model, config, [rt_name, rt])
    return combined_html


def run_embeddings_mammoth(dataset, protected, target_class, model, target_layer):
    # Parse command line arguments

    config = {}

    config["data_dir"] = dataset.root_dir
    config["att_dir"] = dataset.root_dir + "-mask-anno"
    config["csv_dir"] = dataset.path
    config["dataset"] = "bupt"

    config["target_layer"] = target_layer

    config["img_size"] = get_resize_size(dataset.data_transform)[0]
    config["att_list"] = [
        "skin",
        "u_lip",
        "l_lip",
        "hair",
        "l_ear",
        "r_ear",
        "nose",
        "mouth",
        "l_brow",
        "r_brow",
        "l_eye",
        "r_eye",
        "ear_r",
        "neck",
        "neck_l",
        "cloth",
        "background",
        "hat",
        "eye_g",
    ]
    config["seed"] = 1
    config["bs"] = 1
    config["nw"] = 0
    config["K_top_patches"] = 20
    # Get the directory of the current file
    current_directory = os.path.dirname(__file__)

    # Construct the paths relative to the current file's directory
    config["face_prototype_dir"] = os.path.join(current_directory, "face_model_v3.json")
    config["hat_glasses_prototype_dir"] = os.path.join(
        current_directory, "hat_glasses.json"
    )

    config["target"] = dataset.target
    config["target_class"] = target_class

    config["protected"] = protected
    config["transform"] = dataset.data_transform

    config["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set the random seed for reproducibility
    set_seed(config["seed"])
    # mode: (str, optional) Can be `"online"`, `"offline"` or `"disabled"`. Defaults to online.

    rt = config["target_class"]
    rt_name = str(rt) + config["target"]
    # Create the CNN model
    model = model.to(config["device"])
    model.eval()
    # TODO: create a dataloader returning these values: for idx, (data0, data1, target, atts, pth) in enumerate(tqdm(test_loader))
    test_loader = get_dataloader_embeddings(
        task=config["target"],
        protected=config["protected"],
        data_dir=config["data_dir"],
        csv_dir=config["csv_dir"],
        att_dir=config["att_dir"],
        att_list=config["att_list"],
        img_size=config["img_size"],
        bs=config["bs"],
        nw=config["nw"],
        one_class=rt,
        transform=config["transform"],
    )

    patches, heatmap, combined_html = facex_embeddings(
        test_loader, model, config, [rt_name, rt]
    )

    # Close all open plots or figures
    plt.close("all")
    return combined_html
