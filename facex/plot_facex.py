import json
import base64
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io
import matplotlib.cm as cm


def plot(face_prototype_dir, hat_glasses_prototype_dir, region_weights):

    # Load the JSON annotation from the file
    with open(face_prototype_dir, "r") as json_file:
        annotation = json.load(json_file)

    with open(hat_glasses_prototype_dir, "r") as json_file:
        annotation_hat = json.load(json_file)

    # Load the image from the base64-encoded string
    image_data = base64.b64decode(annotation["imageData"])
    image = Image.open(io.BytesIO(image_data))

    # Convert the image to a numpy array
    image_array = np.array(image)

    # Load the image from the base64-encoded string
    image_data_hat = base64.b64decode(annotation_hat["imageData"])
    image_hat = Image.open(io.BytesIO(image_data_hat))

    # Convert the image to a numpy array
    image_array_hat = np.array(image_hat)

    # Create a plot and display the image
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(np.zeros(image_array.shape))
    ax[0].axis("off")
    ax[1].imshow(np.zeros(image_array_hat.shape) + 255)
    ax[1].axis("off")
    region_pos = {
        "r_ear": 0,
        "l_ear": 1,
        "l_eye": 2,
        "r_eye": 3,
        "nose": 4,
        "hair": 5,
        "r_brow": 6,
        "l_brow": 7,
        "l_lip": 8,
        "u_lip": 9,
        "mouth": 10,
        "neck": 11,
        "cloth": 12,
        "neck_l": 13,
        "skin": 14,
        "background": 15,
        "l_ear_r": 16,
        "r_ear_r": 17,
        "hat": 1,
        "eye_g": 0,
    }

    # Define the mapping of region weights to polygon names
    region_mapping = {
        "skin": "skin",
        "nose": "nose",
        "l_eye": "l_eye",
        "r_eye": "r_eye",
        "l_brow": "l_brow",
        "r_brow": "r_brow",
        "l_ear": "l_ear",
        "r_ear": "r_ear",
        "mouth": "mouth",
        "u_lip": "u_lip",
        "l_lip": "l_lip",
        "hair": "hair",
        "ear_r": ["l_ear_r", "r_ear_r"],
        "neck": "neck",
        "neck_l": "neck_l",
        "cloth": "cloth",
        "background": "background",
        "hat": "hat",
        "eye_g": "eye_g",
    }

    value = region_weights.pop("neck_l")

    # Add the item back to the dictionary at the end
    region_weights["neck_l"] = value

    # normalize 0-1
    # max_weight = max(region_weights.values())
    # region_weights = {k: v / max_weight for k, v in region_weights.items()}
    max_value = max(region_weights.values())
    min_value = min(region_weights.values())

    if max_value != min_value:
        # Normalize the values to [0, 1]
        region_weights = {
            key: (value - min_value) / (max_value - min_value)
            for key, value in region_weights.items()
        }

    # Create a colormap
    cmap = cm.get_cmap("coolwarm")
    # colors = cm.RdBu(np.linspace(0, 1, num_steps))
    # cmap = ListedColormap(colors)
    alpha = 1
    # Iterate over the regions and apply colors
    for region, weight in region_weights.items():
        # Get the corresponding polygon names
        polygon_names = region_mapping.get(region)

        # Check if multiple polygons are mapped to the same region
        if isinstance(polygon_names, list):
            for polygon_name in polygon_names:
                # print(polygon_name)
                # Get the segmentation polygon points
                polygon_points = annotation["shapes"][region_pos[polygon_name]][
                    "points"
                ]

                # Calculate the color based on the weight
                color = cmap(weight)

                # Create a polygon patch for visualization
                polygon_patch = patches.Polygon(
                    polygon_points,
                    closed=True,
                    fill=True,
                    facecolor=color,
                    alpha=alpha,
                    edgecolor="black",
                    linewidth=0,
                )
                ax[0].add_patch(polygon_patch)
                # ax[1].add_patch(polygon_patch)
        else:
            # print(polygon_names)
            # Calculate the color based on the weight
            color = cmap(weight)
            if polygon_names == "hat" or polygon_names == "eye_g":
                polygon_points = annotation_hat["shapes"][region_pos[polygon_names]][
                    "points"
                ]
                # Create a polygon patch for visualization
                polygon_patch = patches.Polygon(
                    polygon_points,
                    closed=True,
                    fill=True,
                    facecolor=color,
                    alpha=alpha,
                    edgecolor="black",
                    linewidth=0,
                )
                # ax[0].add_patch(polygon_patch)
                ax[1].add_patch(polygon_patch)
            else:
                polygon_points = annotation["shapes"][region_pos[polygon_names]][
                    "points"
                ]
                # Create a polygon patch for visualization
                polygon_patch = patches.Polygon(
                    polygon_points,
                    closed=True,
                    fill=True,
                    facecolor=color,
                    alpha=alpha,
                    edgecolor="black",
                    linewidth=0,
                )
                ax[0].add_patch(polygon_patch)
        # Show the plot
    plt.axis("off")
    plt.subplots_adjust(wspace=-0.46)

    # plt.savefig("facex.png", bbox_inches="tight")

    plt.show()
    return fig
