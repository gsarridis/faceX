import pandas as pd
import os

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

from PIL import Image
from .farl import create_masks


def find_all_images(
    root_dir,
    extensions=(".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"),
):
    """Recursively find all image files in the directory."""
    image_paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(extensions):
                image_paths.append(os.path.join(dirpath, filename))
    return image_paths


def has_matching_files(directory, prefix):

    return any(f.startswith(prefix) for f in directory)


class CelebaDatasetMask(Dataset):
    """Custom Dataset for loading CelebA face images"""

    def __init__(
        self,
        csv_df,
        data_dir,
        att_dir,
        att_list,
        att_transform,
        att_crop,
        task,
        transform=None,
        get_path=False,
        get_att=True,
    ):
        self.data_dir = data_dir
        self.img_names = csv_df.index.values
        self.y = csv_df[task].values
        self.transform = transform
        self.att_dir = att_dir
        self.att_list = att_list
        self.att_transform = att_transform
        self.get_path = get_path
        self.get_att = get_att
        self.att_crop = att_crop

    def __getitem__(self, index):
        # Load the image and corresponding label
        img_path = os.path.join(self.data_dir, self.img_names[index])
        img = Image.open(img_path)
        img = self.transform(img) if self.transform else img

        label = self.y[index]

        # Load the attention map if needed
        atts = {}
        # att_all = torch.zeros((1, self.att_crop, self.att_crop))
        if self.get_att:
            for a in self.att_list:
                img_name = self.img_names[index].split(".")[0]
                img_name_all = img_name.zfill(5) + "_" + a + ".png"
                pth = os.path.join(
                    self.att_dir, str(int(img_name) // 2000), img_name_all
                )
                if os.path.isfile(pth):
                    att = Image.open(pth).convert("L")
                    att = self.att_transform(att)
                    atts[a] = att
                    background_att = torch.zeros_like(att) + 1
            att_dir = "skin"
            for att_other in list(atts.keys()):
                background_att -= atts[att_other]
                if att_dir != att_other:
                    atts[att_dir] -= atts[att_other]
            atts[att_dir][atts[att_dir] < 0] = 0
            background_att[background_att < 1] = 0

            atts["background"] = background_att
        # Prepare the return values
        return_values = [img, label]
        if self.get_att:
            return_values.append(atts)
        if self.get_path:
            return_values.append(img_path)

        return tuple(return_values)

    def __len__(self):
        return self.y.shape[0]


class CustomDatasetMask(Dataset):
    """Custom Dataset for loading  face images"""

    def __init__(
        self,
        csv_df,
        data_dir,
        att_dir,
        att_list,
        att_transform,
        att_crop,
        task,
        transform=None,
        get_path=False,
        get_att=True,
    ):
        self.data_dir = data_dir
        self.img_names = csv_df["img"].values
        self.y = csv_df[task].values
        self.transform = transform
        self.att_dir = att_dir
        self.att_list = att_list
        self.att_transform = att_transform
        self.get_path = get_path
        self.get_att = get_att
        self.att_crop = att_crop
        img_paths = self.get_all_img_paths(data_dir)
        self.processed_att_img_paths = []
        att_img_paths = self.get_all_img_paths(att_dir)
        for path in att_img_paths:
            new_path = path[:-4]
            self.processed_att_img_paths.append(new_path)
        if not self.check_any_path_starts_with():
            create_masks(
                dspth=self.data_dir, respth=self.att_dir, image_paths=img_paths
            )

    def check_any_path_starts_with(self):
        masks = find_all_images(self.att_dir)
        for path in self.processed_att_img_paths:
            if has_matching_files(masks, path):
                return True
                # continue
            else:
                return False
        return True

    def get_all_img_paths(self, d_dir):
        img_paths = []
        for index in range(len(self.img_names)):
            img_path = os.path.join(d_dir, self.img_names[index])
            img_paths.append(img_path)
        return img_paths

    def __getitem__(self, index):
        # Load the image and corresponding label
        img_path = os.path.join(self.data_dir, self.img_names[index])
        img = self.transform(img_path) if self.transform else Image.open(img_path)

        label = self.y[index]

        # Load the attention map if needed
        atts = {}
        # att_all = torch.zeros((1, self.att_crop, self.att_crop))
        if self.get_att:
            for a in self.att_list:
                img_name = self.img_names[index].split(".")[:-1]
                img_name = ".".join(img_name)
                img_name_all = img_name + "_" + a + ".png"
                pth = os.path.join(self.att_dir, img_name_all)
                if os.path.isfile(pth):
                    att = Image.open(pth).convert("L")
                    att = self.att_transform(att)
                    atts[a] = att
        # Prepare the return values
        return_values = [img, label]
        if self.get_att:
            return_values.append(atts)
        if self.get_path:
            return_values.append(img_path)

        return tuple(return_values)

    def __len__(self):
        return self.y.shape[0]


class CustomDatasetMaskImagePairs(Dataset):
    """Custom Dataset for loading  face image pairs"""

    def __init__(
        self,
        csv_df,
        data_dir,
        att_dir,
        att_list,
        att_transform,
        att_crop,
        task,
        transform=None,
        get_path=False,
        get_att=True,
    ):
        self.data_dir = data_dir
        # TODO: add img_names2 that found in column 1 of df
        # TODO: load 2 images, but get the att of only the second one, the first one acts as target!
        # theoriticaly DONE
        self.img_names = csv_df.iloc[:, 1].values
        self.img_names_reference = csv_df.iloc[:, 0].values
        self.y = csv_df[task].values
        self.transform = transform
        self.att_dir = att_dir
        self.att_list = att_list
        self.att_transform = att_transform
        self.get_path = get_path
        self.get_att = get_att
        self.att_crop = att_crop
        # TODO: maybe dont use all the data, only the ones that are involved in the csv! Ok, get_all_img_paths does exactly this job. So, already done.
        img_paths = self.get_all_img_paths(data_dir)
        self.processed_att_img_paths = []
        # TODO: the below lines make no sense. try to understand. Ok, this function doesnt search existing images, but says which paths should exist!
        att_img_paths = self.get_all_img_paths(att_dir)
        for path in att_img_paths:
            new_path = path[:-4]
            self.processed_att_img_paths.append(new_path)
        if not self.check_any_path_starts_with():
            create_masks(
                dspth=self.data_dir, respth=self.att_dir, image_paths=img_paths
            )

    def check_any_path_starts_with(self):
        masks = find_all_images(self.att_dir)
        for path in self.processed_att_img_paths:
            if has_matching_files(masks, path):
                # TODO: check here, propably we should use continue instead of return. I dont know why i comented out!
                return True
                # continue
            else:
                return False
        return True

    def get_all_img_paths(self, d_dir):
        img_paths = []
        for index in range(len(self.img_names)):
            img_path = os.path.join(d_dir, self.img_names[index])
            img_paths.append(img_path)
        return img_paths

    def __getitem__(self, index):
        # TODO: load 2 images, but get the att of only the second one, the first one acts as target!
        # theoriticaly DONE
        # Load the image and corresponding label
        img_path = os.path.join(self.data_dir, self.img_names[index])
        img_path_reference = os.path.join(
            self.data_dir, self.img_names_reference[index]
        )
        img = self.transform(img_path) if self.transform else Image.open(img_path)

        img_reference = (
            self.transform(img_path_reference) if self.transform else Image.open(img_path_reference)
        )

        label = self.y[index]

        # Load the attention map if needed
        atts = {}
        # att_all = torch.zeros((1, self.att_crop, self.att_crop))
        if self.get_att:
            for a in self.att_list:
                img_name = self.img_names[index].split(".")[:-1]
                img_name = ".".join(img_name)
                img_name_all = img_name + "_" + a + ".png"
                pth = os.path.join(self.att_dir, img_name_all)
                if os.path.isfile(pth):
                    att = Image.open(pth).convert("L")
                    att = self.att_transform(att)
                    atts[a] = att
        # Prepare the return values
        return_values = [img_reference, img, label]
        if self.get_att:
            return_values.append(atts)
        if self.get_path:
            return_values.append(img_path)

        return tuple(return_values)

    def __len__(self):
        return self.y.shape[0]


def balance_set(df, task, protected):
    p0 = 0
    p1 = 1

    df.loc[df[task] == -1, task] = 0
    df.loc[df[protected] == -1, protected] = 0
    if protected == "None":
        return df, p0, p1
    A1 = df[(df[protected] == p0)]
    A0 = df[(df[protected] == p1)]

    S0 = len(A0)
    S1 = len(A1)
    if S0 > S1:
        S0 = S1
    else:
        S1 = S0

    df_new = pd.concat(
        [
            A0.head(S0),
            A1.head(S1),
        ]
    )
    return df_new, p0, p1


def get_dataloaders(
    dataset,
    task,
    protected,
    data_dir,
    csv_dir,
    att_dir,
    att_list,
    img_size=128,
    bs=64,
    nw=4,
    one_class=-1,
    transform=None,
):
    if transform is None:
        transform_test = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
    else:
        transform_test = transform

    att_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ]
    )

    if dataset == "CelebAMask":
        if not protected == "None":
            df = pd.read_csv(csv_dir, sep="\s+", skiprows=1, usecols=[task, protected])
        else:
            df = pd.read_csv(csv_dir, sep="\s+", skiprows=1, usecols=[task])
        test_set, _, _ = balance_set(df, task, protected)

        if one_class > -1:
            test_set = test_set[(test_set[task] == one_class)]
        test_data = CelebaDatasetMask(
            test_set,
            data_dir,
            att_dir,
            att_list,
            att_transform,
            img_size,
            task,
            transform_test,
            get_path=True,
            get_att=True,
        )
    else:
        if not protected == "None":
            df = pd.read_csv(csv_dir, sep=",", usecols=["img", task, protected])
        else:
            df = pd.read_csv(csv_dir, sep=",", usecols=["img", task])

        if one_class > -1:
            df = df[(df[task] == one_class)]

        test_set, _, _ = balance_set(df, task, protected)

        test_data = CustomDatasetMask(
            test_set,
            data_dir,
            att_dir,
            att_list,
            att_transform,
            img_size,
            task,
            transform_test,
            get_path=True,
            get_att=True,
        )
    test_loader = DataLoader(
        dataset=test_data, batch_size=bs, shuffle=False, num_workers=nw
    )
    return test_loader


def get_dataloader_embeddings(
    #  dataset, #not needed
    task,
    protected,
    data_dir,
    csv_dir,
    att_dir,
    att_list,
    img_size=128,
    bs=64,
    nw=4,
    one_class=-1,
    transform=None,
):
    if transform is None:
        transform_test = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
    else:
        transform_test = transform

    att_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ]
    )

    # TODO: it doesn't make any sense to balance the data for the fv scenario.
    # Instead, try to select a subset of the data based on the selected protected attribute class beforehand (eg race-> black),
    # to show how the model makes decesions for a specific race.
    # TODO: maybe get the dataloader directly from mammoth-commons, see the relevant elmokhtar's file in slack.
    # or second option (better that first): use mammoth commons only to load the csv and then use facex to create the data class and loader.

    df = pd.read_csv(csv_dir, sep=",")

    if one_class > -1:
        df = df[(df[task] == one_class)]

    test_data = CustomDatasetMaskImagePairs(
        df,
        data_dir,
        att_dir,
        att_list,
        att_transform,
        img_size,
        task,
        transform_test,
        get_path=True,
        get_att=True,
    )
    test_loader = DataLoader(
        dataset=test_data, batch_size=bs, shuffle=False, num_workers=nw
    )
    return test_loader
