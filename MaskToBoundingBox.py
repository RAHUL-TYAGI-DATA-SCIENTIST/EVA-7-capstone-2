import json
import cv2
import torch

from skimage.measure import label, regionprops, find_contours
import os
import numpy as np
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader

ASSETS_DIRECTORY = "../dataset/"
import matplotlib.pyplot as plt
from skimage import measure

plt.rcParams["savefig.bbox"] = "tight"
import re

START_BOUNDING_BOX_ID = 1


def mask_to_border(mask):
    h, w = mask.shape
    border = np.zeros((h, w))

    contours = find_contours(mask, 128)
    for contour in contours:
        for c in contour:
            x = int(c[0])
            y = int(c[1])
            border[x][y] = 255

    return border


""" Mask to bounding boxes """


def mask_to_bbox(mask):
    bboxes = []

    mask = mask_to_border(mask)
    lbl = label(mask)
    props = regionprops(lbl)
    for prop in props:
        x1 = prop.bbox[1]
        y1 = prop.bbox[0]

        x2 = prop.bbox[3]
        y2 = prop.bbox[2]

        bboxes.append([x1, y1, x2, y2])

    return bboxes


def parse_mask(mask):
    mask = np.expand_dims(mask, axis=-1)
    mask = np.concatenate([mask, mask, mask], axis=-1)
    return mask


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    plt.show()


def masks_to_boxes(masks: torch.Tensor) -> torch.Tensor:
    """
    Compute the bounding boxes around the provided masks.
    Returns a [N, 4] tensor containing bounding boxes. The boxes are in ``(x1, y1, x2, y2)`` format with
    ``0 <= x1 < x2`` and ``0 <= y1 < y2``.
    Args:
        masks (Tensor[N, H, W]): masks to transform where N is the number of masks
            and (H, W) are the spatial dimensions.
    Returns:
        Tensor[N, 4]: bounding boxes
    """
    # if not torch.jit.is_scripting() and not torch.jit.is_tracing():
    #     _log_api_usage_once(masks_to_boxes)
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device, dtype=torch.float)

    n = masks.shape[0]

    bounding_boxes = torch.zeros((n, 4), device=masks.device, dtype=torch.float)

    for index, mask in enumerate(masks):
        y, x = torch.where(mask != 0)

        bounding_boxes[index, 0] = torch.min(x)
        bounding_boxes[index, 1] = torch.min(y)
        bounding_boxes[index, 2] = torch.max(x)
        bounding_boxes[index, 3] = torch.max(y)

    return bounding_boxes


def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour


def binary_mask_to_polygon(binary_mask, tolerance=0):
    """Converts a binary mask to COCO polygon representation
    Args:
        binary_mask: a 2D binary numpy array where '1's represent the object
        tolerance: Maximum distance from original points of polygon to approximated
            polygonal chain. If tolerance is 0, the original coordinate array is returned.
    """
    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
    contours = measure.find_contours(padded_binary_mask, 0.5)
    contours = np.subtract(contours, 1)
    for contour in contours:
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        # after padding and subtracting 1 we may get -0.5 points in our segmentation
        segmentation = [0 if i < 0 else i for i in segmentation]
        polygons.append(segmentation)

    return polygons


def build_coc_json(data, json_file):
    img_id = 0
    json_dict = {"images": [], "type": "instances", "annotations": [], "categories": []}
    cat = {
        "rebar": 1,
        "spall": 2,
        "crack": 3
    }
    for cate, cid in cat.items():
        cat_i = {"supercategory": "none", "id": cid, "name": cate}
        json_dict["categories"].append(cat_i)
    bnd_id = START_BOUNDING_BOX_ID

    for mask_path in data:
        # print(re.split('(\d+)', mask_path)[-1])
        img_id = img_id + 1
        img_name = mask_path.split(re.split('(\d+)', mask_path)[-1])[0] + str(".jpg")
        img_path = os.path.join(ASSETS_DIRECTORY, "images/" + img_name)
        mask_path = os.path.join(ASSETS_DIRECTORY, "masks/" + mask_path)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        boxes = mask_to_bbox(mask)
        image = {
            "file_name": img_name,
            "height": img.shape[0],
            "width": img.shape[1],
            "id": img_id,
        }
        for obj in boxes:
            xmin = obj[0]
            ymin = obj[1]
            xmax = obj[2]
            ymax = obj[3]
            o_width = abs(xmax - xmin)
            o_height = abs(ymax - ymin)
            assert xmax > xmin
            assert ymax > ymin
            ann = {
                "area": o_width * o_height,
                "iscrowd": 0,
                "image_id": img_id,
                "bbox": [xmin, ymin, o_width, o_height],
                "category_id": cat[re.split('(\d+)', mask_path)[-1].split(".")[0]],
                "id": bnd_id,
                "ignore": 0,
                "segmentation": binary_mask_to_polygon(mask, .85),
            }
            json_dict["annotations"].append(ann)
            bnd_id = bnd_id + 1

        json_dict["images"].append(image)

    print(json_dict)
    json_fp = open(json_file, "w")
    json_str = json.dumps(json_dict)
    json_fp.write(json_str)
    json_fp.close()


if __name__ == '__main__':
    path = "../dataset/"
    os.chdir(path)
    full_dataset = os.listdir("masks/")
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    print(train_dataset.indices)
    print(test_dataset.indices)

    train_data = DataLoader(train_dataset, batch_size=len(train_dataset.indices))
    test_data = DataLoader(test_dataset, batch_size=len(train_dataset.indices))
    train_data = iter(train_data).next()
    test_data = iter(test_data).next()

    print(train_data)
    print(test_data)

    build_coc_json(train_data, "instances_train2017.json")
    build_coc_json(test_data, "instances_val2017.json")