import sys
from albumentations.augmentations.transforms import RandomResizedCrop

# sys.path.insert(0, '/opt/ml/code/yolov4')

import cv2
import numpy as np
import albumentations
import albumentations.pytorch

from torch.utils.data import DataLoader
from pprint import pprint

from dataset import RecycleTrashDataset, collater


# denormalize function
def denormalize_image(image, mean, std):
    img_cp = image.copy()
    img_cp *= std
    img_cp += mean
    img_cp *= 255.0
    img_cp = np.clip(img_cp, 0, 255).astype(np.uint8)
    return img_cp


# Class index
CLASSES = [
    "UNKNOWN",
    "General trash",
    "Paper",
    "Paper pack",
    "Metal",
    "Glass",
    "Plastic",
    "Styrofoam",
    "Plastic bag",
    "Battery",
    "Clothing",
]

# bbox Color
COLORS = [
    (39, 129, 113),
    (164, 80, 133),
    (83, 122, 114),
    (99, 81, 172),
    (95, 56, 104),
    (37, 84, 86),
    (14, 89, 122),
    (80, 7, 65),
    (10, 102, 25),
    (90, 185, 109),
    (106, 110, 132),
]

tmp = []
value = 50

for i in COLORS:
    B = i[0] + value if i[0] + value < 255 else 255
    G = i[1] + value if i[1] + value < 255 else 255
    R = i[2] + value if i[2] + value < 255 else 255

    brightness = (B, G, R)

    tmp.append(brightness)

COLORS2 = tmp  # Filled Colors


# basic transform
"""
CAUTION : You must specify the BboxParams.
"""
transform = albumentations.Compose(
    [
        albumentations.RandomResizedCrop(224, 224, p=0.5),
        albumentations.Resize(512, 512),
        albumentations.Normalize(mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25)),
        albumentations.pytorch.transforms.ToTensorV2(),
    ],
    bbox_params=albumentations.BboxParams(format="pascal_voc"),
)

# Simple Dataset Unit Test
dataset = RecycleTrashDataset(transform=transform)

sample = dataset.__getitem__(0)
img = sample["img"].permute(1, 2, 0).detach().cpu().numpy()
img = denormalize_image(img, mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25))
# cv2.imwrite('./example.jpg', img)
# pprint(sample['annot'])


# Dataloader Unit Test
dataloader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collater)


for i, batch_sample in enumerate(dataloader):
    imgs, annots = batch_sample["img"], batch_sample["annot"]
    print(
        imgs.shape, annots.shape
    )  # img shape = (batch_size, 3, width, height), annot shape = (batch_size, max_num_bbox, 5(xmin, ymin, xmax, ymax, class label))
    img_1, img_2 = (
        imgs[0].permute(1, 2, 0).detach().cpu().numpy(),
        imgs[1].permute(1, 2, 0).detach().cpu().numpy(),
    )
    # print(img_1.shape, type(img_1))
    for j in range(annots.shape[0]):
        output_img = imgs[j].permute(1, 2, 0).detach().cpu().numpy()
        output_img = denormalize_image(
            output_img, mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25)
        )  # denormalize image for visualization
        annot = annots[j].detach().cpu().numpy()
        for box_id in range(annot.shape[0]):
            # if target bbox is just padding -> break
            if annot[box_id][-1] < 0:
                break

            # for cv2.rectangle arguments type
            boxes = np.int64(annot[box_id])

            # set class label
            label = int(annot[box_id][-1].item())

            # set bbox coordinates
            xmin, ymin, xmax, ymax = boxes[:4]

            # set color
            color = COLORS[label]
            color2 = COLORS2[label]

            # draw bbox
            cv2.rectangle(output_img, (xmin, ymin), (xmax, ymax), color, 2)
            tmp_img = output_img.copy()
            cv2.rectangle(tmp_img, (xmin, ymin), (xmax, ymax), color2, cv2.FILLED)
            output_img = cv2.addWeighted(output_img, 0.5, tmp_img, 0.5, 0)
            text_size = cv2.getTextSize(CLASSES[label], cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]

            cv2.rectangle(
                output_img,
                (xmin, ymin),
                (xmin + text_size[0] + 2, ymin + text_size[1] + 6),
                color,
                -1,
            )
            cv2.putText(
                output_img,
                CLASSES[label],
                (xmin, ymin + text_size[1] + 4),
                cv2.FONT_ITALIC,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        cv2.imwrite(f"./example_image_{j}.jpg", output_img)
    print(annots[0].shape, annots[1].shape)
    pprint(annots)
    break
