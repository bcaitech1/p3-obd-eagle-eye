from pycocotools.coco import COCO

import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--csv', required=True, help='submission file')
parser.add_argument('--stop', type=int, default=50, help='submission file') # max 838

args = parser.parse_args()
csv = pd.read_csv(args.csv)
STOP = args.stop

dataset_path = "/opt/ml/input/data"
coco = COCO('/opt/ml/input/data/test.json')
imag_ids = coco.getImgIds()

save_dir = '/opt/ml/ViewCSV'

os.makedirs('/opt/ml/ViewCSV/batch_01_vt', exist_ok=True)
os.makedirs('/opt/ml/ViewCSV/batch_02_vt', exist_ok=True)
os.makedirs('/opt/ml/ViewCSV/batch_03', exist_ok=True)

# text 간격
pos = 3 
classes_pos = list(range(0, 11*pos, pos))

classes = ("UNKNOWN", "General trash", "Paper", "Paper pack", "Metal", "Glass",
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

# colors = [
#     (int('0x77', 16), int('0x9c', 16), int('0x74', 16)),
#     (int('0xe8', 16), int('0x52', 16), int('0x55', 16)),
#     (int('0x7f', 16), int('0x09', 16), int('0x09', 16)),
#     (int('0x1d', 16), int('0x2b', 16), int('0x32', 16)),
#     (int('0x33', 16), int('0x29', 16), int('0x43', 16)), 
#     (int('0x22', 16), int('0x4c', 16), int('0x66', 16)), 
#     (int('0xfc', 16), int('0xfc', 16), int('0xe8', 16)), 
#     (int('0xfe', 16), int('0xf7', 16), int('0x7b', 16)), 
#     (int('0xff', 16), int('0xa5', 16), int('0x00', 16)), 
#     (int('0x6f', 16), int('0x0c', 16), int('0x4f', 16)), 
#     (int('0x00', 16), int('0x99', 16), int('0xcc', 16)), 
#     ]

colors = [
    ('#779c74'),
    ('#e85255'),
    ('#7f0909'),
    ('#1d2b32'),
    ('#332943'), 
    ('#224c66'), 
    ('#fcfce8'), 
    ('#fef77b'), 
    ('#ffa500'), 
    ('#6f0c4f'), 
    ('#0099cc'), 
    ]


for i in range(len(csv)):
    image_name = csv.loc[i][1]
    save_path = os.path.join(save_dir, image_name)
    images = cv2.imread(os.path.join(dataset_path, image_name))
    predicts = np.reshape(csv.loc[i][0].split(' ')[:-1], (-1, 6)).astype(np.float)

    fig, ax = plt.subplots()
    ax.imshow(images)
    
    for o in predicts:
        rect = patches.Rectangle(
            (o[2], o[3]), # x, y
            (o[4]-o[2]),  # w
            (o[5]-o[3]),  # h
            linewidth=1,  # line 굵기
            edgecolor=colors[int(o[0])], 
            facecolor='none')
        ax.add_patch(rect)
        label = classes[int(o[0])]
        text = f'{label}|{o[1]:0.2f}'
        ax.text(o[2], o[3]-2-classes_pos[int(o[0])], text, color = colors[int(o[0])])
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

    if i == STOP:
        break