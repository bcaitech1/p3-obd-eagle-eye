import os
import pandas as pd
import numpy as np

from ensemble_boxes import *
from pycocotools.coco import COCO

# ensemble csv files # submission_files = ['submission.csv']
submission_files = os.listdir('./xfor_ensemble_list') # ensemble할 csv 파일이 존재하는 디렉토리 위치
# print(submission_files)
submission_df = [pd.read_csv(f'/opt/ml/xfor_ensemble_list/{file}') for file in submission_files]
print(submission_df)

image_ids = submission_df[0]['image_id'].tolist()

annotation = '/opt/ml/input/data/test.json'
coco = COCO(annotation)

'''
Method  |  mAP(0.5) Result Best params      |    Elapsed time (sec)
NMS	      0.5642	 IOU Thr: 0.5	                    47
Soft-NMS  0.5616	 Sigma: 0.1, Confidence Thr: 0.001	88
NMW	      0.5667	 IOU Thr: 0.5	                    171
WBF	      0.5982	 IOU Thr: 0.6	                    249
'''

prediction_strings = []
file_names = []
iou_thr = 0.4
sigma = 0.1
skip_box_thr = 0.0001

for i, image_id in enumerate(image_ids):
    prediction_string = ''
    boxes_list = []
    scores_list = []
    labels_list = []
    image_info = coco.loadImgs(i)[0]
    for df in submission_df:
        predict_string = df[df['image_id'] == image_id]['PredictionString'].tolist()[0]
        predict_list = str(predict_string).split()
        if len(predict_list)==0 or len(predict_list)==1:
            continue
        predict_list = np.reshape(predict_list, (-1, 6))
        box_list = []
        for box in predict_list[:, 2:6].tolist():
            box[0] = float(box[0]) / image_info['width']
            box[1] = float(box[1]) / image_info['height']
            box[2] = float(box[2]) / image_info['width']
            box[3] = float(box[3]) / image_info['height']
            box_list.append(box)
        boxes_list.append(box_list)
        scores_list.append(list(map(float, predict_list[:, 1].tolist())))
        labels_list.append(list(map(int, predict_list[:, 0].tolist())))
    
    if len(boxes_list):
        # boxes, scores, labels = nms(
        # boxes_list, 
        # scores_list, 
        # labels_list, 
        # # weights=weights, 
        # iou_thr=iou_thr)

        # boxes, scores, labels = soft_nms(
        #     boxes_list, 
        #     scores_list, 
        #     labels_list, 
        #     # weights=weights, 
        #     iou_thr=iou_thr, 
        #     sigma=sigma, 
        #     thresh=skip_box_thr
        #     )

        boxes, scores, labels = non_maximum_weighted(
            boxes_list, 
            scores_list, 
            labels_list, 
            # weights=weights, 
            iou_thr=iou_thr, 
            skip_box_thr=skip_box_thr
            )

        boxes, scores, labels = weighted_boxes_fusion(
            boxes_list, 
            scores_list, 
            labels_list, 
            # weights=weights, 
            iou_thr=iou_thr, 
            skip_box_thr=skip_box_thr
            )

        for box, score, label in zip(boxes, scores, labels):
            prediction_string += str(label) + ' ' + str(score) + ' ' + str(box[0] * image_info['width']) + ' ' + str(box[1] * image_info['height']) + ' ' + str(box[2] * image_info['width']) + ' ' + str(box[3] * image_info['height']) + ' '
    
    prediction_strings.append(prediction_string)
    file_names.append(image_id)

submission = pd.DataFrame()
submission['PredictionString'] = prediction_strings
submission['image_id'] = file_names
submission.to_csv('first_ensemble_weighted_boxes_fusion.csv', index=None)

submission.head()    


# # if __name__ == '__main__':
# #     pass