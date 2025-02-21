import pandas as pd
import cv2
import Hand_Tracking_Module as htm
import os
import math

rows = []
dec = htm.handDetector(maxHands=2, detectionCon=0, trackCon=0)
folder_path = "D:\\Live_ISL_Translator\\Images"

for item in os.listdir(folder_path):
    img_folder = os.path.join(folder_path, item)
    if(os.path.isdir(img_folder)):
        count = 0
        for file in os.listdir(img_folder):
            if(file.endswith('.jpg')):
                os.path.join(img_folder, file)
                img = cv2.imread(os.path.join(img_folder, file))
                img = dec.findHands(img)
                cv2.imshow("Image",img)
                if(dec.results.multi_hand_landmarks is None):
                    continue
                cv2.waitKey(10)
                count+=1
                landmarks = dec.findPosition(img, 0)
                wrist = landmarks[0][0]
                bound_box = landmarks[1]
                data = []
                for i in range(0,21,4):
                    data.append(float(landmarks[0][i][1]-bound_box[0])/float(bound_box[2]-bound_box[0]))
                    data.append(float(landmarks[0][i][2]-bound_box[1])/float(bound_box[3]-bound_box[1]))
                if(len(dec.results.multi_hand_landmarks) == 2):
                    landmarks = dec.findPosition(img, 1)
                    bound_box = landmarks[1]
                    for i in range(0,21,4):
                        data.append(float(landmarks[0][i][1]-bound_box[0])/float(bound_box[2]-bound_box[0]))
                        data.append(float(landmarks[0][i][2]-bound_box[1])/float(bound_box[3]-bound_box[1]))
                    wrist2 = landmarks[0][0]
                    data.append(math.dist(wrist[1:], wrist2[1:]))
                else:
                    for _ in range(13):
                        data.append(0)
                data.append(str(item))
                # print(data)
                rows.append(data)
                if(count == 80):
                    break

df = pd.DataFrame(rows)
df.to_csv('data_2.csv')