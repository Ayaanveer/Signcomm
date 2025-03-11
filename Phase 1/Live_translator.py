import cv2
import Hand_Tracking_Module as htm
import math
import pickle

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

value = "No Hands Detected"

dec = htm.handDetector(maxHands=2)

model = None

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
f.close()

while(True):
    success, img = cap.read()
    cv2.putText(img, f'{value}', (40,70), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 3)
    img = dec.findHands(img)
    landmarks = dec.findPosition(img, draw = False)
    cv2.imshow("Live Image",img)
    cv2.waitKey(1)
    if(dec.results.multi_hand_landmarks is None):
        value = "No Hands Detected"
    else:
        bound_box = landmarks[1]
        wrist = landmarks[0][0]
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
        y = model.predict([data])
        value = str(y[0])