import cv2
import Hand_Tracking_Module as htm
import math
import numpy as np
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.layers import DepthwiseConv2D # type: ignore
import time

# Load Keras model
custom_objects = {'DepthwiseConv2D': DepthwiseConv2D}
try:
    keras_model = load_model(r'C:\Users\ayaan\OneDrive\Desktop\hp\SignComm\Phase_1\keras_model.h5', custom_objects=custom_objects)
except Exception as e:
    print(f"Error loading Keras model: {e}")
    exit()

cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

value = "No Hands Detected"

dec = htm.handDetector(maxHands=2)

class_labels = ["Hello", "Thank You", "Yes", "No", "Help"]  # Replace with your classes

while True:
    success, img = cap.read()
    cv2.putText(img, f'{value}', (40, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
    img = dec.findHands(img)
    landmarks = dec.findPosition(img, draw=False)
    cv2.imshow("Live Image", img)

    if dec.results.multi_hand_landmarks is None:
        value = "No Hands Detected"
    else:
        bound_box = landmarks[1]
        wrist = landmarks[0][0]
        data = []

        for i in range(0, 21, 4):
            data.append(float(landmarks[0][i][1] - bound_box[0]) / float(bound_box[2] - bound_box[0]))
            data.append(float(landmarks[0][i][2] - bound_box[1]) / float(bound_box[3] - bound_box[1]))

        if len(dec.results.multi_hand_landmarks) == 2:
            landmarks = dec.findPosition(img, 1)
            bound_box = landmarks[1]
            for i in range(0, 21, 4):
                data.append(float(landmarks[0][i][1] - bound_box[0]) / float(bound_box[2] - bound_box[0]))
                data.append(float(landmarks[0][i][2] - bound_box[1]) / float(bound_box[3] - bound_box[1]))
            wrist2 = landmarks[0][0]
            data.append(math.dist(wrist[1:], wrist2[1:]))
        else:
            for _ in range(13):
                data.append(0)

        # Predict and Translate Sign Language
        data_array = np.array([data])
        y = keras_model.predict(data_array)
        value = class_labels[np.argmax(y)]  # Map prediction to class label

    # Break condition to stop program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
