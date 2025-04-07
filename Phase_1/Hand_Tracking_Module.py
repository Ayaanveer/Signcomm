import cv2 as cv
import mediapipe as mp
import time

class handDetector():
    def _init_(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode,
                                        max_num_hands=self.maxHands,
                                        min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence=self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLM in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLM, self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, draw=True):
        lmList = []
        bbox = []
        if self.results.multi_hand_landmarks:
            x_min, y_min = float('inf'), float('inf')
            x_max, y_max = -float('inf'), -float('inf')
            for handLms in self.results.multi_hand_landmarks:
                for lm in handLms.landmark:
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([cx, cy])
                    x_min, y_min = min(x_min, cx), min(y_min, cy)
                    x_max, y_max = max(x_max, cx), max(y_max, cy)
            bbox = [x_min, y_min, x_max, y_max]
            if draw:
                # Optionally draw bounding box
                cv.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        return [lmList, bbox]

    def distance(self, point1, point2):
        return (point1[1] - point2[1])*2 + (point1[2] - point2[2])*2

    def getFingers(self, img, handNo=0):
        fingers = [1, 1, 1, 1, 1]
        lmList, _ = self.findPosition(img)  # Get landmark positions.
        if not lmList:  # Check if no hand is detected.
            return None  # Return None if no hand is detected.

        try:
            # Logic for finger detection.
            if abs(lmList[3][1] - lmList[0][1]) < (lmList[2][1] - lmList[0][1]) or self.distance(lmList[0], lmList[2]) > self.distance(lmList[4], lmList[0]):
                fingers[0] = 0
            if self.distance(lmList[0], lmList[6]) > self.distance(lmList[8], lmList[0]):
                fingers[1] = 0
            if self.distance(lmList[0], lmList[10]) > self.distance(lmList[0], lmList[12]):
                fingers[2] = 0
            if self.distance(lmList[0], lmList[14]) > self.distance(lmList[0], lmList[16]):
                fingers[3] = 0
            if self.distance(lmList[0], lmList[18]) > self.distance(lmList[0], lmList[20]):
                fingers[4] = 0
        except Exception as ex:
            print(f"Error in finger detection: {ex}")
            return None  # Return None on error.

        return fingers


def main():
    pTime = 0
    cap = cv.VideoCapture(1)
    detector = handDetector()
    
    while True:
        success, img = cap.read()
        if not success:
            print("Failed to grab frame.")
            break

        img = detector.findHands(img)
        fingers = detector.getFingers(img)

        if fingers is None:  # Handle case where no hand is detected.
            print("No hand detected.")
            continue
        
        print(fingers)  # Print detected fingers.

        # Calculate FPS.
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        # Display FPS on the image.
        cv.putText(img, str(int(fps)), (10, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv.imshow('image', img)

        # Exit on pressing 'Esc'.
        if cv.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv.destroyAllWindows()