import cv2
import mediapipe as mp
import time
import numpy as np
import joblib
import os


class HandDetector():
    def __init__(
            self,
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
    ):
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mp_hands = mp.solutions.hands

        self.hands = self.mp_hands.Hands(
            static_image_mode=self.static_image_mode,
            max_num_hands=self.max_num_hands,
            model_complexity=self.model_complexity,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )

        self.mp_draw = mp.solutions.drawing_utils

    def find_hands(self, img, *, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        if self.results.multi_hand_landmarks:
            for hand_lm in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, hand_lm, self.mp_hands.HAND_CONNECTIONS)
        return img
    
    def find_pos(self, img, *, draw=True):
        lm_list = []
        if self.results.multi_hand_landmarks:
           for hand in self.results.multi_hand_landmarks:
                hand_lm_list = []
                for id, lm in enumerate(hand.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x*w), int(lm.y*h)
                    hand_lm_list.append([id, cx, cy])
                lm_list.append(hand_lm_list)
        return lm_list


def main():
    model_path = os.path.join(os.getcwd(), 'classifier', 'tree.pkl')
    model_data = joblib.load(model_path)
    clf = model_data["model"]
    labels = model_data["labels"]

    p_time = 0
    c_time = 0

    cap = cv2.VideoCapture(0)
    detector = HandDetector()

    while True:
        success, img = cap.read()
        img = detector.find_hands(img, draw=True)
        lm_list = detector.find_pos(img)

        pose_text = "NONE"

        if lm_list and len(lm_list[0]) == 21:
            hand = sorted(lm_list[0], key=lambda x: x[0])

            wx, wy = hand[0][1], hand[0][2]
            mx, my = hand[9][1], hand[9][2]
            scale = ((mx - wx) ** 2 + (my - wy) ** 2) ** 0.5

            if scale != 0:
                vec = []
                for _, cx, cy in hand:
                    vec.extend([(cx - wx) / scale, (cy - wy) / scale])
                vec = np.array(vec, dtype=np.float32)

                pose_text = clf.predict([vec])[0]


        c_time = time.time()
        fps = 1/(c_time - p_time)
        p_time = c_time

        cv2.putText(img, pose_text, (10, 140), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 3)
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 255, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()