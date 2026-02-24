import cv2
import mediapipe as mp

# mediapipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands()

cap = cv2.VideoCapture(0)

# finger tips
tips = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:

            lmList = []
            h, w, c = img.shape

            for id, lm in enumerate(hand.landmark):
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmList.append((cx, cy))

            fingers = []

            # THUMB
            if lmList[tips[0]][0] > lmList[tips[0]-1][0]:
                fingers.append(1)
            else:
                fingers.append(0)

            # OTHER 4 FINGERS
            for i in range(1,5):
                if lmList[tips[i]][1] < lmList[tips[i]-2][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            total = fingers.count(1)

            mp_draw.draw_landmarks(img, hand,
                                   mp_hands.HAND_CONNECTIONS)

            cv2.putText(img, str(total),
                        (50,100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        3,(0,255,0),5)

    cv2.imshow("Finger Counter", img)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()