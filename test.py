import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize MediaPipe Hands and Drawing Utils
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hand_detector = mp_hands.Hands(max_num_hands=1)

# Initialize OpenCV face detector (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to detect hand gesture and count fingers
def detect_gesture(hand_landmarks):
    thumb_open = (hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x)
    index_open = (hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y)
    middle_open = (hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y)
    ring_open = (hand_landmarks.landmark[16].y < hand_landmarks.landmark[14].y)
    pinky_open = (hand_landmarks.landmark[20].y < hand_landmarks.landmark[18].y)
 
    if thumb_open and not index_open and not middle_open and not ring_open and not pinky_open:
        return "thumbs_up"
    elif not thumb_open and index_open and middle_open and not ring_open and not pinky_open:
        return "peace"
    elif thumb_open and index_open and middle_open and ring_open and pinky_open:
        return "five"
    elif not thumb_open and not index_open and not middle_open and not ring_open and not pinky_open:
        return "fist"
    else:
        return "unknown"

# Function to count fingers for gesture recognition
def count_fingers(hand_landmarks):
    cnt = 0
    thresh = (hand_landmarks.landmark[0].y * 100 - hand_landmarks.landmark[9].y * 100) / 2

    if (hand_landmarks.landmark[5].y * 100 - hand_landmarks.landmark[8].y * 100) > thresh:
        cnt += 1

    if (hand_landmarks.landmark[9].y * 100 - hand_landmarks.landmark[12].y * 100) > thresh:
        cnt += 1

    if (hand_landmarks.landmark[13].y * 100 - hand_landmarks.landmark[16].y * 100) > thresh:
        cnt += 1

    if (hand_landmarks.landmark[17].y * 100 - hand_landmarks.landmark[20].y * 100) > thresh:
        cnt += 1

    if (hand_landmarks.landmark[5].x * 100 - hand_landmarks.landmark[4].x * 100) > 6:
        cnt += 1

    return cnt

cap = cv2.VideoCapture(0)
start_time = time.time()
start_init = False
prev_fingers = -1
face_detected = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # Detect face
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
    face_detected = len(faces) > 0

    # Process hand landmarks
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hand_detector.process(frame_rgb)

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]

        # Count fingers and detect gesture
        fingers_count = count_fingers(hand_landmarks)
        gesture = detect_gesture(hand_landmarks)

        if not start_init and fingers_count != prev_fingers:
            start_time = time.time()
            start_init = True

        if start_init and time.time() - start_time > 0.2:
            if fingers_count == 1:
                pyautogui.press("right")
            elif fingers_count == 2:
                pyautogui.press("left")
            elif fingers_count == 3:
                pyautogui.press("up")
            elif fingers_count == 4:
                pyautogui.press("down")
            elif fingers_count == 5:
                pyautogui.press("space")
            prev_fingers = fingers_count
            start_init = False

        # Perform gesture-based media controls
        if gesture == "thumbs_up":
            pyautogui.press("volumeup")
        elif gesture == "peace":
            pyautogui.press("volumedown")
        elif gesture == "fist":
            pyautogui.press("nexttrack")

        # Draw landmarks on the frame
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display face detection status
    cv2.putText(frame, "Face Detected" if face_detected else "Face Not Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if face_detected else (0, 0, 255), 2)

    cv2.imshow("Media Controller", frame)

    if cv2.waitKey(1) == 27:  # Exit on pressing the 'Esc' key
        break

cap.release()
cv2.destroyAllWindows()
