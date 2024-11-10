import cv2
import mediapipe as mp
import numpy as np
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL
import math

# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Access audio interface
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)

# Volume adjustment helper
def adjust_volume(change):
    """Adjust volume by the specified change and display new volume."""
    current_volume = volume.GetMasterVolumeLevelScalar() * 100
    print(f"Current Volume: {current_volume}%")  # Debug print
    new_volume = max(0, min(100, current_volume + change))
    print(f"New calculated volume: {new_volume}%")  # Debug print
    volume.SetMasterVolumeLevelScalar(new_volume / 100)
    print(f"Volume set to: {new_volume}%")  # Debug print
    return new_volume

# Function to check if the camera is accessible
def check_camera():
    cap = cv2.VideoCapture(0)  # Try to open default camera
    if not cap.isOpened():
        print("Error: Could not open default camera.")
        # Try an alternative index for external cameras if available
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("Error: Could not open alternative camera.")
            return None
    return cap

# Capture webcam input
cap = check_camera()
if cap is None:
    print("Error: Camera not accessible.")
    exit()

# Initialize variables
initial_angle = None
rotation_count = 0
rotation_threshold = 30  # Threshold for detecting significant rotation
fingertip_history = []  # History of fingertip positions for tracking

# Initialize volume level
current_volume = adjust_volume(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Error: Failed to capture image from camera.")
        break

    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Get points for index fingertip and wrist
        fingertip = hand_landmarks.landmark[8]  # Index fingertip
        base = hand_landmarks.landmark[0]  # Wrist base

        # Convert to pixel values
        h, w, _ = image.shape
        fingertip_pos = (int(fingertip.x * w), int(fingertip.y * h))
        base_pos = (int(base.x * w), int(base.y * h))

        # Draw hand landmarks and track fingertip
        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        fingertip_history.append(fingertip_pos)

        # Display fingertip trail
        for i in range(1, len(fingertip_history)):
            cv2.line(image, fingertip_history[i - 1], fingertip_history[i], (0, 255, 0), 2)
        if len(fingertip_history) > 20:
            fingertip_history.pop(0)  # Limit trail length

        # Calculate the angle of the fingertip relative to the wrist
        dx, dy = fingertip_pos[0] - base_pos[0], fingertip_pos[1] - base_pos[1]
        current_angle = math.degrees(math.atan2(dy, dx))

        print(f"Fingertip: {fingertip_pos}, Base: {base_pos}, Angle: {current_angle}")  # Debug print

        if initial_angle is None:
            # Set initial angle at the start
            initial_angle = current_angle
        else:
            # Calculate angle difference
            angle_diff = current_angle - initial_angle
            print(f"Angle difference: {angle_diff}")  # Debug print

            if angle_diff > rotation_threshold:
                print(f"Angle difference {angle_diff} > {rotation_threshold}, increasing volume.")
                current_volume = adjust_volume(10)  # Clockwise, increase volume
                print(f"Volume increased to: {current_volume}")  # Debug print
                initial_angle = current_angle
            elif angle_diff < -rotation_threshold:
                print(f"Angle difference {angle_diff} < {-rotation_threshold}, decreasing volume.")
                current_volume = adjust_volume(-10)  # Anti-clockwise, decrease volume
                print(f"Volume decreased to: {current_volume}")  # Debug print
                initial_angle = current_angle

    # Draw volume level icon
    cv2.rectangle(image, (10, 10), (110, 60), (0, 0, 0), -1)
    cv2.putText(image, f"Volume: {int(current_volume)}%", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow('Hand Gesture Volume Control', image)

    if cv2.waitKey(5) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
