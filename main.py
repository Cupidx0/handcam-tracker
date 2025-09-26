import cv2
import math
import mediapipe as mp
import subprocess

def set_system_volume(volume_percent):
    # Clamp volume between 0 and 100
    volume = max(0, min(100, int(volume_percent)))
    script = f"set volume output volume {volume}"
    subprocess.run(["osascript", "-e", script])

drawingModule = mp.solutions.drawing_utils
handsModule = mp.solutions.hands

cap = cv2.VideoCapture(0)
with handsModule.Hands(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=1) as hands:
    while True:
        ret, frame = cap.read()
        frame1 = cv2.resize(frame, (640, 480))
        results = hands.process(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks is not None:
            for handLandmarks in results.multi_hand_landmarks:
                drawingModule.draw_landmarks(frame1, handLandmarks, handsModule.HAND_CONNECTIONS)
                # Get coordinates for index finger tip (8) and thumb tip (4)
                indexTip_pixel = drawingModule._normalized_to_pixel_coordinates(
                    handLandmarks.landmark[8].x, handLandmarks.landmark[8].y, 640, 480)
                thumbTip_pixel = drawingModule._normalized_to_pixel_coordinates(
                    handLandmarks.landmark[4].x, handLandmarks.landmark[4].y, 640, 480)
                if indexTip_pixel and thumbTip_pixel:
                    # Draw line between thumb and index
                    cv2.line(frame1, indexTip_pixel, thumbTip_pixel, (0, 255, 0), 2)
                    # Calculate distance between thumb and index
                    distance = math.sqrt(
                        (indexTip_pixel[0] - thumbTip_pixel[0]) ** 2 +
                        (indexTip_pixel[1] - thumbTip_pixel[1]) ** 2
                    )
                    # Map distance to volume (adjust min/max as needed)
                    min_dist = 30   # Minimum distance for 0% volume
                    max_dist = 200  # Maximum distance for 100% volume
                    volume = (distance - min_dist) / (max_dist - min_dist) * 100
                    volume = max(0, min(100, volume))
                    set_system_volume(volume)
                    cv2.putText(frame1, f"Volume: {int(volume)}%", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Frame", frame1)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC to quit
            break
cap.release()
cv2.destroyAllWindows()