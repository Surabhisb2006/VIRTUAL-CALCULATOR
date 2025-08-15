import cv2

cap = cv2.VideoCapture(0)
# Set camera resolution
cap.set(3, 1920)
cap.set(4, 1080)

# Make window full screen
cv2.namedWindow("Virtual Calculator - Day 7", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Virtual Calculator - Day 7", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("Webcam", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
