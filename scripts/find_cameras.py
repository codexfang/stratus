import cv2
for i in range(5):
    cap = cv2.VideoCapture(i)
    ok = cap.isOpened()
    cap.release()
    print(f"Camera {i}: {'OK' if ok else '—'}")
