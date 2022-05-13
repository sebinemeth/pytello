import cv2

from detector import Detection, Detector

detector = Detector(adaptive=True)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

    detections = detector.detect(frame)
    frame = Detection.print_to_frame(detections, frame)

    cv2.putText(frame, f"{round(detector.fps, 2)} fps {'*' * detector.det_freq}", (8, 24), 0, 0.6, (255, 255, 255))

    cv2.imshow('Input', frame)

    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()
