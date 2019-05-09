import cv2
import imutils

cap = cv2.VideoCapture('./video.mp4')

while cap.isOpened():
    res, frame = cap.read()
    imgray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(imgray, 230, 250, cv2.THRESH_BINARY)[1]
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        detector = cv2.SimpleBlobDetector_create()
        x, y, w, h = cv2.boundingRect(c)
        roi = frame[y:h+y, x:w+x]

        keypoints = detector.detect(roi)

        if(len(keypoints) > 0):
            frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, str(len(keypoints)), (x+w, y+h),
                        cv2.FONT_HERSHEY_TRIPLEX, 1.0, (0, 0, 255), lineType=cv2.LINE_AA)

        cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
