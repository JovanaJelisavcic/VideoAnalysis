import numpy as np
import cv2

video_capture = cv2.VideoCapture('video-0.avi')
while True:
    ret, frame = video_capture.read()
    if ret:
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
video_capture.release()
cv2.destroyAllWindows()