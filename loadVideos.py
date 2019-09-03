import numpy as np
import cv2

video_capture = cv2.VideoCapture('video-0.avi')
width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
while True:
    ret, frame = video_capture.read()
    if ret:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, 'ZBIR: ', (width-150, height-20), font, 1, (255, 255, 255), 5, cv2.LINE_AA)
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
video_capture.release()
cv2.destroyAllWindows()