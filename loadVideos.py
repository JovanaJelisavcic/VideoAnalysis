import numpy as np
import cv2


video_capture = cv2.VideoCapture('video-0.avi')


width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

ret, frame = video_capture.read()

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150, apertureSize=3)
minLineLength = 100
maxLineGap = 10
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength, maxLineGap)

x1_min, y2_min, x2_max, y1_max = 50000, 50000, -1, -1

for line in lines:
    x11, y11, x22, y22 = line[0]
    if x11 < x1_min:
        x1_min = x11
    if y11 > y1_max:
        y1_max = y11
    if x22 > x2_max:
        x2_max = x22
    if y22 < y2_min:
        y2_min = y22

while True:
    ret, frame = video_capture.read()

    cv2.line(frame, (x1_min, y1_max), (x2_max, y2_min), (0, 0, 255), thickness=2)
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
