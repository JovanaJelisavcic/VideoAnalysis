import numpy as np
import cv2

video_capture = cv2.VideoCapture('video-0.avi')
width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
while True:
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    minLineLength = 100
    maxLineGap = 10
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength, maxLineGap)
    for x1, y1, x2, y2 in lines[0]:
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

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
""" 
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([110, 50, 50])
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    res = cv2.bitwise_and(frame, frame, mask = mask)
    cv2.imshow('mask', mask)
    cv2.imshow('res', res)
    """