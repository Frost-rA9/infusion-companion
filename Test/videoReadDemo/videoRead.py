import cv2
import numpy

cap = cv2.VideoCapture("../TestFiles/0003.avi")
ret, frame = cap.read()
while ret:
    cv2.imshow("capture", frame)
    ret, frame = cap.read()
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
