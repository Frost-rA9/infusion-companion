import cv2

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
while ret:
    cv2.imshow("capture", frame)
    ret, frame = cap.read()
    k = cv2.waitKey(10) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()
