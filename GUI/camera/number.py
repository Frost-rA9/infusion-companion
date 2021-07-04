class number:
    import cv2
    index = 0
    while True:
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            break
        else:
            index += 1
        cap.release()
