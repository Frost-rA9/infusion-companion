import cv2 as cv
bottle_roi = "./YoLo_train.txt"
output_path = "H:/new_bottle_black"

file = open(bottle_roi)
for file in file.readlines():
    file = file[:-1]
    file = file.split()
    bottle_path = file[0]
    roi = file[1]
    roi = list(map(int, roi.split(",")))[:-1]

    img = cv.imread(bottle_path)
    roi = img[roi[1]: roi[3], roi[0]: roi[2]]

    file_name = bottle_path.split("\\")[-1]
    output = output_path + "/" + file_name
    cv.imwrite(output, roi)
