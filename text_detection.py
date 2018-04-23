import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import pytesseract
from PIL import Image
import time

start = time.time()

IMG_PATH = os.getcwd() + '/images'

STRONG_CANNY = 50
LOW_CANNY = 30
BINARY_THRESHOLD = 77
RECTANGLE = 'inner'  # or rotated or dfs
DFS = False

MEDIAN_SIZE = 3
OPENING = False
OPENING_SIZE = (3, 1)
DILATATION_SIZE_1 = (9, 1)
DILATATION_SIZE_2 = (1, 5)

DILATATION_ITERATIONS = 1
WIDTH_THRESHOLD = 0.04
HEIGHT_THRESHOLD = 0.007


VERBOSE = 1

def process_image(img):
    bgr = cv2.imread(img)

    print('w_ratio {}'.format(WIDTH_THRESHOLD/float(bgr.shape[1])))
    print('h_ratio {}'.format(HEIGHT_THRESHOLD/float(bgr.shape[0])))
    if VERBOSE == 1:
        cv2.namedWindow("bgr", cv2.WINDOW_NORMAL)
        cv2.imshow("bgr", bgr)
        cv2.waitKey(0)

    imgray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    if VERBOSE == 1:
        cv2.namedWindow("grayscale", cv2.WINDOW_NORMAL)
        cv2.imshow("grayscale", imgray)
        cv2.waitKey(0)

    canny = cv2.Canny(imgray, LOW_CANNY, STRONG_CANNY)

    if VERBOSE == 1:
        cv2.namedWindow("canny", cv2.WINDOW_NORMAL)
        cv2.imshow("canny", canny)
        cv2.waitKey(0)

    # create binary image
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    (t, binary) = cv2.threshold(gray, BINARY_THRESHOLD, 255, cv2.THRESH_BINARY)

    # find contours
    (_, contours, _) = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

    # print table of contours and sizes
    print("Found %d objects." % len(contours))
    max_idx = 0
    max_len = 0
    for (i, c) in enumerate(contours):
        print("\tSize of contour %d: %d" % (i, len(c)))
        if len(c) > max_len:
            max_len = len(c)
            max_idx = i

    if VERBOSE == 1:
        # draw contours over original image
        cv2.drawContours(bgr, contours, max_idx, (0, 0, 255), 5)

    if RECTANGLE == "straight":
        # straight bounding rectangle
        x, y, w, h = cv2.boundingRect(contours[max_idx])
        bgr_ext = cv2.rectangle(bgr, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # set all points outside the rectangle to 0
        canny[:y, :] = 0
        canny[:, :x] = 0
        canny[:, x+w:] = 0
        canny[y+h:, :] = 0

        if DFS:
            sub_image = canny[y:y+h,x:x+w]

            queue = [[0, 0], [0, sub_image.shape[1]-1],
                    [sub_image.shape[0]-1, 0], [sub_image.shape[0]-1, sub_image.shape[1]-1]]

            seen = {}
            seen[(0, 0)] = 1
            seen[(0, sub_image.shape[1]-1)] = 1
            seen[(sub_image.shape[0]-1, 0)] = 1
            seen[(sub_image.shape[0]-1, sub_image.shape[1]-1)] = 1

            contour_dic = {}
            
            for idx, pix in enumerate(contours[max_idx]):
                contour_dic[(pix[0][0], pix[0][1])] = 1

            def get_neighbors(y, x, img, contour, seen):
                all_neighbors = [[y-1, x-1], [y-1, x], [y-1, x+1],
                                [y, x+1], [y+1, x+1], [y+1, x], [y+1, x-1], [y, x-1]]
                valid = []

                for idx in range(len(all_neighbors)):
                    this_x = all_neighbors[idx][1]
                    this_y = all_neighbors[idx][0]
                    if 0 <= this_y < img.shape[0]:
                        if 0 <= this_x < img.shape[1]:
                            if (this_y, this_x) not in seen.keys():
                                if (this_y, this_x) not in contour.keys():
                                    valid += [[this_y, this_x]]
                                    seen[(this_y, this_x)] = 1
                return valid, seen

            while queue != []:
                this = queue.pop(-1)
                while this in seen.keys():
                    this = queue.pop(-1)

                neighbors, seen = get_neighbors(
                    this[0], this[1], sub_image, contour_dic, seen)
                queue += neighbors

                canny[this[0] + y, this[1] + x] = 0

    elif RECTANGLE == 'inner':
        # find the top side of contour
        contours_concat = np.concatenate(contours[max_idx])

        np.savetxt("contours.csv",contours_concat)
        
        left = int(np.percentile(contours_concat[:,0],15))
        right = int(np.percentile(contours_concat[:,0],85))

        top =  int(np.percentile(contours_concat[:,1],15))
        bottom =  int(np.percentile(contours_concat[:,1],85))


        bgr_ext = cv2.rectangle(bgr, (left, top), (right, bottom), (0, 255, 0), 2)

        # set all points outside the rectangle to 0
        canny[:top, :] = 0
        canny[:, :left] = 0
        canny[:, right:] = 0
        canny[bottom:, :] = 0       

    elif RECTANGLE == "rotated":
        # rotated bounding rectangle
        rect = cv2.minAreaRect(contours[max_idx])
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        bgr_ext = cv2.drawContours(bgr, [box], 0, (0, 255, 0), 2)

    if VERBOSE == 1:
        # display original image with contours
        cv2.namedWindow("contours", cv2.WINDOW_NORMAL)
        cv2.imshow("contours", bgr_ext)
        cv2.waitKey(0)

    if VERBOSE == 1:
        # display original image with contours
        cv2.namedWindow("cropped canny", cv2.WINDOW_NORMAL)
        cv2.imshow("cropped canny", canny)
        cv2.waitKey(0)


    for pix in contours[max_idx]:
        canny[pix[0][1],pix[0][0]] = 0
    
    median = cv2.medianBlur(canny, MEDIAN_SIZE)

    if VERBOSE == 1:
        # display original image with contours
        cv2.namedWindow("median", cv2.WINDOW_NORMAL)
        cv2.imshow("median", median)
        cv2.waitKey(0)

    if OPENING:
        # opening operation
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, OPENING_SIZE)
        median = cv2.morphologyEx(median, cv2.MORPH_OPEN, kernel)

        if VERBOSE == 1:
            # display original image with contours
            cv2.namedWindow("opening", cv2.WINDOW_NORMAL)
            cv2.imshow("opening", median)
            cv2.waitKey(0)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, DILATATION_SIZE_1)
    dilation_1 = cv2.dilate(median, kernel, iterations=DILATATION_ITERATIONS)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, DILATATION_SIZE_2)
    dilation = cv2.dilate(dilation_1, kernel, iterations=DILATATION_ITERATIONS)

    if VERBOSE == 1:
        # display original image with contours
        cv2.namedWindow("dilatation", cv2.WINDOW_NORMAL)
        cv2.imshow("dilatation", dilation)
        cv2.waitKey(0)

    # find contours again
    (_, contours, _) = cv2.findContours(dilation, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

    blocs = []
    bgr_int = cv2.imread(img)
    for idx in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[idx])
        blocs += [[x, y, w, h]]
        bgr_int = cv2.rectangle(bgr_int, (x, y), (x+w, y+h), (0, 255, 0), 2)

    if VERBOSE == 1:
        # display original image with contours
        cv2.namedWindow("contours", cv2.WINDOW_NORMAL)
        cv2.imshow("contours", bgr_int)
        cv2.waitKey(0)

    # filter too small areas
    filtered = []
    bgr_filtered = cv2.imread(img)
    for idx, bloc in enumerate(blocs):
        x, y, w, h = bloc
        if w > WIDTH_THRESHOLD * bgr_filtered.shape[1]:
            if h > HEIGHT_THRESHOLD * bgr_filtered.shape[0]:
                filtered += [bloc]
                bgr_filtered = cv2.rectangle(
                    bgr_filtered, (x, y), (x+w, y+h), (0, 255, 0), 2)

    base = os.path.basename(img)
    dirname = os.path.dirname(os.path.dirname(img))
    print(dirname + "/output/result_"+base)
    cv2.imwrite(dirname + "/output/result_"+base,bgr_filtered)

    if VERBOSE == 1:
        # display original image with contours
        cv2.namedWindow("filtered contours", cv2.WINDOW_NORMAL)
        cv2.imshow("filtered contours", bgr_filtered)
        cv2.waitKey(0)
    

    cropped = [imgray[y:y+h,x:x+w] for [x,y,w,h] in filtered]
    
    return cropped

    
img_list = [x for x in os.listdir(IMG_PATH) if x.split('.')[-1] in ['jpg', 'png']]
for img_name in img_list:
    print(img_name)

    cropped = process_image(IMG_PATH + '/' + img_name)

    for crop in cropped:
        text = pytesseract.image_to_string(Image.fromarray(np.uint8(crop)), lang = 'fra')
        print(text)

        if VERBOSE == 1:
            # display original image with contours
            cv2.namedWindow("cropped text", cv2.WINDOW_NORMAL)
            cv2.imshow("cropped text", crop)
            cv2.waitKey(0)

        
    end = time.time()
    print("elapsed time :{} s".format(end-start))
    print("{} image per s".format(len(img_list)/float(end-start)))