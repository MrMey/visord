# coding: utf-8
import cv2
import numpy as np
from matplotlib import pyplot as plt
import pickle

img_path = 'house.jpg'


bgr = cv2.imread(img_path)
bgr = cv2.resize(bgr, (540, 360))

imgray = cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray,127,255,0)

im2, contours = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
print(contours)

exit()

plt.imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
plt.title("original picture")
plt.show()


yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)

# Gaussian blur
blur = cv2.GaussianBlur(yuv[:, :, 0], (5, 5), 2)

plt.imshow(blur, cmap='gray')
plt.title("blurred picture")
plt.show()

# Sobel Filter
sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=5)

# Norm computation
norm = np.power(sobelx, 2)+np.power(sobely, 2)

# Direction computation
direction = np.zeros(sobelx.shape)
direction[sobelx != 0] = np.arctan(sobely[sobelx != 0]/sobelx[sobelx != 0])
direction[sobelx == 0] = np.pi/2

# NMS filtering
nms = np.zeros(norm.shape)
for x in range(nms.shape[0]):
    for y in range(nms.shape[1]):
        if direction[x, y] > 7 * np.pi/8:
            direction[x, y] -= np.pi
        if -np.pi/8 <= direction[x, y] < np.pi/8:
            comp = [[0, 1], [0, -1]]
        if np.pi/8 <= direction[x, y] < 3 * np.pi/8:
            comp = [[-1, 1], [1, -1]]
        if 3*np.pi/8 <= direction[x, y] < 5 * np.pi/8:
            comp = [[-1, 0], [1, 0]]
        if 5 * np.pi/8 <= direction[x, y] < 7 * np.pi/8:
            comp = [[-1, -1], [1, 1]]
        if 0 <= x+comp[0][0] < nms.shape[0] and 0 <= y+comp[0][1] < nms.shape[1]:
            if 0 <= x+comp[1][0] < nms.shape[0] and 0 <= y+comp[1][1] < nms.shape[1]:
                if abs(norm[x, y]) == max(abs(norm[x, y]), abs(norm[x+comp[0][0], y+comp[0][1]]), abs(norm[x+comp[1][0], y+comp[1][1]])):
                    nms[x, y] = norm[x, y]
            else:
                if abs(norm[x, y]) == max(abs(norm[x, y]), abs(norm[x+comp[0][0], y+comp[0][1]])):
                    nms[x, y] = norm[x, y]
        else:
            if abs(norm[x, y]) == max(abs(norm[x, y]), abs(norm[x+comp[1][0], y+comp[1][1]])):
                nms[x, y] = norm[x, y]


low_threshold = 0.7
high_threshold = 0.9

weak_pixel = np.zeros(nms.shape)
strong_pixel = np.zeros(nms.shape)
strong_pixel[nms > high_threshold] = 1
weak_pixel[nms <= high_threshold] = 1
weak_pixel[low_threshold < nms] = 0

print("nb of strong pixel : {}".format(len(np.where(strong_pixel == 1.0)[0])))
print("nb of weak pixel : {}".format(len(np.where(weak_pixel == 1.0)[0])))

dist = 1

canny = strong_pixel.copy()
for x in range(weak_pixel.shape[0]):
    for y in range(weak_pixel.shape[1]):
        canny[x, y] = np.max(strong_pixel[max(0, x-dist):min(weak_pixel.shape[0],
                                                             x+dist), max(0, y-dist):min(weak_pixel.shape[1], y+dist)])

plt.imshow(canny, cmap='gray')
plt.title("canny result")
plt.show()

# Counting neighbors

neighbor_filter = np.ones([3, 3])
neighbor_filter[1, 1] = 0
neighbors = cv2.filter2D(canny, -1, neighbor_filter)





# DFS

def get_neighbors(x, y, neighbors):
    all_neighbors = [[x-1, y-1], [x-1, y], [x-1, y+1],
                     [x, y+1], [x+1, x+1], [x+1, y], [x+1, y-1], [x, y-1]]
    valid = []
    for idx in range(len(all_neighbors)):
        if 0 <= all_neighbors[idx][0] < neighbors.shape[0]:
            if 0 < all_neighbors[idx][1] < neighbors.shape[1]:
                if neighbors[all_neighbors[idx][0], all_neighbors[idx][1]] > 0:
                    valid += [all_neighbors[idx]]
    return valid

load_pickle = True
if load_pickle:
    with open('edges.pickle', 'rb') as handle:
        edges = pickle.load(handle)
else:
    neighbors_process = np.copy(neighbors)
    neighbors_process[canny == 0] = 0

    print("nb of valid neighboured pixel : {}".format(len(np.where(neighbors_process >= 1.0)[0])))

    plt.imshow(neighbors_process, cmap='gray')
    plt.title("count of neighbors")
    plt.show()

    queue = []
    edges = []
    current_queue = []
    current_edge = []

    while len(np.where(neighbors_process > 0)[0]) > 0:
        if queue == []:
            # as long as there is no pixel with 1 neighbor we decrease the neighbors by 1
            while len(np.where(neighbors_process>0)[0]) > 0:
                # if we find a pixel with 1 neighbor we break
                if len(np.where(neighbors_process == 1.0)[0]) > 0:
                    one_neighbors = np.where(neighbors_process == 1.0)
                    break
                # else we decrease neighbor by 1
                neighbors_process = neighbors_process - 1

            
            queue = [[one_neighbors[0][0], one_neighbors[1][0]]]
        
        this = queue.pop(-1)
        neighbors_process[this[0], this[1]
                        ] = neighbors_process[this[0], this[1]] - 1
        current_edge += [this]
        this_neighbors = get_neighbors(this[0], this[1], neighbors_process)

        if this_neighbors == []:
            edges += [current_edge]
            print("starting edge no : {}".format(len(edges)))
            current_edge = []

        queue += [x for x in this_neighbors if x not in queue]
    
    with open('edges.pickle', 'wb') as handle:
        pickle.dump(edges, handle)


test = np.ones(yuv[:,:,0].shape)
print(len(edges))
for edge in edges:
    if len(edge)>20:
        print(len(edge))
        for pix in edge:

            test[pix[0], pix[1]] = 0


plt.imshow(test, cmap='gray')
plt.show()

def get_patch(img, patch_size):
    start_i = 0
    start_j = 0
    end_i = img.shape[0]
    end_j = img.shape[1]

