import cv2 as cv
import csvIO
import numpy as np
import sys
import scipy as sp
import matplotlib.pyplot as pl
import voronoiPy

eps = sys.float_info.epsilon
img = cv.imread(r"t2.jpg")
cv.namedWindow("Image")
print("%s readed  %s * %s" % (r"t2.jpg", img.shape[0], img.shape[1]))

bounding_box = np.array([0., float(img.shape[1]), 0.,
                         float(img.shape[0])])  # [x_min, x_max, y_min, y_max]
centroidList = csvIO.readStream("t2")
for i in range(len(centroidList)):
    centroidList[i] = [abs(float(centroidList[i][1])),
                       abs(float(centroidList[i][0]))]
centroidListNP = np.array(centroidList)
originalCentroidListNP = np.copy(centroidListNP)
imgHsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)  # [H,S,V]in[180,255,255]


def normalizeClr(clrListNP):
    nClrList = []
    for c in clrListNP:
        nClrList.append([c[0] / 180, c[1] / 255])
    return np.array(nClrList)


def getCentColors(listNP, imgHsv, c='hsv'):
    color = []
    intListNP = np.rint(listNP).astype(int)
    for centPix in intListNP:
        if c == 'hsv':
            color.append(imgHsv[centPix[0], centPix[1]])
        else:
            color.append(img[centPix[0], centPix[1]][::-1])
    return np.array(color)


def kMeansClrListNP(rgbListNP, n=1, k=3):
    clrArray = rgbListNP.copy()
    # print("original shape:", clrArray.shape)
    # clrArray.reshape((-1, 3))
    # print("changed shape:", clrArray.shape)
    clrArray = np.float32(clrArray)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    ret, label, center = cv.kmeans(clrArray, 41 - n, None, criteria, 20,
                                   cv.KMEANS_RANDOM_CENTERS)
    # print("label.flatten() shape:",label.flatten().shape)
    # print("center shape:", center[label.flatten()].shape)
    # print(np.unique(center[label.flatten()], axis=0))
    # raise Exception
    return center[label.flatten()]


def plotIt(vor, n=0):
    # Image Setting
    myDpi = 72
    fig = pl.figure(1, figsize=(img.shape[0] / myDpi, img.shape[1] / myDpi),
                    dpi=myDpi, frameon=False)
    ax = pl.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    cellRegions = []

    if n > 0:
        # Compute and plot centroids\
        centroids = []
        for region in vor.filtered_regions:
            vertices = vor.vertices[region + [region[0]], :]
            centroid = voronoiPy.centroid_region(vertices)
            centroids.append(list(centroid[0, :]))
            ax.plot(centroid[:, 0], centroid[:, 1], 'r.')
        newCentroids = np.array(centroids)
        vor = voronoiPy.voronoi(newCentroids,
                                np.array([0., img.shape[0], 0., img.shape[1]]))

    # Plot ridges
    for region in vor.filtered_regions:
        vertices = vor.vertices[region + [region[0]], :]
        ax.plot(vertices[:, 0], vertices[:, 1], 'k-')
    # Plot initial points
    ax.plot(vor.filtered_points[:, 0], vor.filtered_points[:, 1], 'b.',
            alpha=0.3, marker=".")

    # Plot Fill
    global rgbListNP
    if n == 0:
        rgbListNP = getCentColors(centroidListNP, img, 'rgb') / 255
    fillListNP = kMeansClrListNP(rgbListNP, n, 6)
    for r in range(len(vor.point_region)):

        flag = False
        for col in vor.filtered_points:
            if (col == vor.points[r]).all():
                flag = True
                break
            region = vor.regions[vor.point_region[r]]
        if flag and -1 not in vor.regions[vor.point_region[r]]:

            cellRegions.append(region)
            polygon = [vor.vertices[i] for i in region]
            if n == 0:
                pl.fill(*zip(*polygon), color=rgbListNP[r])
            else:
                pl.fill(*zip(*polygon), color=fillListNP[r])

    print("ploting result_%s.png..." % n)
    pl.savefig(r"t2_%s.png" % n, dpi=myDpi)
    pl.clf()

    return vor


if __name__ == '__main__':
    vor = voronoiPy.voronoi(centroidListNP,
                            np.array([0., img.shape[0], 0., img.shape[1]]))
    # plotIt(vor)
    for n in range(40):
        vor = plotIt(vor, n)
