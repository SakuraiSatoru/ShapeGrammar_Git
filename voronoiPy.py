import matplotlib.pyplot as pl
import numpy as np
import scipy as sp
import scipy.spatial

def in_box(centroidListNP, bounding_box):
    return np.logical_and(np.logical_and(bounding_box[0] <= centroidListNP[:, 0],
                                         centroidListNP[:, 0] <= bounding_box[1]),
                          np.logical_and(bounding_box[2] <= centroidListNP[:, 1],
                                         centroidListNP[:, 1] <= bounding_box[3]))


def voronoi(centroidListNP, bounding_box):
    # Select centroidListNP inside the bounding box
    i = in_box(centroidListNP, bounding_box)
    # Mirror points
    points_center = centroidListNP[i, :]
    points_left = np.copy(points_center)
    points_left[:, 0] = bounding_box[0] - (points_left[:, 0] - bounding_box[0])
    points_right = np.copy(points_center)
    points_right[:, 0] = bounding_box[1] + (bounding_box[1] - points_right[:, 0])
    points_down = np.copy(points_center)
    points_down[:, 1] = bounding_box[2] - (points_down[:, 1] - bounding_box[2])
    points_up = np.copy(points_center)
    points_up[:, 1] = bounding_box[3] + (bounding_box[3] - points_up[:, 1])
    points = np.append(points_center,
                       np.append(np.append(points_left,
                                           points_right,
                                           axis=0),
                                 np.append(points_down,
                                           points_up,
                                           axis=0),
                                 axis=0),
                       axis=0)
    # Compute Voronoi
    vor = sp.spatial.Voronoi(points)
    # Filter regions
    regions = []
    for r in range(len(vor.point_region)):
        region = vor.regions[vor.point_region[r]]
        flag = True
        # print("region =", region)
        for index in region:
            if index == -1:
                flag = False
                break
            else:
                x = vor.vertices[index, 0]
                y = vor.vertices[index, 1]
                if not (bounding_box[0] - 0.1 <= x and x <= bounding_box[1] + 0.1 and
                        bounding_box[2] - 0.1 <= y and y <= bounding_box[3] + 0.1):
                    flag = False
                    break
        if region != [] and flag:
            regions.append(region)
    vor.filtered_points = points_center
    vor.filtered_regions = regions
    return vor


def centroid_region(vertices):
    # Polygon's signed area
    A = 0
    # Centroid's x
    C_x = 0
    # Centroid's y
    C_y = 0
    for i in range(0, len(vertices) - 1):
        s = (vertices[i, 0] * vertices[i + 1, 1] - vertices[i + 1, 0] * vertices[i, 1])
        A = A + s
        C_x = C_x + (vertices[i, 0] + vertices[i + 1, 0]) * s
        C_y = C_y + (vertices[i, 1] + vertices[i + 1, 1]) * s
    A = 0.5 * A
    C_x = (1.0 / (6.0 * A)) * C_x
    C_y = (1.0 / (6.0 * A)) * C_y
    return np.array([[C_x, C_y]])
