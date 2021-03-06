import math
from scipy.spatial.distance import euclidean
import numpy as np

def polar_to_cartesian(r, theta):
    x = r*math.cos(theta)
    y = r*math.sin(theta)
    return np.array([x,y])

def radius_from_area(a):
    return (a/(2*math.pi))**(1/2)

def ccw_angle(point_a, point_b, just_lower = False):
    flip = False
    if point_a[1] > point_b[1]:
        start_dot_center = point_a
        end_dot_center = point_b
    else:
        end_dot_center = point_a
        start_dot_center = point_b
        flip = True
    angle = math.degrees(math.acos((start_dot_center[1] - end_dot_center[1]) / euclidean(start_dot_center, end_dot_center)))
    if start_dot_center[0] < end_dot_center[0]:
        angle = 90 - angle
    else:
        angle = 90 + angle
    if just_lower:
        return angle
    if flip:
        return [180 + angle, angle]
    else:
        return [angle, 180 + angle]
