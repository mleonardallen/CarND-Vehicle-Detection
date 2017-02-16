from collections import deque
import numpy as np
import cv2
from pykalman import KalmanFilter

class Vehicle():

    def __init__(self, threshold=5, remove_threshold=10):

        self.n = 20

        self.detected = True       # was the vehicle detected
        self.measurement = True    # did we get a vehicle measurement, False if overlaps occur

        self.threshold = threshold          # number of detections required before displaying bounding box
        self.remove_threshold = remove_threshold

        self.n_detections = 0       # number of times this vehicle has been detected
        self.n_nondetections = 0    # number of times this vehicle has been lost

        self.points = deque(maxlen = self.n)       # center points of measurements
        self.heights = deque(maxlen = self.n)      # heights of bounding box
        self.widths = deque(maxlen = self.n)       # widths of bounding box

        self.white = (1, 1, 1)
        self.font = cv2.FONT_HERSHEY_SIMPLEX

        self.min_size = 150

    def matches(self, label):

        bbox = self.get_label_box(label)
        width, height = self.get_dimensions(bbox)
        if width * height < self.min_size:
            return False

        # Identify x and y values of those pixels
        point = self.get_predicted_location()

        if self.is_point_in_bbox(point, bbox):
            self.detected = True
            return True

        return False

    def update(self, label):

        point = self.get_center_point(label)
        self.point = point

        bbox = self.get_label_box(label)
        width, height = self.get_dimensions(bbox)

        self.points.append(point)
        self.widths.append(width)
        self.heights.append(height)

        self.measurement = True

    def get_center_point(self, label):
        labely = np.array(label[0])
        labelx = np.array(label[1])

        return (int(np.mean(labelx)), int(np.mean(labely)))

    def get_label_box(self, label):
        nonzeroy = np.array(label[0])
        nonzerox = np.array(label[1])
        x1, y1, x2, y2 = np.min(nonzerox), np.min(nonzeroy), np.max(nonzerox), np.max(nonzeroy)

        return ((x1, y1),(x2, y2))

    def check_detected(self):

        # if we didn't get a measurement this time around
        # keep updating the predicted location until we can get another measurement
        if not self.measurement or not self.detected:
            point = self.get_predicted_location()
            self.points.append(point)
            self.widths.append(self.widths[-1])
            self.heights.append(self.heights[-1])

        # increment detected/not detected counts
        if self.detected == True:
            self.n_detections += 1
        else:
            self.n_nondetections += 1

        # if we get too many non-detections, then fail check
        valid_vehicle = self.n_nondetections < self.remove_threshold
        return valid_vehicle

    # TODO: incorporate Kalman Filter
    def get_predicted_location(self):
        return self.points[-1]

    def is_point_in_bbox(self, point, bbox):

        x, y = point
        x1, y1 = bbox[0]
        x2, y2 = bbox[1]

        x_in_range = x1 < x and x < x2
        y_in_range = y1 < y and y < y2

        return x_in_range and y_in_range

    def get_dimensions(self, bbox):
        x1, y1 = bbox[0]
        x2, y2 = bbox[1]
        width, height = x2 - x1, y2 - y1

        return width, height

    def get_bbox(self, point):

        width = np.average(self.widths, axis=0)
        height = np.average(self.heights, axis=0)

        x, y = point

        return (
            (int(x - width / 2), int(y - height / 2)),
            (int(x + width / 2), int(y + height / 2)),
        )

    def draw_bbox(self, img):
        # do not draw until we see the car a few times
        # this is to remove false positives

        if self.n_detections < self.threshold:
            return img

        point = np.average(self.points, axis=0)
        point = (int(point[0]), int(point[1]))

        bbox = self.get_bbox(point)

        if not self.detected:
            return img

        if not self.measurement:
            return img

        color = (0,1,0) 
        if not self.measurement:
            color = (1,1,0)

        cv2.rectangle(img, (bbox[0][0], bbox[0][1]), (bbox[1][0], bbox[1][1]), color, 6)
        cv2.circle(img, self.point, 6, color)

        return img
