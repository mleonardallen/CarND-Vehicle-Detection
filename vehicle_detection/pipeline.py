from vehicle_detection.model import Model
from vehicle_detection.logger import Logger
import vehicle_detection.calibration as calibration

import matplotlib.pyplot as plt
from scipy.ndimage.measurements import label

import numpy as np
import cv2
import copy

class Pipeline():

    def __init__(self):
        self.model = Model()

    def process(self, image):
        # sliding window on image

        image = calibration.undistort(image)

        height = image.shape[0]
        width = image.shape[1]

        window_list = []
        window_list += self.get_window_list(image, window_size=48, skip_rows=5, num_rows=1.5)
        window_list += self.get_window_list(image, window_size=64, skip_rows=3.5, num_rows=1.5)
        window_list += self.get_window_list(image, window_size=96, skip_rows=2.25, num_rows=1)
        window_list += self.get_window_list(image, window_size=128, skip_rows=0, num_rows=3)

        # windows to search
        if Logger.logging:
            tmp = copy.copy(image)
            tmp = self.draw_boxes(tmp, window_list, color=(0, 0, 255))
            Logger.save(tmp, 'window-list')

        # detections
        detections = self.search_windows(image, window_list)
        if Logger.logging:
            tmp = copy.copy(image)
            tmp = self.draw_boxes(tmp, detections, color=(255, 0, 0))
            Logger.save(tmp, 'detections')

        # heatmap
        heatmap = np.zeros_like(image[:,:,0]).astype(np.float)
        heatmap = self.add_heat(heatmap, detections)

        if Logger.logging:
            fig = plt.figure(figsize=(8, 6))
            plt.imshow(heatmap, cmap='hot')
            Logger.save(fig, 'heat-map')
            plt.close()

        # threshold
        heatmap = self.apply_threshold(heatmap, 2)
        if Logger.logging:
            fig = plt.figure(figsize=(8, 6))
            plt.imshow(heatmap, cmap='hot')
            Logger.save(fig, 'heat-map-thresholded')
            plt.close()

        labels = label(heatmap)
        if Logger.logging:
            print(labels[1], 'cars found')
            fig = plt.figure(figsize=(8, 6))
            plt.imshow(labels[0], cmap='gray')
            Logger.save(fig, 'labels')
            plt.close()

        image = self.draw_labeled_bboxes(image, labels)
        Logger.save(image, 'final')

        Logger.increment()
        return image

    def get_window_list(self, image, window_size=64, num_rows=1, skip_rows=0):
        height = image.shape[0]
        width = image.shape[1]

        return self.slide_window(image,
            y_start_stop=self.get_y_start_stop(height, window_size, skip_rows=skip_rows, num_rows=num_rows),
            x_start_stop=self.get_x_start_stop(width, window_size),
            xy_window=(window_size, window_size)
        )

    def get_x_start_stop(self, width, window_size):
        remainder = width % window_size
        return [
            int(remainder / 8),
            None
        ]

    def get_y_start_stop(self, height, window_size, num_rows=1, skip_rows=0):
        return [
            height - int(window_size * (skip_rows + num_rows)),
            height - int(window_size * skip_rows),
        ]

    # Define a function that takes an image,
    # start and stop positions in both x and y, 
    # window size (x and y dimensions),  
    # and overlap fraction (for both x and y)
    def slide_window(
        self, 
        img, 
        x_start_stop=[None, None],
        y_start_stop=[None, None], 
        xy_window=(64, 64),
        xy_overlap=(0.5, 0.5)
    ):
        # If x and/or y start/stop positions not defined, set to image size
        if x_start_stop[0] == None:
            x_start_stop[0] = 0
        if x_start_stop[1] == None:
            x_start_stop[1] = img.shape[1]
        if y_start_stop[0] == None:
            y_start_stop[0] = 0
        if y_start_stop[1] == None:
            y_start_stop[1] = img.shape[0]

        # Compute the span of the region to be searched
        xspan = x_start_stop[1] - x_start_stop[0]
        yspan = y_start_stop[1] - y_start_stop[0]

        # Compute the number of pixels per step in x/y
        nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
        ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))

        # Compute the number of windows in x/y
        nx_windows = np.int(xspan/nx_pix_per_step) - 1
        ny_windows = np.int(yspan/ny_pix_per_step) - 1

        # Initialize a list to append window positions to
        window_list = []
        # Loop through finding x and y window positions
        # Note: you could vectorize this step, but in practice
        # you'll be considering windows one by one with your
        # classifier, so looping makes sense
        for ys in range(ny_windows):
            for xs in range(nx_windows):
                # Calculate window position
                startx = xs*nx_pix_per_step + x_start_stop[0]
                endx = startx + xy_window[0]
                starty = ys*ny_pix_per_step + y_start_stop[0]
                endy = starty + xy_window[1]
                # Append window position to list
                window_list.append(((startx, starty), (endx, endy)))

        # Return the list of windows
        return window_list

    # Define a function you will pass an image 
    # and the list of windows to be searched (output of slide_windows())
    def search_windows(self, img, windows):

        # Create an empty list to receive positive detection windows

        # Iterate over all windows in the list
        images = []
        for window in windows:
            # Extract the test window from original image
            img_test = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
            images.append(img_test)

        # Predict using your classifier
        import time
        start = time.time()
        preds = self.model.predict(images)
        end = time.time()
        total = end - start
        print('prediction time:', total)

        print('num images', len(images))
        print('', len(images) / total)

        # If positive (prediction == 1) then save the window
        idxs = np.where(preds == 1)

        # Return windows for positive detections
        return np.array(windows)[idxs]


    # Define a function to draw bounding boxes
    def draw_boxes(self, img, bboxes, color=(0, 0, 255), thick=2):

        # Make a copy of the image
        imcopy = np.copy(img)
        # Iterate through the bounding boxes
        for idx, bbox in enumerate(bboxes):
            # Draw a rectangle given bbox coordinates
            cv2.rectangle(imcopy, tuple(bbox[0]), tuple(bbox[1]), color, thick)
        # Return the image copy with boxes drawn
        return imcopy

    def add_heat(self, heatmap, bbox_list):
        # Iterate through list of bboxes
        for box in bbox_list:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

        # Return updated heatmap
        return heatmap

    def apply_threshold(self, heatmap, threshold):
        # Zero out pixels below the threshold
        heatmap[heatmap <= threshold] = 0
        # Return thresholded map
        return heatmap

    def draw_labeled_bboxes(self, img, labels):
        # Iterate through all detected cars
        for car_number in range(1, labels[1]+1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            # Draw the box on the image
            cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
        # Return the image
        return img