from vehicle_detection.model import Model
from vehicle_detection.logger import Logger
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import label
from collections import deque
import numpy as np
import cv2
import copy
import time

class Pipeline():

    def __init__(self, model):

        self.n = 5
        self.model = model

        # todo
        self.out_maps = []
        self.out_boxes = []

        self.heatmaps = deque(maxlen = self.n)

        self.vehicles = []

        self.cutoff = 400
        self.search = [
            {
                'scale': 1,
                'cells_per_step': 2,
                'y_start_stop': [None, 4]
            },
            {
                'scale': 1.5,
                'cells_per_step': 2,
                'y_start_stop': [0, 6]
            },
            {
                'scale': 2,
                'cells_per_step': 2,
                'y_start_stop': [None, None]
            }
        ]

    def process(self, image):

        image = image.astype(np.float32)/255

        if Logger.logging:
            print('--- process image ---')

        start_total = time.time()

        height = image.shape[0]
        width = image.shape[1]
        scale = 2

        image_to_search = image[self.cutoff:height,:,:]

        if Logger.logging:
            Logger.save(image, 'original')
            Logger.save(image_to_search, 'image-to-search')

        start = time.time()
        window_list, features = [], []
        for search in self.search:
            find_window_list, find_features = self.get_scaled_features(
                image_to_search,
                scale=search.get('scale'),
                cells_per_step=search.get('cells_per_step'),
                y_start_stop=search.get('y_start_stop')
            )
            window_list += find_window_list
            features += find_features
        end = time.time()

        if Logger.logging:
            print('time (features):', end-start)

        # Predict using your classifier
        start = time.time()
        preds = self.model.pipeline.predict(features)
        end = time.time()

        if Logger.logging:
            print('time (predict):', end-start)

        idxs = np.where(preds == 1)
        hot_windows = np.array(window_list)[idxs]

        # windows to search
        if Logger.logging:
            print('windows:', len(window_list))
            tmp = self.draw_boxes(image, window_list, color=(0, 0, 1))
            Logger.save(tmp, 'window-list')
            tmp = self.draw_boxes(image, hot_windows, color=(1, 0, 0))
            Logger.save(tmp, 'hot-windows')

        # heatmap
        heatmap = np.zeros_like(image[:,:,0]).astype(np.float)
        heatmap = self.add_heat(heatmap, hot_windows)
        self.heatmaps.append(heatmap)

        if Logger.logging:
            fig = plt.figure(figsize=(8, 6))
            plt.imshow(heatmap, cmap='hot')
            Logger.save(fig, 'heat-map')
            plt.close()

        # average heatmap
        avg_heatmap = np.sum(self.heatmaps, axis=0) / len(self.heatmaps)
        if Logger.logging:
            fig = plt.figure(figsize=(8, 6))
            plt.imshow(heatmap, cmap='hot')
            Logger.save(fig, 'average-heat-map')
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

        end_total = time.time()
        if Logger.logging:
            print('time (total)', end_total - start_total)

        Logger.increment()
        return image * 255

    def get_scaled_features(self, image, scale=1, cells_per_step=4, y_start_stop=[None, None]):

        find_image = copy.copy(image)
        find_image = self.scale_image(find_image, scale=scale)

        window_list = self.get_window_list(find_image, xy_window=(64, 64), cells_per_step=cells_per_step, y_start_stop=y_start_stop)

        features = []
        for window in window_list:
            # Extract the test window from original image
            window_pixels = self.window_to_pixels(window)
            window_image = image[window_pixels[0][1]:window_pixels[1][1], window_pixels[0][0]:window_pixels[1][0]]
            features.append(self.model.single_img_features(window_image))

        window_list = [self.window_to_draw(x, scale=scale) for x in window_list]

        return window_list, features


    def get_window_list(self, 
        image,
        x_start_stop=[None, None],
        y_start_stop=[None, None],
        xy_window=(64, 64),
        cells_per_step=2
    ):

        # Compute the number of steps in x/y
        nxsteps, nysteps, xblocks_per_window, yblocks_per_window = self.get_window_steps(image.shape, xy_window, cells_per_step=cells_per_step)

        # If x and/or y start/stop positions not defined, set to image size
        if x_start_stop[0] == None:
            x_start_stop[0] = 0
        if x_start_stop[1] == None:
            x_start_stop[1] = image.shape[1]

        if y_start_stop[0] == None:
            y_start_stop[0] = 0
        if y_start_stop[1] == None:
            y_start_stop[1] = nysteps

        # Initialize a list to append window positions to
        window_list = []
        for xs in range(nxsteps):
            for ys in range(y_start_stop[0], y_start_stop[1]):

                # Calculate window position
                xpos = xs*cells_per_step
                ypos = ys*cells_per_step

                # Append window position to list
                window_list.append(((xpos, ypos), (xpos+xblocks_per_window, ypos+yblocks_per_window)))

        # Return the list of windows
        return window_list

    def window_to_draw(self, window, scale=1):
        window = self.window_to_pixels(window)
        return (
            (int(window[0][0] * scale), int((window[0][1] * scale) + self.cutoff)),
            (int(window[1][0] * scale), int((window[1][1] * scale) + self.cutoff))
        )

    def window_to_pixels(self, window):
        ppc = self.model.pixels_per_cell
        return (
            (window[0][0] * ppc, window[0][1] * ppc),
            ((window[1][0] + 1) * ppc, (window[1][1] + 1) * ppc)
        )

    def get_window_steps(self, image_size, xy_window, cells_per_step=2):

        xblocks = image_size[1] // self.model.pixels_per_cell - 1
        yblocks = image_size[0] // self.model.pixels_per_cell - 1

        features_per_block = self.model.orientations * self.model.pixels_per_cell**2

        xblocks_per_window = (xy_window[1] // self.model.pixels_per_cell) - 1
        yblocks_per_window = (xy_window[0] // self.model.pixels_per_cell) - 1

        nxsteps = (xblocks - xblocks_per_window) // cells_per_step + 1
        nysteps = (yblocks - yblocks_per_window) // cells_per_step + 1

        return nxsteps, nysteps, xblocks_per_window, yblocks_per_window

    def scale_image(self, image, scale=1):
        if scale != 1:
            size = image.shape
            return cv2.resize(image, (np.int(size[1] / scale), np.int(size[0] // scale)))

        return image

    # Define a function to draw bounding boxes
    def draw_boxes(self, img, bboxes, color=(0, 0, 255), thick=2):

        # Make a copy of the image
        imcopy = np.copy(img)
        # Iterate through the bounding boxes
        for idx, bbox in enumerate(bboxes):

            # Draw a rectangle given bbox coordinates
            cv2.rectangle(
                imcopy, 
                tuple(bbox[0]),
                tuple(bbox[1]),
                color,
                thick
            )
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
            cv2.rectangle(img, (bbox[0][0], bbox[0][1]), (bbox[1][0], bbox[1][1]), (0,0,1), 6)
        # Return the image
        return img