import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn import svm
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectPercentile
from sklearn.model_selection import  train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import Pipeline

from vehicle_detection.logger import Logger

class Model():

    def __init__(self, 
        pixels_per_cell=8,
        cells_per_block=2,
        orientations=9,
        hog_channels=[0,1,2],
        spatial_channels=[0,1,2],
        hist_chanels=[0,1,2],
        hog_color_space=cv2.COLOR_RGB2YCrCb,
        spatial_color_space=cv2.COLOR_RGB2YCrCb,
        hist_color_space=cv2.COLOR_RGB2YCrCb,
        use_hog=True,
        use_hist=True,
        use_spatial=True,
        C=1,
        gamma=0.001,
        kernel='rbf'
    ):

        self.feature_selection = None
        self.pipeline = None

        self.hog_color_space = hog_color_space
        self.spatial_color_space = spatial_color_space
        self.hist_color_space = hist_color_space

        # HOG Properties
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.orientations = orientations
        self.block_size = pixels_per_cell * cells_per_block

        # Classifier Parameters
        self.C = C
        self.gamma = gamma
        self.kernel = kernel
        self.n_estimators = 10

        # Features
        self.use_hog = use_hog
        self.use_hist = use_hist
        self.use_spatial = use_spatial

        self.hog_channels = hog_channels
        self.spatial_channels = spatial_channels
        self.hist_chanels = hist_chanels

        winSize = (64,64)
        blockStride = (8,8)
        derivAperture = 1

        winSigma = 4.
        histogramNormType = 0
        L2HysThreshold = 2.0000000000000001e-01
        gammaCorrection = 0
        nlevels = 64

        self.cv2Hog = cv2.HOGDescriptor(
            winSize,
            (self.block_size, self.block_size),
            blockStride,
            (self.pixels_per_cell, self.pixels_per_cell),
            orientations,
            derivAperture, winSigma, histogramNormType, L2HysThreshold, gammaCorrection, nlevels
        )

        self.load()

    def fit(self, X, y):

        # self.visualise(X, y)

        print("Extract Features...")
        start = time.time()
        X = [self.single_img_features(x) for x in X]
        end = time.time()
        print('Num Features:', len(X[0]))
        print('time (extract features):', end - start)

        X_train, X_test, y_train, y_test = train_test_split(X, y)

        clf = RandomForestClassifier(n_jobs=-1, min_samples_split=5)
        clf = AdaBoostClassifier(base_estimator=clf, n_estimators=5)

        self.pipeline = Pipeline([
            ('scaler', StandardScaler(with_mean=True, with_std=True)),
            ('clf', clf)
        ])
        start = time.time()
        self.pipeline.fit(X_train, y_train)
        end = time.time()

        print('\n')
        print('time (fit):', end - start)

        start = time.time()
        y_pred = self.pipeline.predict(X_test)
        end = time.time()
        print('time (predict):', end - start)

        print("\nClassification Report")
        score = accuracy_score(y_test, y_pred)
        print('Accuracy:', score)
        target_names = ['Not Car', 'Car']
        print(classification_report(y_test, y_pred, target_names=target_names))

        self.save()

    def visualise(self, X, y):

        rows = 4
        cols = 5
        plt.subplots(figsize=(12, 16))
        for idx, x in enumerate(X[:5]):

            text = 'Car' if y[idx] == 1 else 'Not-Car'

            self.subplot(x, rows, cols, idx + 1, text)

            hog, hog_image = self.hog(x, visualise=True)
            self.subplot(hog_image, rows, cols, cols + idx + 1, text=text + ' Hog')
            plt.imshow(image)

            spatial = self.spatial(x)
            self.subplot(hog_image, rows, cols, cols * 2 + idx + 1, text=text + ' Hog')
            plt.imshow(image)


        plt.show()

    def subplot(self, image, row, col, idx, text = ''):
        subplot = plt.subplot(9, 5, idx)
        subplot.get_xaxis().set_visible(False)
        subplot.get_yaxis().set_visible(False)
        subplot.title.set_text(text)

    def single_img_features(self, image):

        features = []

        # hog features
        if self.use_hog == True:
            hog_features = self.hog(image)
            features.append(hog_features)

        # spatial features
        if self.use_spatial == True:
            bin_spatial = self.bin_spatial(image)
            features.append(bin_spatial)

        # color histogram features
        if self.use_hist == True:
            hist_features = self.color_hist(image)
            features.append(hist_features)

        return np.concatenate(features)

    def hog(self, image, visualise=False, feature_vector=True):

        image = cv2.cvtColor(image, self.hog_color_space)
        image = (image * 255).astype('uint8')
        features = self.cv2Hog.compute(image).ravel()

        if feature_vector:
            return features

        yblocks = self.get_num_blocks(image.shape[0])
        xblocks = self.get_num_blocks(image.shape[1])
        return np.reshape(
            features,
            (yblocks, xblocks, self.cells_per_block, self.cells_per_block, self.orientations)
        )



        #     if visualise:

        #         channel_features, channel_image = hog(
        #             image_channel,
        #             orientations=self.orientations,
        #             pixels_per_cell=(self.pixels_per_cell, self.pixels_per_cell),
        #             cells_per_block=(self.cells_per_block, self.cells_per_block),
        #             visualise=True, feature_vector=False, transform_sqrt=False
        #         )

        #         # reduce features by adding all hogs together
        #         hog_image = np.add(hog_image, channel_image)
        #         features = np.add(features, channel_features)
        #     else:

        #         start = time.time()
        #         channel_features = hog(
        #             image_channel,
        #             orientations=self.orientations,
        #             pixels_per_cell=(self.pixels_per_cell, self.pixels_per_cell),
        #             cells_per_block=(self.cells_per_block, self.cells_per_block),
        #             visualise=False, feature_vector=False, transform_sqrt=False
        #         )
        #         end = time.time()

        #         # reduce features by adding all hogs together
        #         features = np.add(features, channel_features)

        # if feature_vector:
        #     features = features.ravel()

        # if visualise:
        #     return features, hog_image

        # return features

    def get_num_blocks(self, size):
        return (size // self.pixels_per_cell) - 1

    # Define a function to compute color histogram features
    def bin_spatial(self, image, size=(32, 32)):
        if self.spatial_color_space:
            image = cv2.cvtColor(image, self.spatial_color_space)

        features = []
        for channel in self.spatial_channels:
            image_channel = image[:,:,channel]
            features.append(cv2.resize(image_channel, size).ravel())

        return np.concatenate(features)

    # Define a function to compute color histogram features
    def color_hist(self, image, nbins=32):
        if self.hist_color_space:
            image = cv2.cvtColor(image, self.hist_color_space)

        features = []
        for channel in self.hist_chanels:
            channel_hist = np.histogram(image[:,:,channel], bins=nbins)
            features.append(channel_hist[0])

        return np.concatenate(features)

    def save(self):
        pickle.dump(self.pipeline, open("pipeline.pkl", "wb"))

    def load(self):
        try:
            self.pipeline = pickle.load(open("pipeline.pkl", "rb"))
        except(Exception) as e:
            pass

