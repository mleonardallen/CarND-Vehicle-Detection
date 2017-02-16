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

        ###
        # SVM Classifier
        # Note: Swapped out with RandomForestClassifier to improve performance
        ###
        # parameters = {
        #     'kernel': ['linear', 'rbf'],
        #     'gamma': ['auto', 0.1, 0.001, 0.0001],
        #     'C': [1, 10, 100, 1000]
        # }
        # svr = svm.SVC()
        # f1_scorer = make_scorer(f1_score)
        # clf = GridSearchCV(svr, verbose=100, param_grid=parameters, scoring=f1_scorer, cv=10)

        clf = RandomForestClassifier(n_jobs=-1, min_samples_split=5)
        clf = AdaBoostClassifier(base_estimator=clf, n_estimators=5)

        self.pipeline = Pipeline([
            ('scaler', StandardScaler(with_mean=True, with_std=True)),
            # ('features', SelectPercentile(percentile=30)),
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
        
        fig, axes = plt.subplots(figsize=(12, 16))
        fig.subplots_adjust(hspace=.5)
        fig.tight_layout()
        for idx, x in enumerate(X[:5]):

            subidx = 0

            pos = (cols * subidx) + idx + 1
            text = 'Car' if y[idx] == 1 else 'Not-Car'
            self.subplot(rows, cols, pos, text)
            plt.imshow(x)
            subidx += 1

            pos = (cols * subidx) + idx + 1
            hog_image = self.hog(x, visualise=True)
            self.subplot(rows, cols, pos, text=text + ' Hog')
            plt.imshow(hog_image)
            subidx += 1

            pos = (cols * subidx) + idx + 1
            hog_image = self.hog(x, pixels_per_cell=16, visualise=True)
            self.subplot(rows, cols, pos, text=text + ' Hog, YCrCb, pixels_per_cell=16')
            plt.imshow(hog_image)
            subidx += 1

            spatials = self.bin_spatial(x, visualise=True)
            for spatidx, spatial in enumerate(spatials):
                pos = (cols * subidx) + idx + 1
                self.subplot(rows, cols, pos, text=text + ' Spatial, YUV, Channel ' + str(spatidx))
                plt.imshow(spatial)
                subidx += 1

            hists = self.color_hist(x, visualise=True)
            for histidx, hist in enumerate(hists):
                pos = (cols * subidx) + idx + 1
                self.subplot(rows, cols, pos, text=text + ' Hist, RGB, Channel ' + str(histidx))
                plt.plot(hist)
                subidx += 1

        plt.show()

    def subplot(self, row, col, idx, text = ''):
        from textwrap import wrap
        subplot = plt.subplot(9, 5, idx)
        subplot.get_xaxis().set_visible(False)
        subplot.get_yaxis().set_visible(False)
        text = "\n".join(wrap(text, 20))
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

    def hog(self, 
        image, 
        visualise=False, 
        pixels_per_cell=None,
        cells_per_block=None,
        orientations=None
    ):

        if orientations is None:
            orientations = self.orientations

        if pixels_per_cell is None:
            pixels_per_cell = self.pixels_per_cell

        if cells_per_block is None:
            cells_per_block = self.cells_per_block

        image = cv2.cvtColor(image, self.hog_color_space)

        if not visualise:
            image = (image * 255).astype('uint8')
            features = self.cv2Hog.compute(image).ravel()

            return features

        # only return the visualization
        hog_image = np.zeros((64, 64))
        for channel in self.hog_channels:
            image_channel = image[:,:,channel]
            channel_features, channel_image = hog(
                image_channel,
                orientations=orientations,
                pixels_per_cell=(pixels_per_cell, pixels_per_cell),
                cells_per_block=(cells_per_block, cells_per_block),
                visualise=True, feature_vector=False, transform_sqrt=False
            )

            hog_image = np.add(hog_image, channel_image)

        return hog_image

    def get_num_blocks(self, size):
        return (size // self.pixels_per_cell) - 1

    # Define a function to compute color histogram features
    def bin_spatial(self, image, size=(32, 32), visualise=False):
        if self.spatial_color_space:
            image = cv2.cvtColor(image, self.spatial_color_space)

        features = []
        for channel in self.spatial_channels:
            image_channel = image[:,:,channel]
            channel_features = cv2.resize(image_channel, size)

            if not visualise:
                channel_features = channel_features.ravel()

            features.append(channel_features)

        if not visualise:
            return np.concatenate(features)

        return features

    # Define a function to compute color histogram features
    def color_hist(self, image, nbins=32, visualise=False):
        if self.hist_color_space:
            image = cv2.cvtColor(image, self.hist_color_space)

        features = []
        for channel in self.hist_chanels:
            channel_hist = np.histogram(image[:,:,channel], bins=nbins)
            features.append(channel_hist[0])

        if visualise:
            return features

        return np.concatenate(features)

    def save(self):
        pickle.dump(self.pipeline, open("pipeline.pkl", "wb"))

    def load(self):
        try:
            self.pipeline = pickle.load(open("pipeline.pkl", "rb"))
        except(Exception) as e:
            pass

