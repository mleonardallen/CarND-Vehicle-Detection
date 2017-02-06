from sklearn import svm
import pickle
from skimage.feature import hog
import cv2
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import  train_test_split
from sklearn.metrics import f1_score, make_scorer
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import chi2

class Model():

    def __init__(self):

        self.feature_selection = None
        self.model = None
        self.load()

    def fit(self, X, y):

        print("Extracting Features...")
        X = [self.extract_features(x) for x in X]
        print("Feature Selection...")
        self.feature_selection = SelectPercentile(chi2, percentile=20).fit(X, y)
        X = self.feature_selection.transform(X)
        X -= 0.5
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        print("Fitting Model...")
        self.model = svm.SVC(kernel='rbf', C=100, gamma=0.001, verbose=True, probability=False)
        self.model.fit(X_train, y_train)
        print("\nScoring...")
        accuracy = self.model.score(X_test, y_test)
        print("Score", accuracy)
        print("Done")

        self.save()

    def predict(self, X):
        X = [self.extract_features(x) for x in X]
        X = self.feature_selection.transform(X)
        X -= 0.5

        return self.model.predict(X)

    def grid_search(self, X, y):
        parameters = {
            'kernel': ['linear', 'rbf'],
            'gamma': ['auto', 0.1, 0.001, 0.0001],
            'C': [1, 10, 100, 1000]
        }
        svr = svm.SVC()
        f1_scorer = make_scorer(f1_score)

        model = GridSearchCV(svr, verbose=100, param_grid=parameters, scoring=f1_scorer, cv=10)
        model.fit(X, y)

        print(model.best_estimator_)

    def extract_features(self, image):

        hog = self.hog(image)
        # hist_features = self.color_hist(image)
        bin_spatial = self.bin_spatial(image, cvt_color=cv2.COLOR_RGB2HSV)

        hog = self.normalize(hog)
        # hist_features = self.normalize(hist_features)
        bin_spatial = self.normalize(bin_spatial)

        features = np.concatenate((hog, bin_spatial))

        return features

    def normalize(self, features):
        features = features.astype('float64').reshape(1,-1)
        features = normalize(features, norm='max', return_norm=True)
        features = features[0][0]

        return features

    def hog(self, image, pixels_per_cell=8, cell_per_block=3, orient=9, visualise=False, feature_vector=True):

        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        return hog(
            image,
            orientations=orient,
            pixels_per_cell=(pixels_per_cell, pixels_per_cell),
            cells_per_block=(cell_per_block, cell_per_block),
            visualise=visualise,
            feature_vector=feature_vector,
            transform_sqrt=True,
        )

    # Define a function to compute color histogram features  
    def bin_spatial(self, img, cvt_color=None, channels=[], size=(32, 32)):
        # Convert image to new color space (if specified)
        if cvt_color:
            img = cv2.cvtColor(img, cvt_color)

        spat0 = img[:,:,0]
        spat1 = img[:,:,1]
        spat2 = img[:,:,2]

        img = np.concatenate((spat1, spat2))
        features = cv2.resize(img, size).ravel()
        # Return the feature vector
        return features

    # Define a function to compute color histogram features  
    def color_hist(self, image, nbins=32, bins_range=(0, 256)):
        # Compute the histogram of the RGB channels separately
        rhist = np.histogram(image[:,:,0], bins=nbins, range=bins_range)
        ghist = np.histogram(image[:,:,1], bins=nbins, range=bins_range)
        bhist = np.histogram(image[:,:,2], bins=nbins, range=bins_range)
        # Generating bin centers
        bin_edges = rhist[1]
        bin_centers = (bin_edges[1:]  + bin_edges[0:len(bin_edges)-1])/2
        # Concatenate the histograms into a single feature vector
        hist_features = np.concatenate((rhist[0], ghist[0], bhist[0]))

        return hist_features

    def save(self):
        pickle.dump(self.model, open("model.pkl", "wb"))
        pickle.dump(self.feature_selection, open("feature_selection.pkl", "wb"))

    def load(self):
        try:
            self.model = pickle.load(open("model.pkl", "rb"))
            self.feature_selection = pickle.load(open("feature_selection.pkl", "rb"))
        except(Exception) as e:
            pass


    # Define a function to extract features from a single image window
    # This function is very similar to extract_features()
    # just for a single image rather than list of images
    def single_img_features(self, img, color_space='RGB', spatial_size=(32, 32),
                            hist_bins=32, orient=9, 
                            pix_per_cell=8, cell_per_block=2, hog_channel=0,
                            spatial_feat=True, hist_feat=True, hog_feat=True):
        #1) Define an empty list to receive features
        img_features = []
        #2) Apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(img)      
        #3) Compute spatial features if flag is set
        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            #4) Append features to list
            img_features.append(spatial_features)
        #5) Compute histogram features if flag is set
        if hist_feat == True:
            hist_features = color_hist(feature_image, nbins=hist_bins)
            #6) Append features to list
            img_features.append(hist_features)
        #7) Compute HOG features if flag is set
        if hog_feat == True:
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.extend(get_hog_features(feature_image[:,:,channel], 
                                        orient, pix_per_cell, cell_per_block, 
                                        vis=False, feature_vec=True))      
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            #8) Append features to list
            img_features.append(hog_features)

        #9) Return concatenated array of features
        return np.concatenate(img_features)
