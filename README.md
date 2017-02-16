# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

---

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I start by reading in all the `vehicle` and `non-vehicle` images.

> `main.py`, starting at `line 31`

Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

TODO

For all `Vehicle` and `Not Vehicle` images in the dataset, I extract features using the `single_img_features` method.

> `vehicle_detection/model.py`, in method `fit` on `line 96`

In the `single_img_features` method there are flags to control which features are added to my feature array.  For convenience, the feature flags are controlled from `main.py`, so that the same flags are used during training and prediction.  Additional parameters allowed me to test different color spaces and channel combinations.

With the feature flag active, `single_img_features` leverages a corresponding feature extraction method for each feature type: `hog`, `bin_spatial`, and `color_hist`.  Each method returns a 1d array that gets concatenated into one feature vector containing each leveraged feature.

To extract `HOG` features, initially I experimented with `skimage.hog()`, which allows visualization by passing in the `visualise` parameter.  Using this parameter, I visually inspected the HOG feature output for different color spaces on a few images from the training data.  Inspected color spaces included `RGB`, `LUV`, `HSV`, `YUV`, `HLS`, and `YCrCb`.  Visually I noted that HOG gradients appeared to more consistently visible under the `YCrCb` color space.

Here are a few examples of the `hog`, `color spatial`, and `color histogram` features from both the `vehicle` and `non-vehicle` classes:

![alt text][image1]

Note: The `skimage.hog()` provided a great way to visualise the features, but in the end I switched to `cv2.HOGDescriptor` due to performance.  See my section on [optimization](#optimization)

####2. Explain how you settled on your final choice of HOG parameters.

In determining which features and color spaces to leverage, I considered two factors: prediction accuracy and time to extract features.  Below are my results after training each feature individually with a non-optimized classifier, and looking at feature selection performance within my pipeline.

| Parameter    | Chosen Value | Reasoning |
| ------------ | -----:|:--------- |
| orientations | 9     | Lowering the orientation bins down to 6 improved the speed for hog feature extraction, but I felt that the cost to accuracy was too much.  Improvements to testing accuracy plataued at 9 bins.  Increasing beyond 9 increased time needed for feature extraction. |
| pixels per cell | 8 | TODO |
| cells per block | 2 | TODO |

I opted to remove the color histogram features due to the added cost of feature extraction and relatively little value compared with the other two features.

| Feature    | Feature Extraction on 1000 64x64 images | Test Accuracy |
| ---------- | -----:| ----:|
| HOG        | 0.12s | 0.97 |
| Spatial    | 0.1s  | 0.97 |
| Color Hist | 0.5s  | 0.95 |

Next, I looked at updating color spaces when combining HOG and Spatial features.  After training my classifier and measuring the accuracy with each combination of color spaces, I got the best results by using `YCrCB` for HOG features and a separate color space for Color Spatial features.  My intuition is that one color space was able to pick up on features that the other missed.

| Feature | Color Space | Channels |
| ------- | ----------- | -------- |
| HOG     | YCrCB       | ALL      |
| Spatial | YUV         | ALL      |

For what color spaces to include, I decided to not limit the channels for two reasons.  My first reason is because the features have very little overhead in extracting after optimization. Secondly, I opted to use a decision tree family classifier, which do not suffer from the curse of dimensionality due to their implicit ability to do feature selection.  Because of this, I was not worried about the number of features generated.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

After extracting features for all of the `Vehicle` and `Non-Vehicle` as described above, I then prepare to train my classifier.

> `vehicle_detection/model.py`, method `fit`

Before training the classifier, I split the data into training and testing datasets using `sklearn.model_selection.train_test_split`.  The test dataset is witheld so we can verify the trained model can generalize against unseen data.

> `vehicle_detection/model.py`, method `fit` on `line 101`

After separating my data into train and test datasets, I then put together my training pipeline which consists of a classifier, feature scaling, and feature selection.

Note: To easily run the entire pipeline, I found it useful to wrap everything with [sklearn.pipeline.Pipeline](http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html).

```
self.pipeline = Pipeline([
    ('scaler', StandardScaler(with_mean=True, with_std=True)),
    ('features', SelectPercentile(percentile=30))
    ('clf', clf)
])
```

Starting with my classifier, I initially experimented with the SVM classifier.  To tune my model, I leveraged `GridSearchCV` with 10-fold cross validation.  Parameters searched are included below.

| Parameter | Values | Best |
| --------- | ------ | ---- |
| kernel    | linear, rbf | rbf |
| gamma     | auto, 0.1, 0.001, 0.0001 | 0.001 |
| C         | 1, 10, 100, 1000 | 100 |

> `vehicle_detection/model.py`, method `fit` on `line 103`

Note: The above code is commented out because I opted to use a `RandomForestClassifer` in the end.  See the section on [optimiazation](#optimization).

My classifier is then trained on the training dataset by calling the `fit` method of the `pipeline`.  The pipeline begins the fitting process by using the StandardScaler to normalize the data with zero mean and unit variance.  After normalization, `sklearn.feature_slection.SelectPercentile` removes all but 30% of the highest scoring features (Scored using ANOVA f values).

> `vehicle_detection/model.py`, method `fit` on `line 125`

After training, I then evaluate the performance on the test dataset.

> `vehicle_detection/model.py`, method `fit` on `line 136`

The trained pipeline is then saved.  Storing the entire pipeline this way allows be to use the same feature scaler, feature selection, and model on my vehicle detection pipeline later.

```
pickle.dump(self.pipeline, open("pipeline.pkl", "wb"))
```

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

My sliding window search is contained in `vehicle_detection/pipeline.py` in the `process` method.

##### Original Image

![Original Image](https://github.com/mleonardallen/CarND-Vehicle-Detection/blob/master/output_images/test_images/test1-01-original.jpg)

##### Region of Interest

To begin, I first slice off the top of the image to reduce processing time for some of the later operations.

> `vehicle_detection/pipeline.py` in `process` method on `line 60`.

![Sliced Image](https://github.com/mleonardallen/CarND-Vehicle-Detection/blob/master/output_images/test_images/test1-02-image-to-search.jpg)

##### Window Search

I then generate impliment a sliding window search to extract individual images to send to the classifier.  

> `vehicle_detection/pipeline.py` in `process` method on `line 66`.

Initially, I used pixel values and overlap values to define the window sliding behavior.  After reviewing the project hints, I switched to a sliding window scheme that steps by increments of HOG cells.  This was for batching up the HOG operation so that the hog features were not extracted redundantly on overlapping window slices.

> Note: After experimenting, I did end up switching out the batch HOG operation in favor of the very fast `cv2.HOGDescriptor`.  See the [optimization](#optimization) section.

For window sizes, I used the `scale` property to scale the entire region of interest down before doing the sliding window search.  For example, at scale `2` the image size is halved, meaning a 64x64 window now covers double the space within the image, essentially get the same effect as if the region of interest is kept constant but a 128x128 window size is used.  The added benefit is that only one resize operation is done per window scale instead of resizing each window slice.

> `vehicle_detection/pipeline.py`, methods: `get_window_list`, `get_features`

I chose 3 window scales.  A scale of `1 (64x64)`, `1.5 (96x96)`, and `2 (128x128)`.  For the smaller sizes I do not extend down to the bottom of the image as to not affect performance too much.  Vehicles at the top of the region of interest will also be smaller.  For scale of 2, I search all but the very top region of interest.  This is because vehicles can still appear large until very close to the horizon.  For cells per step, I opted to search at `1 cell per step` at the smaller scales.  Even though this increased the number of search windows and feature selection time, I was able to produce smoother bounding boxes and it also allowed me to increaase the heatmap threshold later on to reduce false positives.

> Total number of windows: 1391

![Window List](https://github.com/mleonardallen/CarND-Vehicle-Detection/blob/master/output_images/test_images/test1-03-window-list.jpg)

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

##### Hot Windows & Class Prediction

Class label predictions are made on extracted features using the same pipeline created during the training phase.  Hot windows shown below are those that are predicted to be a vehicle by the classifier.  These could also be false positives.

> `vehicle_detection/pipeline.py` in `process` method on `line 85`.

![Hot Windows](https://github.com/mleonardallen/CarND-Vehicle-Detection/blob/master/output_images/test_images/test1-04-hot-windows.jpg)

##### Heat Map

From the hot windows, a heat map is generated to indicate the spots where many detections occurred.  Hot spots occur where hot windows overlap.

> `vehicle_detection/pipeline.py` in `process` method on `line 104` and method `add_heat` on `line 289`.

![Heat Map](https://github.com/mleonardallen/CarND-Vehicle-Detection/blob/master/output_images/test_images/test1-05-heat-map.jpg)

##### Heat Map - Thresholded

The heat map is then thresholded to remove superfluous detections.  False positives occur but they generally do not exceed the heat map threshold.  In practice, I increased the threshold until most of the fasle positives went away.

> `vehicle_detection/pipeline.py` in `process` method on `line 123`

Note: I did end up averaging the heat map here.  Instead I chose to allow a Vehicle instance to keep track of internal measurements and do averages from those measurements.

![Heat Map Thresholded](https://github.com/mleonardallen/CarND-Vehicle-Detection/blob/master/output_images/test_images/test1-06-heat-map-thresholded.jpg)

##### Labels

`scipy.ndimage.measurements.label` is used to isolate heat maps into single vehicle detections.

> `vehicle_detection/pipeline.py` in `process` method on `line 131`

![Vehicle Labels](https://github.com/mleonardallen/CarND-Vehicle-Detection/blob/master/output_images/test_images/test1-07-labels.jpg)

##### Bounding Box

For test images, the bounding box is drawn based on the label.  For the video pipeline there is additional thresholding before the bounding box is shown.

![Final Image](https://github.com/mleonardallen/CarND-Vehicle-Detection/blob/master/output_images/test_images/test1-08-final.jpg)

<a name="optimization"/>

##### Optimizing Classifier Performance

My initial tuned pipeline performed well with accuracy, with around 99% accuracy, however it was very slow.  Here I focused on increasing the speed of my classifier.

###### Prediction Time

My initial pipeline and classifier took several seconds to make predictions per frame of the video.

I first sought ways to bring my assifier prediction time down.  Since the computational complexity of SVM is linear with respect to the number of support vectors (nSV) and the dimensionality of the data, I sought ways to reduce them both.

 * With StandardScaler `std_dev = True` caused the number of support vectors to increase, slowing down prediction times.
 * Using feature selection, I found that 30% of the feature still gave good accuracy while improving prediction speed.
 * I found some images in the `non-vehicles` images that contained partial vehicle iamges.  I removed those hoping to reduce the noise in the data.
 * Wrapping with a `Bagging Classifier` improved prediction time.

After making these improvements, my prediction time reduced from around `32s` to `1.5s` per frame.

###### Feature Extraction

To bring feature extraction time down, I implimented a suggestion from the project hints, running the `hog` function only once on the entire image.  However, I found this took several seconds to run on the entire image, and the effect is compounded when searching multiple scales.

This led me to discover the [OpenCV Version of HOG](http://docs.opencv.org/2.4/modules/gpu/doc/object_detection.html), and after reading that OpenCV could be up to [30x faster](http://bbabenko.tumblr.com/post/56701676646/when-hogs-py), I decided to give it a try.  With the `OpenCV HOG` in place, I found similar results.  Extracting features for around 1000 windows now took only `0.2s`.  The bottleneck now became the classifer.

###### Prediction Time (Round Two)

To bring my prediction time down even further, I started looking into classifiers other than SVM.  My thoughts here is that perhaps the data is just to noisy for SVM to perform well.

In addition, while training time isn't necessarily my primary concern, the [Scikit-Learn SVM Documentation](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) indicates that SVMs do not scale well with large datasets.

> The implementation is based on libsvm. The fit time complexity is more than quadratic with the number of samples which makes it hard to scale to dataset with more than a couple of 10000 samples.

With this in mind, I experimented with `RandomForestClassfier`.

###### Random Forest Classifier
After switching to `RandomForestClassifier`, my prediction time when way down.  It took about `0.16` seconds to run predictions on about 1000 windows.  However my accuracy did go down slightly to 98%.  Even at 98%, I was getting quite a bit of false positives.  Using Adaboost in conjunction with a Random Forest Classifier, I was able to improve the accurancy even beyond what I was getting with my SVM classifier.  My final classifier did not run as fast as I wished, but still notably faster than my original SVM.

---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

##### False Positives

Instead, of averaging the heatmaps to remove false positives and create smooth bounding boxes, I chose to track each detection with an instance of the `Vehicle` class.  This is in addition to thresholding the heatmap as described above.

> `vehicle_detection/vehicle.py`

First I determine if the `label` from above matches a current vehicle.  To determine if a car matches a given label, I get the center point of the last known position of the car and determine if it is contained within the bounding box defined by the label.

> `vehicle_detection/pipeline.py` in `process` method starting on `line 139` and `vehicle_detection/vehicle.py` in the `matches` method.

##### Overlapping Match

In the case that two vehicles overlap, I remove both of those vehicles from the tracking list.  Then a new `vehicle` instance is created to track the overlapping vehicles.  In this case the newly created vehicle is assumed to have already met the thresholding requirements.

> `vehicle_detection/pipeline.py` in `process` method starting on `line 152`

![Overlap](https://github.com/mleonardallen/CarND-Vehicle-Detection/blob/master/examples/overlap.png)

##### Single Match

If matching, the `vehicle` instance is updated with the new measurement.  The new measurement is stored and later leveraged to create a smooth bounding box by averaging over a few frames.

> `vehicle_detection/pipeline.py` in `process` method starting on `line 160` and `vehicle_detection/vehicle.py` in the `update` method.

##### New Vehicle
If the label did not match any existing vehicles, a new vehicle instance in created to keep track of the detection.

> `vehicle_detection/pipeline.py` in `process` method starting on `line 160` and `vehicle_detection/vehicle.py` in the `update` method.

##### Removing Vehicles

If a vehicle is undetected beyond a threshold number of frames, it is removed from the tracked vehicles.

> `vehicle_detection/pipeline.py` in `process` method starting on `line 170` and `vehicle_detection/vehicle.py` in the `check_detected` method.

##### Bounding Box

For all vehicles that meet the thresholding criteria, a bounding box is drawn.  In addition the width/height and bounding box are averaged over a few frames to provide a smoother bounding box.  

> `vehicle_detection/pipeline.py` in `process` method starting on `line 174` and `vehicle_detection/vehicle.py` in the `update` method.

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

