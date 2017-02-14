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

My `HOG` feature extraction is contained in the method `hog` in the file `vehicle_detection/model.py`.  Initially I experimented with the `skimage hog`.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

####2. Explain how you settled on your final choice of HOG parameters.

For orientations, I found that increasing up to `orientations=9` gave better performance with my classifier.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

In determining which features and color spaces to leverage, I considered two factors: prediction accuracy and time to extract features.

| Feature    | Feature Extraction on 1000 64x64 images | Prediction Accuracy (non-optimized classifier) |
| ---------- | ---------------------------------------:| ----------------------------------------------:|
| HOG        | 0.12s                                   | 0.97                                           |
| Spatial    | 0.1s                                    | 0.97                                           |
| Color Hist | 0.5s                                    | 0.90                                           |

I opted to remove the color histogram features due to the added cost of feature extraction and relatively little value compared with the other two features.

Next, I looked at updating color spaces when combining HOG and Spatial features.  After training my classifier and measuring the accuracy with each combination of color spaces, I decided I got the best accuracy with the following combination.  I did end up using two separate color spaces.  My intuition is that one color space was able to pick up on features that the other missed.  For what color spaces to include, I decided to not limit the channels for two reasons.  My first reason is because the features have very little overhead in extracting. Secondly, I opted to use a decision tree family classifier, which do not suffer from the curse of dimensionality due to their implicit ability to do feature selection.  Because of this, I was not worried about the number of features generated.

| Feature | Color Space |
| ------- | ----------- |
| HOG     | YCrCB       |
| Spatial | YUV         |


Note: To easily save my normalization, feature selection and classifier, I found it useful to wrap everything with (sklearn.pipeline.Pipeline)[http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html].
```
self.pipeline = Pipeline([
    ('scaler', StandardScaler(with_mean=True, with_std=True)),
    ('features', SelectPercentile(percentile=20))
    ('clf', clf)
])

// ...

pickle.dump(self.pipeline, open("pipeline.pkl", "wb"))

```

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

##### Accuracy

I started out using a SVM using Grid Search to determine the best parameters.  I found `C = 100`, 'gamma = 0.001', and `kernel = rbf` gave good results with accuracy around 99%.

##### Prediction Time

My initial pipeline and classifier took several seconds to make predictions per frame of the video.

I first sought ways to bring my assifier prediction time down.  Since the computational complexity of SVM is linear with respect to the number of support vectors (nSV) and the dimensionality of the data, I sought ways to reduce them both.

 * With StandardScaler `std_dev = True` caused the number of support vectors to increase, slowing down prediction times.
 * Using feature selection, I found that 30% of the feature still gave good accuracy while improving prediction speed.
 * I found some images in the `not-cars` images that contained partial car iamges.  I removed those hoping to reduce the noice in the data.
 * Wrapping with a `Bagging Classifier` improved prediction time.

After making these improvements, my prediction time reduced from around `32s` to `1.5s` per frame.

##### Feature Extraction

To bring feature extraction time down, I implimented a suggestion from the project hints, running the `hog` function only once on the entire image.  However, I found this took several seconds to run on the entire image, and the effect is compounded when searching multiple scales.

This led me to discover the (OpenCV Version of HOG)[http://docs.opencv.org/2.4/modules/gpu/doc/object_detection.html], and after reading that OpenCV could be up to (30x faster)[http://bbabenko.tumblr.com/post/56701676646/when-hogs-py], I decided to give it a try.  With the `OpenCV HOG` in place, I found similar results.  Extracting features for around 1000 windows now took only `0.2s`.  The bottleneck now became the classifer.

##### Prediction Time (Round Two)

To bring my prediction time down even further, I started looking into classifiers other than SVM.  My thoughts here is that perhaps the data is just to noisy for SVM to perform well.

In addition, while training time isn't necessarily my primary concern, the (Scikit-Learn SCV Documentation)[http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html] indicates that SVMs do not scale well with large datasets.

> The implementation is based on libsvm. The fit time complexity is more than quadratic with the number of samples which makes it hard to scale to dataset with more than a couple of 10000 samples.

With this in mind, I experimented with `RandomForestClassier`.

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

