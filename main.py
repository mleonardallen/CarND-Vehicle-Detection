import argparse
import pandas as pd
import numpy as np
import matplotlib.image as image
from moviepy.editor import VideoFileClip
import glob
import os.path
import time
import cv2

from vehicle_detection.model import Model
from vehicle_detection.logger import Logger
from vehicle_detection.pipeline import Pipeline
from sklearn.utils import shuffle

def main(mode=None, source=None, out=None, log=False, mine=False):

    Logger.logging = log
    Logger.mode = mode
    model = Model(
        pixels_per_cell=8,
        cells_per_block=2,
        orientations=9,
        use_hog=True,
        use_spatial=True,
        use_hist=False,
        hog_color_space=cv2.COLOR_RGB2YCrCb,
        spatial_color_space=cv2.COLOR_RGB2HSV,
        hist_color_space=None,
    )

    if mode == 'train':
        print("Reading images.csv -- contains image paths")
        images = pd.read_csv('images.csv', names=['filepath', 'class'])
        X_all = images['filepath'].values
        y_all = images['class'].values.astype('uint8')

        print('Cars:', len(np.where(y_all == 1)[0]))
        print('Not-Cars:', len(np.where(y_all == 0)[0]))

        print("Load Images...")
        X = []
        y = []

        start = time.time()
        for idx, x in enumerate(X_all):
            if os.path.isfile(x):
                X.append(image.imread(x))
                y.append(y_all[idx])
        end = time.time()
        print('time (load images):', end-start)

        print(len(y))

        model.fit(X, y)

    elif mode == 'video':
        Logger.source = source
        pipeline = Pipeline(model=model, mode='video', mine=mine)
        source_video = VideoFileClip(source)
        output_video = source_video.fl_image(pipeline.process)
        output_video.write_videofile(out, audio=False)

    elif mode == 'test_images':
        pipeline = Pipeline(model=model, mode='test_images', mine=mine)
        images = glob.glob('test_images/*.jpg')
        for idx, fname in enumerate(images):
            pipeline.reset()
            Logger.source = fname
            img = image.imread(fname)
            img = pipeline.process(img)

    elif mode == 'visualise':
        print("Reading images.csv -- contains image paths")
        images = pd.read_csv('images.csv', names=['filepath', 'class'])
        X_all = images['filepath'].values
        y_all = images['class'].values.astype('uint8')
        print('Cars:', len(np.where(y_all == 1)[0]))
        print('Not-Cars:', len(np.where(y_all == 0)[0]))

        print("Load Images...")
        X = []
        y = []

        X_all, y_all = shuffle(X_all, y_all)
        X_all, y_all = X_all[:5], y_all[:5]

        start = time.time()
        for idx, x in enumerate(X_all):
            if os.path.isfile(x):
                X.append(image.imread(x))
                y.append(y_all[idx])
        end = time.time()
        print('time (load images):', end-start)

        model.visualise(X, y)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=None)

    parser.add_argument('--mode', nargs='?', default='features',
        choices=['train', 'video', 'test_images', 'visualise'],
        help='Calibrate camera or run pipeline on test images or video')
    parser.add_argument('--source', nargs='?', default='project_video.mp4', help='Input video')
    parser.add_argument('--out', nargs='?', default='out.mp4', help='Output video')
    parser.add_argument('--log', action='store_true', help='Log output images')
    parser.add_argument('--mine', action='store_true', help='Hard negative mining')
    args = parser.parse_args()

    main(
        mode=args.mode, 
        source=args.source, 
        out=args.out,
        log=args.log,
        mine=args.mine
    )