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

def main(mode=None, source=None, out=None, log=False):

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
        spatial_color_space=cv2.COLOR_RGB2YUV,
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

        model.fit(X, y)

    elif mode == 'video':
        Logger.source = source
        pipeline = Pipeline(model=model, mode='video')
        source_video = VideoFileClip(source)
        output_video = source_video.fl_image(pipeline.process)
        output_video.write_videofile(out, audio=False)

    elif mode == 'test_images':
        pipeline = Pipeline(model=model, mode='test_images')
        images = glob.glob('test_images/*.jpg')
        for idx, fname in enumerate(images):
            pipeline.reset()
            Logger.source = fname
            img = image.imread(fname)
            img = pipeline.process(img)

    elif mode == 'calibrate':
        images = glob.glob('camera_cal/calibration*.jpg')
        calibration.calibrate(images)

    elif mode == 'visualize':
        pass


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=None)

    parser.add_argument('--mode', nargs='?', default='features',
        choices=['train', 'video', 'test_images', 'calibrate'],
        help='Calibrate camera or run pipeline on test images or video')
    parser.add_argument('--source', nargs='?', default='project_video.mp4', help='Input video')
    parser.add_argument('--out', nargs='?', default='out.mp4', help='Output video')
    parser.add_argument('--log', action='store_true', help='Log output images')
    args = parser.parse_args()

    main(
        mode=args.mode, 
        source=args.source, 
        out=args.out,
        log=args.log
    )