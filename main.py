import argparse
import pandas as pd
import numpy as np
import matplotlib.image as image
from sklearn.utils import shuffle
from vehicle_detection.model import Model
from vehicle_detection.pipeline import Pipeline
import vehicle_detection.calibration as calibration
from moviepy.editor import VideoFileClip
import glob

def main(mode=None, source=None, out=None, log=False):

    model = Model()

    if mode == 'train':
        print("Reading images.csv -- contains image paths")
        images = pd.read_csv('images.csv', names=['filepath', 'class'])
        print("Loading Images...")
        X = images['filepath'].values
        y = images['class'].values.astype('uint8')
        X = [image.imread(x) for x in X]
        model.fit(X, y)

    elif mode == 'video':
        pipeline = Pipeline()
        source_video = VideoFileClip(source)
        output_video = source_video.fl_image(pipeline.process)
        output_video.write_videofile(out, audio=False)

    elif mode == 'test_images':
        pipeline = Pipeline()
        images = glob.glob('test_images/*.jpg')
        for idx, fname in enumerate(images):
            img = image.imread(fname)
            img = pipeline.process(img)

    elif mode == 'calibrate':
        images = glob.glob('camera_cal/calibration*.jpg')
        calibration.calibrate(images)


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