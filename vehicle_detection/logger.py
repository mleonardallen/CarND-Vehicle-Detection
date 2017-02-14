from configparser import ConfigParser
from os.path import basename, splitext
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

class Logger():

    """ Class to save debug output for each step in the pipeline """

    # if True, saves each step in the pipeline as an image to output_images
    logging = False

    # these properties are used for output filename
    mode = None # test_images, video
    source = None # name of the input image
    frame = 0 # frame in the video
    step = 1 # step number for image pipeline
    log_per_frames = 1

    @staticmethod
    def increment():
        Logger.frame += 1
        Logger.step = 1

    @staticmethod
    def get_source():
        return splitext(basename(Logger.source))[0]

    @staticmethod
    def check_directory(dir):
        try:
            os.stat(dir)
        except:
            os.mkdir(dir)

    @staticmethod
    def save(image, name):

        image_type = type(image).__name__

        # do not save images if logging is turned off
        if Logger.logging == False:
            return

        assert Logger.mode is not None, "mode is not set [video, test]"

        # convert binary images to color before saving
        # if image_type == 'ndarray' and len(image.shape)== 2:
        #     image = image.reshape(image.shape + (1,)) * 255
        #     image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # for video mode, save every x frames
        if Logger.mode == 'video' and Logger.frame % Logger.log_per_frames != 0:
            return

        fname = 'output_images/'
        Logger.check_directory(fname)
        fname += Logger.mode + '/'
        Logger.check_directory(fname)

        # if image/video source is given use as prefix
        if Logger.source:
            source = splitext(basename(Logger.source))[0]
            fname += source

        # if video mode, include the frame number
        if Logger.mode == 'video':
            fname += '-' + str(Logger.frame)

        # the processing step number/name -- ex: 02-
        fname += '-' + str(Logger.step).zfill(2) + '-' + name

        fname += '.jpg'

        if image_type == 'ndarray':
            mpimg.imsave(fname, image)
        elif image_type == 'Figure':
            image.savefig(fname)

        Logger.step += 1
