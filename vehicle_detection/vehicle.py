class Vehicle():
    def __init__(self):
        self.detected = False       # was the vehicle detected in the last iteration?
        self.n_detections = 0       # number of times this vehicle has been detected
        self.n_nondetections = 0    # number of times this vehicle has been lost
        self.xpixels = None         # pixel x values of last detection
        self.ypixels = None         # pixel y values of last detection
        self.recent_xfitted = []    # x position of the last n fits of the bounding box
        self.bestx = None           # average x position of the last n fits
        self.recent_yfitted = []    # y position of the last n fits of the bounding box
        self.besty = None           # average y position of the last n fits
        self.recent_wfitted = []    # width of the last n fits of the bounding box
        self.bestw = None           # average width of the last n fits
        self.recent_hfitted = []    # height of the last n fits of the bounding box
        self.besth = None           # average height of the last n fits
