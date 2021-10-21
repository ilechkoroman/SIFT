import cv2
import numpy as np
from easydict import EasyDict
from processing.utils import visualisation
from processing.sift import SIFTDetectorDescriptor

img = cv2.imread('test_images/test.png')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255.
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)/255.

config = EasyDict()
config.octave_count = 4
config.image_per_octave = 4
config.sigma_zero = 1.6
config.convolution_type = 'fft'
config.corner_th = 11**2 / 10
config.contrast_th = 0.03
config.sigma_increase = 2
config.width_area = 16
config.num_subregion = 4
config.num_bin = 8

detecor = SIFTDetectorDescriptor(config)
detecor.scale_space_extr_detection(gray_img)
keypoints_pyramid = detecor.rescaled_keypoints()
sigmas = detecor.get_sigmas()
features = detecor.get_feature()

visualisation(img_rgb, keypoints_pyramid, sigmas)
