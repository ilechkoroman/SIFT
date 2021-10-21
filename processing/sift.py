import numpy as np
from processing.utils import gaussian, DoG, get_fst_der, get_scnd_der, assign_orientation, \
    get_patch_grads, keypoint_not_in_octave, get_histogram_for_subregion, cart_to_polar_grad
# from processing.utils import cnv as convolve
from scipy.ndimage.filters import convolve
from scipy.signal import fftconvolve as fast_convolve
from easydict import EasyDict
import cv2


class SIFTDetectorDescriptor:
    def __init__(self, config: dict):
        config = EasyDict(config)
        self.octave_count = config.octave_count
        self.image_per_octave = config.image_per_octave
        self.sigma_zero = config.sigma_zero
        self.convolution_type = config.convolution_type
        self.corner_th = config.corner_th
        self.contrast_th = config.contrast_th
        self.sigma_increase = config.sigma_increase

        self.width_area = config.width_area
        self.num_subregion = config.num_subregion
        self.num_bin = config.num_bin

        self.image_pyramid = list()
        self.DoG_pyramid = list()
        self.keypoints_pyramid = list()

        self.feature = list()
        self.descriptors = list()

        self.convolve = None
        self.mode = None
        self.set_conv_func()

    def set_conv_func(self):
        if self.convolution_type == 'fft':
            self.convolve = fast_convolve
            self.mode = 'same'
        else:
            self.mode = 'reflect'
            self.convolve = convolve

    def preprocess_img(self, img: np.array, s0=1.3) -> np.array:
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if np.max(img) > 1:
            img = img / 255.
        return self.convolve(img, gaussian(s0), mode=self.mode)

    def scale_space_extr_detection(self, img: np.array):
        print('making image pyramid')
        self.set_pyramid(img)
        print('computed pyramid')

        print('computing DoG pyramid')
        self.set_DoG_pyramid()
        print('computed DoG pyramid')

        print('computing keypoints')
        self.set_key_points()
        print('computed keypoints')

    def get_sigmas(self):
        return [self.sigma_zero * self.sigma_increase ** k for k in range(self.image_per_octave + 1)]

    def generate_octave(self, input_image: np.array) -> [np.array]:
        input_image = self.preprocess_img(input_image)
        blurred_images = [input_image]
        sigmas = self.get_sigmas()

        for i in range(self.image_per_octave + 1):
            filter_ = gaussian(sigmas[i])
            blurred_images.append(self.convolve(input_image, filter_, mode=self.mode))
        return blurred_images

    def set_pyramid(self, input_image: np.array):
        working_image = input_image.copy()

        for i in range(self.octave_count):
            self.image_pyramid.append(self.generate_octave(working_image))
            working_image = working_image[::2, ::2]

    def set_DoG_pyramid(self):
        for oct in self.image_pyramid:
            # stack all DoG octave in one (shape w, h, octave_size) and
            # collect this stacked DoG octaves to pyramid
            # further need to find local minima/maxima across all dim in some area
            self.DoG_pyramid.append(np.stack(DoG(oct), axis=0))

    def DoG_octave_keypoints(self, DoG_octave: np.array) -> np.array:
        key_points = list()
        print('computing local extremums for octave')
        extremums = self.get_local_extr(DoG_octave)
        print('computed local extremums for octave')

        print('keypoint localisation')
        for extremum in extremums:
            sigma_idx, y_idx, x_idx = extremum
            offset, jac_mat, hess_mat = self.localize_keypoint(DoG_octave, sigma_idx, y_idx, x_idx)

            contrast = DoG_octave[sigma_idx, y_idx, x_idx] + 0.5 * jac_mat @ offset
            eig_value, eig_vec = np.linalg.eig(hess_mat[1:, 1:])  # we need only hess mat for x and y

            if (eig_value[0] / eig_value[1] + 1) ** 2 / (eig_value[0] / eig_value[1]) <= self.corner_th:
                if abs(contrast) >= self.contrast_th:
                    corrected_key_point = np.array([sigma_idx, y_idx, x_idx]) + offset
                    if keypoint_not_in_octave(corrected_key_point, DoG_octave):
                        continue
                    corrected_key_point = np.array([int(corrected_key_point[0] + 0.5), \
                                                    int(corrected_key_point[1]), \
                                                    int(corrected_key_point[2])])
                    key_points.append(corrected_key_point)
        return np.array(key_points)

    def localize_keypoint(self, DoG_octave: np.array, sigma_idx: int, y_idx: int, x_idx: int) -> (
            np.array, np.array, np.array):
        d_sigma, dy, dx = get_fst_der(DoG_octave, sigma_idx, y_idx, x_idx)
        dxx, dyy, dss, dxy, dxs, dys = get_scnd_der(DoG_octave, sigma_idx, y_idx, x_idx)

        jac_mat = np.array([d_sigma, dy, dx])
        hess_mat = np.array([
            [dss, dys, dxs],
            [dys, dyy, dxy],
            [dxs, dxy, dxx]])

        offset = - np.linalg.inv(hess_mat) @ jac_mat
        return offset, jac_mat, hess_mat

    def get_local_extr(self, DoG_octave: np.array, strides: int = 1) -> list:
        window_size = 3
        extremum = dict()

        depth, height, weight = DoG_octave.shape
        diff_octave = DoG_octave.copy()

        # do not include first and last idx, as if they will be detected as extr
        # we not be able to compute derivatives
        for i in range(1, height - window_size, strides):
            for j in range(1, weight - window_size, strides):
                # also do not include first and last idx in octave
                patch = diff_octave[1:-1, i:i + window_size, j:j + window_size]
                max_id, min_id = np.argmax(patch), np.argmin(patch)
                if max_id % 9 == 4 or min_id % 9 == 4:
                    if max_id % 9 == 4:
                        k = max_id // 9 + 1
                    else:
                        k = min_id // 9 + 1
                    extremum[(k, i + 1, j + 1)] = 1
                    # i and j it's coord of top left corner of our slide window
                    # we are checking does centre point is local extremum, if it is the centre coord is
                    # x + 1, y + 1
                    # k + 1 because we are searching trough 2...n - 1 octaves, where n is number of octaves
                    # armax/armin return coordinates related to the patch and after we goes back to octaves coord
        return list(extremum.keys())

    def _get_key_points(self) -> np.array:
        return np.array([self.DoG_octave_keypoints(dog_octave) for dog_octave in self.DoG_pyramid])

    def rescaled_keypoints(self):
        rescaled_keypoints_pyramid = self.keypoints_pyramid.copy()
        for i in range(len(rescaled_keypoints_pyramid)):
            rescaled_keypoints_pyramid[i] = rescaled_keypoints_pyramid[i] * (2 ** i)
        return rescaled_keypoints_pyramid

    def set_key_points(self):
        self.keypoints_pyramid = self._get_key_points()

        for i, DoG_octave in enumerate(self.DoG_pyramid):
            self.keypoints_pyramid[i] = assign_orientation(self.keypoints_pyramid[i], DoG_octave)

    def get_feature(self) -> list:
        for keypoints, DoG_octave in zip(self.keypoints_pyramid, self.DoG_pyramid):
            self.feature.append(self.get_local_descriptors(keypoints, DoG_octave))
        return self.feature

    def get_local_descriptors(self, key_points: np.array, octave: np.array) -> np.array:
        descriptors = list()
        bin_width = 360 // self.num_bin

        for key_point in key_points:
            sigma_idx, y_idx, x_idx, fitted_parabola = key_point
            sigma_idx, y_idx, x_idx = np.clip(int(sigma_idx), 0, octave.shape[2] - 1), int(y_idx), int(x_idx)

            DoG_img = octave[sigma_idx]
            bottom_y, bottom_x = max(0, y_idx - self.width_area // 2), max(0, x_idx - self.width_area // 2)
            top_y, top_x = min(DoG_img.shape[0], y_idx + self.width_area // 2 + 1), min(DoG_img.shape[1], x_idx + self.width_area // 2 + 1)

            patch = DoG_img[bottom_y:top_y, bottom_x:top_x]

            dx, dy = get_patch_grads(patch)
            magnitude, theta = cart_to_polar_grad(dx, dy)

            subregion_w = self.width_area // self.num_subregion

            feature_vector = self.get_feature_vector(DoG_img, subregion_w, magnitude, theta, fitted_parabola, bin_width)
            feature_vector = self.process_feature_vector(feature_vector)
            descriptors.append(feature_vector)
        self.descriptors = descriptors

        return np.array(descriptors)

    def get_feature_vector(self, DoG_img: np.array, subregion_w: float, magnitude: float,
                           theta: float, fitted_parabola, bin_width) -> np.array:
        feature_vector = np.array([0.0] * self.num_bin * self.num_subregion ** 2)

        for i in range(0, subregion_w):
            for j in range(0, subregion_w):
                bottom_y, bottom_x = i * subregion_w, j * subregion_w
                top_y, top_x = min(DoG_img.shape[0], (i + 1) * subregion_w), min(DoG_img.shape[1], (j + 1) * subregion_w)

                hist = get_histogram_for_subregion(magnitude[bottom_y:top_y, bottom_x:top_x].ravel(),
                                                   theta[bottom_y:top_y, bottom_x:top_x].ravel(), self.num_bin,
                                                   fitted_parabola, bin_width, subregion_w)
                feature_vector[
                i * subregion_w * self.num_bin + j * self.num_bin:
                i * subregion_w * self.num_bin + (j + 1) * self.num_bin] = hist.flatten()
        return feature_vector

    def process_feature_vector(self, feature_vector: np.array) -> np.array:
        feature_vector /= max(1e-6, np.linalg.norm(feature_vector))
        feature_vector[feature_vector > 0.2] = 0.2
        feature_vector /= max(1e-6, np.linalg.norm(feature_vector))
        return feature_vector
