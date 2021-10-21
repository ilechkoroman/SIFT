import numpy as np
import matplotlib.pyplot as plt


def gaussian(sigma: float) -> np.array:
    filter_size = int(2 * 3 * sigma) + 1
    filter_size = filter_size if filter_size % 2 else filter_size + 1
    grid_x = np.linspace([-filter_size // 2 + 1 for _ in range(filter_size)],
                         [filter_size // 2 for _ in range(filter_size)], filter_size)
    grid_y = np.linspace(list(range(-filter_size // 2 + 1, filter_size // 2 + 1, 1)),
                         list(range(-filter_size // 2 + 1, filter_size // 2 + 1, 1)), filter_size)
    gauss = np.exp(-((grid_x ** 2 + grid_y ** 2) / (2.0 * sigma ** 2))) / (2 * np.pi * sigma ** 2)
    return gauss / gauss.sum()  # normalize


def DoG(octave: [np.array]) -> list:
    return [octave[i + 1] - octave[i] for i in range(1, len(octave) - 1)]


def get_scnd_der(DoG_octave: np.array, sigma_idx: int, y_idx: int, x_idx: int) -> (
        float, float, float, float, float, float):
    dxx = DoG_octave[sigma_idx, y_idx, x_idx + 1] - 2 * DoG_octave[sigma_idx, y_idx, x_idx] + DoG_octave[
        sigma_idx, y_idx, x_idx - 1]
    dyy = DoG_octave[sigma_idx, y_idx + 1, x_idx] - 2 * DoG_octave[sigma_idx, y_idx, x_idx] + DoG_octave[
        sigma_idx, y_idx - 1, x_idx]
    dss = DoG_octave[sigma_idx + 1, y_idx, x_idx] - 2 * DoG_octave[sigma_idx, y_idx, x_idx] + DoG_octave[
        sigma_idx - 1, y_idx, x_idx]

    diff = (DoG_octave[sigma_idx, y_idx + 1, x_idx + 1] - DoG_octave[sigma_idx, y_idx + 1, x_idx - 1]) - (
                DoG_octave[sigma_idx, y_idx - 1, x_idx + 1] - DoG_octave[sigma_idx, y_idx - 1, x_idx - 1])
    dxy = diff / 4.0

    diff = (DoG_octave[sigma_idx + 1, y_idx, x_idx + 1] - DoG_octave[sigma_idx + 1, y_idx, x_idx - 1]) - (
                DoG_octave[sigma_idx - 1, y_idx, x_idx + 1] - DoG_octave[sigma_idx - 1, y_idx, x_idx - 1])
    dxs = diff / 4.0

    diff = (DoG_octave[sigma_idx + 1, y_idx + 1, x_idx] - DoG_octave[sigma_idx + 1, y_idx - 1, x_idx]) - (
                DoG_octave[sigma_idx - 1, y_idx + 1, x_idx] - DoG_octave[sigma_idx - 1, y_idx - 1, x_idx])
    dys = diff / 4.0

    return dxx, dyy, dss, dxy, dxs, dys


def get_fst_der(DoG_octave: np.array, sigma_idx: int, y_idx: int, x_idx: int) -> (float, float, float):
    dx = (DoG_octave[sigma_idx, y_idx, x_idx + 1] - DoG_octave[sigma_idx, y_idx, x_idx - 1]) / 2.0
    dy = (DoG_octave[sigma_idx, y_idx + 1, x_idx] - DoG_octave[sigma_idx, y_idx - 1, x_idx]) / 2.0
    d_sigma = (DoG_octave[sigma_idx + 1, y_idx, x_idx] - DoG_octave[sigma_idx - 1, y_idx, x_idx]) / 2.0

    return d_sigma, dy, dx


def cart_to_polar_grad(dx: float, dy: float) -> (float, float):
    magnitude = (dx ** 2 + dy ** 2)**0.5
    theta = (np.arctan2(dy, dx) + np.pi) * 180.0 / np.pi
    return magnitude, theta


def get_grad(patch: np.array, x_idx: int, y_idx: int) -> (float, float) :
    dy = patch[min(patch.shape[0] - 1, y_idx + 1), x_idx] - patch[max(0, y_idx - 1), x_idx]
    dx = patch[y_idx, min(patch.shape[1] - 1, x_idx + 1)] - patch[y_idx, max(0, x_idx - 1)]
    return cart_to_polar_grad(dx, dy)


def quantize_orientation(theta: float, num_bins: int) -> int:
    return int(theta) // (360 // num_bins)


def fit_parabola(histogram: list, bin_idx: int, bins_angle: int) -> float:
    centerval = bin_idx * bins_angle + bins_angle / 2.
    if bin_idx == len(histogram) - 1:
        rightval = 360 + bins_angle / 2.
    else:
        rightval = (bin_idx + 1) * bins_angle + bins_angle / 2.
    if bin_idx == 0:
        leftval = -bins_angle / 2.
    else:
        leftval = (bin_idx - 1) * bins_angle + bins_angle / 2.
    A = np.array([
        [centerval ** 2, centerval, 1],
        [rightval ** 2, rightval, 1],
        [leftval ** 2, leftval, 1]])
    b = np.array([
        histogram[bin_idx],
        histogram[(bin_idx + 1) % len(histogram)],
        histogram[(bin_idx - 1) % len(histogram)]])

    x = np.linalg.lstsq(A, b, rcond=None)[0]
    if x[0] == 0:
        x[0] = 1e-6
    return -x[1] / (2 * x[0])


def assign_orientation(keypoints: list, octave: np.array, num_bins=36) -> np.array:
    fitted_keypoints = list()
    bins_angle = 360 // num_bins

    for kp in keypoints:
        histogram = np.array([0.0] * num_bins)
        sigma_idx, y_idx, x_idx = int(kp[0]), int(kp[1]), int(kp[2] + 0.5)
        sigma_idx = np.clip(sigma_idx, 0, octave.shape[2]-1)

        sigma = 1.5 * sigma_idx
        filter_ = gaussian(sigma)
        if filter_.shape[0] == 1:
            continue
        if sigma_idx >= octave.shape[0]:
            continue
        width = int(2 * np.ceil(sigma) + 1)
        DoG_img = octave[sigma_idx]
        last = width + 1
        if 2*last > filter_.shape[0]:
            last = filter_.shape[0]//2

        for around_y in range(-width, last + 1):
            for around_x in range(-width, last + 1):
                x, y = x_idx + around_x, y_idx + around_x

                if x < 0 or x > octave.shape[1] - 1:
                    continue
                elif y < 0 or y > octave.shape[0] - 1:
                    continue
                if around_x + last < 0 or around_x + last > filter_.shape[1]-1:
                    continue
                elif around_y + last < 0 or around_y + last > filter_.shape[0]-1:
                    continue

                magnitude, theta = get_grad(DoG_img, x, y)
                weight = filter_[around_y + width, around_x + width] * magnitude

                bin_number = quantize_orientation(theta, num_bins)
                histogram[bin_number] += weight

        max_bin_idx = np.argmax(histogram)
        fitted_keypoints.append([*kp, fit_parabola(histogram, max_bin_idx, bins_angle)])

        max_magnitude = np.max(histogram)
        for bin_idx, magnitide in enumerate(histogram):
            if max_magnitude == max_bin_idx:
                continue
            if 0.8 * max_magnitude <= magnitide:
                fitted_keypoints.append([*kp, fit_parabola(histogram, bin_idx, bins_angle)])
    return np.array(fitted_keypoints)


def get_patch_grads(patch: np.array) -> (float, float):
    r1 = np.zeros_like(patch)
    r1[-1] = patch[-1]
    r1[:-1] = patch[1:]

    r2 = np.zeros_like(patch)
    r2[0] = patch[0]
    r2[1:] = patch[:-1]

    dy = r1 - r2

    r1[:, -1] = patch[:, -1]
    r1[:, :-1] = patch[:, 1:]

    r2[:, 0] = patch[:, 0]
    r2[:, 1:] = patch[:, :-1]

    dx = r1 - r2

    return dx, dy


def get_histogram_for_subregion(magnitude: float, theta: float,
                                num_bin: int, reference_angle: float,
                                bin_width: float, subregion_w: float) -> np.array:
    hist = np.array([0.0] * num_bin)
    c = subregion_w / 2 - .5

    for i, (mag, angle) in enumerate(zip(magnitude, theta)):
        angle = (angle - reference_angle) % 360
        binno = quantize_orientation(angle, num_bin)
        vote = mag

        hist_interp_weight = 1 - abs(angle - (binno * bin_width + bin_width / 2)) / (bin_width / 2)
        vote *= max(hist_interp_weight, 1e-6)

        gy, gx = np.unravel_index(i, (subregion_w, subregion_w))
        x_interp_weight = max(1 - abs(gx - c) / c, 1e-6)
        y_interp_weight = max(1 - abs(gy - c) / c, 1e-6)
        vote *= x_interp_weight * y_interp_weight

        hist[binno] += vote

    return hist


def visualisation(img_rgb: np.array, rescaled_keypoints_pyramid: np.array, sigma: list, num_octaves=4):
    for i in range(num_octaves):
        x = rescaled_keypoints_pyramid[i][:, 2]
        y = rescaled_keypoints_pyramid[i][:, 1]
        plt.scatter(x, y, c='r', s=2.5)

        # draw circle for each of key point
        for x_, y_ in zip(x, y):
            theta = np.linspace(0, 2 * np.pi , 150)
            radius = sigma[i]
            a = radius * np.cos(theta)
            b = radius * np.sin(theta)
            plt.plot(x_ + a, y_ + b, c='y')
    plt.imshow(img_rgb)
    plt.show()


def cnv(x: np.array, y: np.array, mode: any) -> np.array:
    return np.fft.irfft2(np.fft.rfft2(x) * np.fft.rfft2(y, x.shape))


def keypoint_not_in_octave(corrected_key_point: np.array, DoG_octave: np.array) -> bool:
    return corrected_key_point[0] >= DoG_octave.shape[0] \
           or corrected_key_point[1] >= DoG_octave.shape[1] \
           or corrected_key_point[2] >= DoG_octave.shape[2] \
           or corrected_key_point[0] < 0 \
           or corrected_key_point[1] < 0 \
           or corrected_key_point[2] < 0
