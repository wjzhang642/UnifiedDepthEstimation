import random
import numpy as np
import scipy


cache_index = {}
cache_boundary = {}
projection_funs = {
    0: lambda rf, f: rf / f,  # equidistant
    1: lambda rf, f: 2 * np.arcsin(rf / 2 / f),  # equisolid
    2: lambda rf, f: np.arcsin(rf / f),  # orthogonal
    3: lambda rf, f: 2 * np.arctan(rf / 2 / f),  # stereographic
}
projection_mask = {
    0: lambda rf, f: rf < f * np.pi / 2,  # equidistant
    1: lambda rf, f: rf < 2 * f * np.sin(np.pi/4),  # equisolid
    2: lambda rf, f: rf < f,  # orthogonal
    3: lambda rf, f: rf < 2 * f * np.tan(np.pi/4),  # stereographic
}


def pinhole2fisheye(sample, focal_length=200, projection_model=None, crop_valid=False, enable_cache=False):
    """
    The combination of highly varied image resolutions and fine-grained focal length perturbations in your dataset
    may increase memory overhead when enable_cache turned on.
    """
    image = sample['image']
    h, w = image.shape[:2]

    x = np.arange(w)
    y = np.arange(h)
    x, y = np.meshgrid(x, y)

    cache_key = (h, w, projection_model, focal_length)
    if enable_cache and cache_key in cache_index:
        yf, yx = cache_key[cache_key]
        y_min, y_max, x_min, x_max = cache_boundary[cache_key]
    else:
        cx, cy = w / 2, h / 2
        dx = x - cx
        dy = y - cy

        rf = np.sqrt(dx ** 2 + dy ** 2)
        gamma = np.arctan2(dy, dx)

        if projection_model is None:
            pm_id = random.randint(0, 3)
            valid_mask = projection_mask[pm_id](rf, focal_length)
            theta = np.zeros_like(rf)
            theta[valid_mask] = projection_funs[pm_id](rf[valid_mask], focal_length)
        else:
            valid_mask = projection_mask[projection_model](rf, focal_length)
            theta = np.zeros_like(rf)
            theta[valid_mask] = projection_funs[projection_model](rf[valid_mask], focal_length)

        rc = np.zeros_like(theta)
        rc[valid_mask] = np.tan(theta[valid_mask]) * focal_length

        xp, yp = np.zeros_like(rc), np.zeros_like(rc)
        xp[valid_mask] = rc[valid_mask] * np.cos(gamma[valid_mask])
        yp[valid_mask] = rc[valid_mask] * np.sin(gamma[valid_mask])

        xp[valid_mask] = xp[valid_mask] + cx
        yp[valid_mask] = yp[valid_mask] + cy

        valid_mask = valid_mask & (xp >= 0) & (xp < w) & (yp >= 0) & (yp < h)

        y_indices, x_indices = np.nonzero(valid_mask)
        x_min, x_max = x_indices.min(), x_indices.max()
        y_min, y_max = y_indices.min(), y_indices.max()

        if enable_cache:
            cache_index[cache_key] = (xp, yp, valid_mask)
            cache_boundary[cache_key] = (y_min, y_max, x_min, x_max)

    result_sample = {}
    for name, arr in sample.items():
        dst_arr = np.zeros(arr.shape, dtype=arr.dtype)

        coordinates = [yp, xp]

        if arr.ndim == 3:  # RGB
            for c in range(arr.shape[2]):
                interpolated_values = scipy.ndimage.map_coordinates(arr[:, :, c], coordinates, order=1, mode='constant',
                                                                    cval=0.0)
                dst_arr[:, :, c][valid_mask] = interpolated_values[valid_mask]
        else:  # Depth
            interpolated_values = scipy.ndimage.map_coordinates(arr, coordinates, order=0, mode='constant', cval=0.0)
            dst_arr = interpolated_values

        result_sample[name] = dst_arr[y_min:y_max, x_min:x_max] if crop_valid else dst_arr

    return result_sample
