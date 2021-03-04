import numpy as np

def Scale(img, feature_wise=True):
    "Scale (feature-wise) to [0,1]"
    dim = img.ndim
    if feature_wise and dim > 2:
        axis = tuple(range(dim-1))
        min_feat = np.min(img, axis=axis)
        max_feat = np.max(img, axis=axis)
        num = img-min_feat
        den = max_feat-min_feat
        return np.divide(num, den, where=den != 0)
    else:
        min_total = np.min(img)
        max_total = np.max(img)
        num = img-min_total
        den = max_total-min_total
        return np.divide(num, den, where=den != 0)
