# -*- coding: utf-8 -*-

import numpy as np

# This not a colour index, but a colour space


def BGR2bgr(img, normalize=True):
    img_norm = np.zeros_like(img, dtype='float32')
    B = img[..., 0].astype('float32')
    G = img[..., 1].astype('float32')
    R = img[..., 2].astype('float32')
    s = B + G + R
    img_norm[..., 0] = np.divide(B, s, where=s != 0)
    img_norm[..., 1] = np.divide(G, s, where=s != 0)
    img_norm[..., 2] = np.divide(R, s, where=s != 0)
    return img_norm


def BGR2NDI(img):
    "Normalised Difference Index"
    G = img[..., 1].astype('float32')
    R = img[..., 2].astype('float32')
    frac = np.divide(G-R, G+R, where=(G+R) != 0)
    NDI = 128 * (frac + 1)
    return NDI


def BGR2ExG(img):
    "Excess Green Index"
    img_bgr = BGR2bgr(img)
    b = img_bgr[..., 0].astype('float32')
    g = img_bgr[..., 1].astype('float32')
    r = img_bgr[..., 2].astype('float32')
    ExG = 2 * g - r - b
    return ExG


def BGR2ExR(img):
    "Excess Red Index"
    G = img[..., 1].astype('float32')
    R = img[..., 2].astype('float32')
    ExR = 1.3 * R - G
    return ExR


def BGR2CIVE(img):
    "Colour Index of Vegetation Extraction"
    B = img[..., 0].astype('float32')
    G = img[..., 1].astype('float32')
    R = img[..., 2].astype('float32')
    CIVE = 0.441 * R - 0.811 * G + 0.385 * B + 18.78745
    return CIVE


def BGR2ExGR(img):
    "Excess Green minus Excess Red Index"
    ExG = BGR2ExG(img)
    ExR = BGR2ExR(img)
    ExGR = ExG - ExR
    return ExGR


def BGR2NGRDI(img):
    "Normalised Green–Red Difference Index"
    G = img[..., 1].astype('float32')
    R = img[..., 2].astype('float32')
    NGRDI = np.divide(G-R, G+R, where=(G+R) != 0)
    return NGRDI


def BGR2VEG(img):
    "Vegetative Index"
    B = img[..., 0].astype('float32')
    G = img[..., 1].astype('float32')
    R = img[..., 2].astype('float32')
    denominator = np.multiply(np.power(R, 0.667), np.power(B, 0.333))
    VEG = np.divide(G, denominator, where=denominator != 0)
    return VEG


def BGR2COM1(img):
    "Combined Indices 1"
    ExG = BGR2ExG(img)
    CIVE = BGR2CIVE(img)
    ExGR = BGR2ExGR(img)
    VEG = BGR2VEG(img)
    COM1 = ExG + CIVE + ExGR + VEG
    return COM1


def BGR2MExG(img):
    "Modified Excess Green Index"
    B = img[..., 0].astype('float32')
    G = img[..., 1].astype('float32')
    R = img[..., 2].astype('float32')
    MExG = 1.262 * G - 0.884 * R - 0.311 * B
    return MExG


def BGR2COM2(img):
    "Combined Indices 2"
    ExG = BGR2ExG(img)
    CIVE = BGR2CIVE(img)
    VEG = BGR2VEG(img)
    COM2 = 0.36 * ExG + 0.47 * CIVE + 0.17 * VEG
    return COM2
