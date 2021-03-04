import cv2
import matplotlib.pyplot as plt
import numpy as np


def get_confusion_matrix_intersection_mats(groundtruth, predicted):
    """
    Returns a dictionary of 4 boolean numpy arrays containing True at TP, FP, FN, TN.
    """
    confusion_matrix_arrs = {}

    groundtruth_inverse = np.logical_not(groundtruth)
    predicted_inverse = np.logical_not(predicted)

    confusion_matrix_arrs["tp"] = np.logical_and(groundtruth, predicted)
    confusion_matrix_arrs["tn"] = np.logical_and(
        groundtruth_inverse, predicted_inverse)
    confusion_matrix_arrs["fp"] = np.logical_and(
        groundtruth_inverse, predicted)
    confusion_matrix_arrs["fn"] = np.logical_and(
        groundtruth, predicted_inverse)

    return confusion_matrix_arrs


def get_confusion_matrix_overlaid_mask(image, groundtruth, predicted, alpha, colors):
    """
    Returns overlay the 'image' with a color mask where TP, FP, FN, TN are
    each a color given by the 'colors' dictionary
    """
    # image =  cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    masks = get_confusion_matrix_intersection_mats(groundtruth, predicted)
    plt.imshow(masks["tp"])
    color_mask = np.zeros_like(image)
    for label, mask in masks.items():
        color = colors[label]
        mask_rgb = np.zeros_like(image)
        mask_rgb[mask != 0] = color  # Insert color same index of mask
        color_mask += mask_rgb
    # Image Blending
    return cv2.addWeighted(image, alpha, color_mask, 1 - alpha, 0)


def plot_evaluation(image_path, image, prediction_mask, target_mask, validation_metrics, evolution=None):
    if (evolution):
        fig, ax = plt.subplots(nrows=1, ncols=6, figsize=(23, 3), constrained_layout = True)
    else:
        fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(18, 3), constrained_layout = True)
    fig.patch.set_facecolor('white')

    alpha = 0.5

    confusion_matrix_colors = {
        'tp': (0, 255, 0),  # green
        'fp': (255, 0, 0),  # red
        'fn': (255, 0, 0),  # red
        'tn': (0, 0, 255)  # blue
    }

    # confusion_matrix_colors = {
    #   'tp': (0, 255, 255),  #cyan
    #   'fp': (255, 0, 255),  #magenta
    #   'fn': (255, 255, 0),  #yellow
    #   'tn': (0, 0, 0)     #black
    #   }
    validation_mask = get_confusion_matrix_overlaid_mask(
        image, target_mask, prediction_mask, alpha, confusion_matrix_colors)

    ax[0].axis('off')
    ax[1].axis('off')
    ax[2].axis('off')
    ax[4].axis('off')

    ax[0].title.set_text('Real Image')
    ax[0].imshow(image, cmap='gray')

    ax[1].title.set_text('Target mask')
    ax[1].imshow(target_mask, cmap='gray')

    ax[2].title.set_text('Predicted mask')
    ax[2].imshow(prediction_mask, cmap='gray')

    ax[4].title.set_text('Validation mask')
    ax[4].imshow(validation_mask, cmap='gray')


    metrics_text = "\n"
    for k, v in validation_metrics.items():
        metrics_text = metrics_text + k + " : " + str(round(v, 3)) + "\n"

    ax[3].text(0.5, 0.5, metrics_text, ha='center', va='center', fontsize=15)
    ax[3].set_xticks([])
    ax[3].set_yticks([])
    ax[3].title.set_text("Metrics")

    if (evolution):
        ax[5].imshow(image, cmap="gray", interpolation='bilinear')
        ax[5].set_axis_off()
        contour = ax[5].contour(evolution[0], [0.5], colors='C3', linewidths=3)
        contour = ax[5].contour(evolution[-5], [0.5],
                                colors='C4', linewidths=3)
        contour = ax[5].contour(evolution[-4], [0.5],
                                colors='C5', linewidths=3)
        contour = ax[5].contour(evolution[-3], [0.5],
                                colors='C6', linewidths=3)
        contour = ax[5].contour(evolution[-2], [0.5],
                                colors='C7', linewidths=3)
        contour = ax[5].contour(evolution[-1], [0.5],
                                colors='C8', linewidths=3)
        title = "Morphological ACWE evolution"
        ax[5].set_title(title, fontsize=12)
    plt.show()
