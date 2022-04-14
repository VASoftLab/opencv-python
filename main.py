import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


def print_modules_version():
    print(f'Opencv2: {cv2.__version__}.')
    print(f'NumPy: {np.__version__}.')


def draw_histogram(img):
    plt.figure(figsize=(9, 3), dpi=150)
    plt.hist(x=img.ravel(),
             bins=256,
             range=[0, 256],
             color='crimson')
    # Histogram Showing Pixel Intensities And Counts
    plt.title('Histogram Showing Pixel Intensities And Counts', color='crimson')
    # Number Of Pixels Belonging To The Pixel Intensity
    plt.ylabel('Number Of Pixels', color='crimson')
    plt.xlabel('Pixel Intensity', color='crimson')
    plt.show()


def draw_contour(img_org, img_thresh):
    contours, hierarchy = cv2.findContours(image=img_thresh,
                                           mode=cv2.RETR_TREE,
                                           method=cv2.CHAIN_APPROX_NONE)
    # Draw contours on the original image
    img_contour = img_org.copy()
    cv2.drawContours(image=img_contour,
                     contours=contours,
                     contourIdx=-1,
                     color=(0, 255, 0),
                     thickness=2,
                     lineType=cv2.LINE_AA)
    cv2.imshow('Contour image', img_contour)
    cv2.imwrite('img\\out\\img_contour.jpg', img_contour)


if __name__ == '__main__':

    # Get installed modules version info
    print_modules_version()

    # Load an original imag
    img_original = cv2.imread('img\\L1030.jpg', cv2.IMREAD_COLOR)

    # Convert imag from BGR to the new format
    img_converted = cv2.cvtColor(img_original, cv2.COLOR_BGR2YCrCb)

    # Equalize the histogram of the Y channel
    img_converted[:, :, 0] = cv2.equalizeHist(img_converted[:, :, 0])
    img_equalized = cv2.cvtColor(img_converted, cv2.COLOR_YCrCb2BGR)

    cv2.imshow('Original image', img_original)
    cv2.imshow('Equalized image', img_equalized)

    #cv2.waitKey()
    #cv2.destroyAllWindows()

    img_gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)

    # Threshold parameters
    thresh = 155    # Threshold value
    maxval = 255    # Maximum value
    blockSize = 11  # Size of a pixel neighborhood that is used to calculate a threshold value for the pixel
    C = 2           # Constant subtracted from the mean or weighted mean

    ret, img_thresh_bunary = cv2.threshold(img_gray, thresh, maxval, cv2.THRESH_BINARY)
    img_thresh_adaptive_mean = cv2.adaptiveThreshold(img_gray, maxval, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                     cv2.THRESH_BINARY, blockSize, C)
    img_thresh_adaptive_gaussian = cv2.adaptiveThreshold(img_gray, maxval, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                         cv2.THRESH_BINARY, blockSize, C)

    cv2.imshow('Gray image', img_gray)
    cv2.imshow('Adaptive mean image', img_thresh_adaptive_mean)
    cv2.imshow('Adaptive gaussian image', img_thresh_adaptive_gaussian)

    # Detect the contours on the binary image using cv22.CHAIN_APPROX_NONE
    draw_contour(img_original, img_thresh_bunary)

    #contours, hierarchy = cv2.findContours(image=img_thresh_bunary,
    #                                       mode=cv2.RETR_TREE,
    #                                       method=cv2.CHAIN_APPROX_NONE)
    # Draw contours on the original image
    #img_contour = img_original.copy()
    #cv2.drawContours(image=img_contour,
    #                 contours=contours,
    #                 contourIdx=-1,
    #                 color=(0, 255, 0),
    #                 thickness=2,
    #                 lineType=cv2.LINE_AA)
    #cv2.imshow('Contour image', img_contour)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite('img\\out\\img_equalized.jpg', img_equalized)
    cv2.imwrite('img\\out\\image_gray.jpg', img_gray)
    cv2.imwrite('img\\out\\adaptive_mean_image.jpg', img_thresh_adaptive_mean)
    cv2.imwrite('img\\out\\img_thresh_adaptive_gaussian.jpg', img_thresh_adaptive_gaussian)

    draw_histogram(img_original)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print('DONE...')