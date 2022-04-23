import cv2
import numpy as np
import matplotlib.pyplot as plt

# Global Variables
title_window = 'HSV Tuner'

low_h = 0
low_s = 50
low_v = 0
hi_h = 10
hi_s = 255
hi_v = 255


def draw_contour(img_org, img_thresh):
    contours, hierarchy = cv2.findContours(image=img_thresh,
                                           mode=cv2.RETR_EXTERNAL,
                                           method=cv2.CHAIN_APPROX_NONE)
    # Draw contours on the original image
    img_contour = img_org.copy()
    cv2.drawContours(image=img_contour,
                     contours=contours,
                     contourIdx=-1,
                     color=(0, 255, 0),
                     thickness=2,
                     lineType=cv2.LINE_AA)
    img_contour = cv2.resize(img_contour, (1280, 800))
    cv2.namedWindow('Contour image')
    cv2.moveWindow('Contour image', 10 + 640 + 10, 10)
    cv2.imshow('Contour image', img_contour)


def draw_histogram(img_org, img_cor):
    plt.figure(figsize=(8, 5), dpi=150)
    x_org = img_org.ravel()
    x_cor = img_cor.ravel()

    plt.subplot(211)
    plt.title('Histogram Showing Pixel Intensities And Counts', color='crimson')
    plt.hist(x=x_org,
             bins=256,
             range=[0, 256],
             color='crimson')
    plt.ylabel('Number Of Pixels', color='crimson')

    plt.subplot(212)
    plt.hist(x=x_cor,
             bins=256,
             range=[0, 256],
             color='crimson')
    plt.ylabel('Number Of Pixels', color='crimson')
    plt.xlabel('Pixel Intensity', color='crimson')
    plt.show()


def call_inrange():
    hsv_min = np.array([low_h, low_s, low_v])
    hsv_max = np.array([hi_h, hi_s, hi_v])
    thresh = cv2.inRange(img_hsv, hsv_min, hsv_max)
    draw_contour(img_original, thresh)


def on_low_h(val):
    global low_h
    low_h = val
    call_inrange()


def on_low_s(val):
    global low_s
    low_s = val
    call_inrange()


def on_low_v(val):
    global low_v
    low_v = val
    call_inrange()


def on_hi_h(val):
    global hi_h
    hi_h = val
    call_inrange()


def on_hi_s(val):
    global hi_s
    hi_s = val
    call_inrange()


def on_hi_v(val):
    global hi_v
    hi_v = val
    call_inrange()


if __name__ == '__main__':

    enable_correction = True

    # img_original = cv2.imread('img\\L1038.jpg', cv2.IMREAD_COLOR)
    img_original = cv2.imread('img\\ubuntu.jpg', cv2.IMREAD_COLOR)
    # img_original = cv2.imread('img\\pepsi.jpg', cv2.IMREAD_COLOR)

    img_original_small = cv2.resize(img_original, (640, 400))

    cv2.namedWindow('Original image')
    cv2.moveWindow('Original image', 10, 10)
    cv2.imshow('Original image', img_original_small)

    if enable_correction:
        low_h = 0
        low_s = 40
        low_v = 10
        hi_h = 5 # 21 for ubuntu and pepsi
        hi_s = 255
        hi_v = 255
        img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2YCrCb)
        img_original[:, :, 0] = cv2.equalizeHist(img_original[:, :, 0])
        img_corrected = cv2.cvtColor(img_original, cv2.COLOR_YCrCb2BGR)
        img_corrected_small = cv2.resize(img_corrected, (640, 400))
        cv2.namedWindow('Corrected image')
        cv2.moveWindow('Corrected image', 10, 10 + 400 + 25)
        cv2.imshow('Corrected image', img_corrected_small)
        draw_histogram(img_original, img_corrected)
        img_original = img_corrected

    img_hsv = cv2.cvtColor(img_original, cv2.COLOR_BGR2HSV)
    img_hsv[:, :, 2] = cv2.equalizeHist(img_hsv[:, :, 2])
    img_hsv = cv2.fastNlMeansDenoisingColored(img_hsv, None, 10, 10, 7, 21)

    cv2.namedWindow(title_window)
    cv2.createTrackbar('Low H', title_window, 0, 255, on_low_h)
    cv2.createTrackbar('Low S', title_window, 0, 255, on_low_s)
    cv2.createTrackbar('Low V', title_window, 0, 255, on_low_v)
    cv2.createTrackbar('Hi H', title_window, 0, 255, on_hi_h)
    cv2.createTrackbar('Hi S', title_window, 0, 255, on_hi_s)
    cv2.createTrackbar('Hi V', title_window, 0, 255, on_hi_v)

    cv2.setTrackbarPos('Low H', title_window, low_h)
    cv2.setTrackbarPos('Low S', title_window, low_s)
    cv2.setTrackbarPos('Low V', title_window, low_v)
    cv2.setTrackbarPos('Hi H', title_window, hi_h)
    cv2.setTrackbarPos('Hi S', title_window, hi_s)
    cv2.setTrackbarPos('Hi V', title_window, hi_v)

    cv2.waitKey()
    cv2.destroyAllWindows()