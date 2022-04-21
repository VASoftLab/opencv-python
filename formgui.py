import cv2
import numpy as np
import matplotlib.pyplot as plt

# Global Varibales
title_window = 'HSV Tuner'
lowh = 0
lows = 50
lowv = 0

hih = 10
his = 255
hiv = 255


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


def call_inrrange():
    hsv_min = np.array([lowh, lows, lowv])
    hsv_max = np.array([hih, his, hiv])
    thresh = cv2.inRange(img_hsv, hsv_min, hsv_max)
    draw_contour(img_original, thresh)


def on_hsv_lowh(val):
    global lowh
    lowh = val
    call_inrrange()


def on_hsv_lows(val):
    global lows
    lows = val
    call_inrrange()


def on_hsv_lowv(val):
    global lowv
    lowv = val
    call_inrrange()


def on_hsv_hih(val):
    global hih
    hih = val
    call_inrrange()


def on_hsv_his(val):
    global his
    his = val
    call_inrrange()


def on_hsv_hiv(val):
    global hiv
    hiv = val
    call_inrrange()


if __name__ == '__main__':

    enable_correction = False

    img_original = cv2.imread('img\\L1036.jpg', cv2.IMREAD_COLOR)
    # img_original = cv2.imread('img\\ubuntu.jpg', cv2.IMREAD_COLOR)
    # img_original = cv2.imread('img\\pepsi.jpg', cv2.IMREAD_COLOR)

    img_original_small = cv2.resize(img_original, (640, 400))

    cv2.namedWindow('Original image')
    cv2.moveWindow('Original image', 10, 10)
    cv2.imshow('Original image', img_original_small)

    if enable_correction:
        hih = 21
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
    cv2.createTrackbar('Low H', title_window, 0, 255, on_hsv_lowh)
    cv2.createTrackbar('Low S', title_window, 0, 255, on_hsv_lows)
    cv2.createTrackbar('Low V', title_window, 0, 255, on_hsv_lowv)
    cv2.createTrackbar('Hi H', title_window, 0, 255, on_hsv_hih)
    cv2.createTrackbar('Hi S', title_window, 0, 255, on_hsv_his)
    cv2.createTrackbar('Hi V', title_window, 0, 255, on_hsv_hiv)

    cv2.setTrackbarPos('Low H', title_window, lowh)
    cv2.setTrackbarPos('Low S', title_window, lows)
    cv2.setTrackbarPos('Low V', title_window, lowv)
    cv2.setTrackbarPos('Hi H', title_window, hih)
    cv2.setTrackbarPos('Hi S', title_window, his)
    cv2.setTrackbarPos('Hi V', title_window, hiv)

    cv2.waitKey()
    cv2.destroyAllWindows()