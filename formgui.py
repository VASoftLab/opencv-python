import cv2
import numpy as np

title_window = 'HSV Tuner'
lowh = 0
lows = 21
lowv = 12
hih = 10
his = 255
hiv = 255


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


def call_inrrange():
    global img_hsv
    global img_original
    global lowh
    global lows
    global lowv
    global hih
    global his
    global hiv

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
    img_original = cv2.imread('img\\L1030.jpg', cv2.IMREAD_COLOR)
    img_hsv = cv2.cvtColor(img_original, cv2.COLOR_BGR2HSV)

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