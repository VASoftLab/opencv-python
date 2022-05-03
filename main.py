import cv2
import numpy as np
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


if __name__ == '__main__':

    # Get installed modules version info
    print_modules_version()

    # Load an original imag
    img_original = cv2.imread('/home/va/Sources/opencv-python/img/L1030.jpg', cv2.IMREAD_COLOR)

    # Convert imag from BGR to the new format
    img_converted = cv2.cvtColor(img_original, cv2.COLOR_BGR2YCrCb)

    # Equalize the histogram of the Y channel
    img_converted[:, :, 0] = cv2.equalizeHist(img_converted[:, :, 0])
    img_equalized = cv2.cvtColor(img_converted, cv2.COLOR_YCrCb2BGR)

    cv2.imshow('Original image', img_original)
    cv2.imshow('Equalized image', img_equalized)

    draw_histogram(img_original)
    draw_histogram(img_equalized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print('DONE...')