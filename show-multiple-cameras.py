import cv2
import threading

# https://stackoverflow.com/questions/29664399/capturing-video-from-two-cameras-in-opencv-at-once
class camThread(threading.Thread):
    def __init__(self, previewName, camID):
        threading.Thread.__init__(self)
        self.previewName = previewName
        self.camID = camID

    def run(self):
        print("Starting " + self.previewName)
        camPreview(self.previewName, self.camID)


def camPreview(previewName, camID):
    cv2.namedWindow(previewName)
    cam = cv2.VideoCapture(camID)
    if cam.isOpened():
        rval, frame = cam.read()
    else:
        rval = False

    while rval:
        cv2.imshow(previewName, frame)
        rval, frame = cam.read()
        key = cv2.waitKey(20)
        if key == 27:  # exit on ESC
            break
    cv2.destroyWindow(previewName)


if __name__ == '__main__':
    # Create threads as follows
    thread1 = camThread("Camera 1", 1)
    thread2 = camThread("Camera 2", 2)
    thread1.start()
    thread2.start()
    print()
    print("Active threads", threading.active_count())