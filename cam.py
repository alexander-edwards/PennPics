import cv2
import time
from PIL import Image 
import os

def show_webcam(mirror=False):
    cam = cv2.VideoCapture(0)
    while True:
        ret_val, img = cam.read()
        time.sleep(2)
        if mirror: 
            img = cv2.flip(img, 1)
        cv2.imshow('my webcam', img)
        cv2.imwrite('picture.jpg', img)
        os.system('git add *')
        os.system('git commit -m "asdfasdf"')
        os.system('git push origin master')
        if cv2.waitKey(1) == 27: 
            break  # esc to quit
    cv2.destroyAllWindows()


def main():
    show_webcam(mirror=True)


if __name__ == '__main__':
    main()