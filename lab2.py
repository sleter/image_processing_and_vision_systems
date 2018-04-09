import cv2
import numpy as np
from matplotlib import pyplot as plt

def zad1():
    def trackbar1(x):
        print('Trackbar R value: ' + str(x))

    def trackbar2(x):
        print('Trackbar G value: ' + str(x))

    def trackbar3(x):
        print('Trackbar B value: ' + str(x))

    def switch_(x):
        print('Switch value: ' + str(x))


    # Create a black image, a window
    img = np.zeros((300,512,3), np.uint8)
    cv2.namedWindow('image')

    # create trackbars for color change
    cv2.createTrackbar('R','image',0,255,trackbar1)
    cv2.createTrackbar('G','image',0,255,trackbar2)
    cv2.createTrackbar('B','image',0,255,trackbar3)

    # create switch for ON/OFF functionality
    switch = '0 : OFF \n1 : ON'
    cv2.createTrackbar(switch, 'image',0,1,switch_)

    while(1):
        cv2.imshow('image',img)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

        # get current positions of four trackbars
        r = cv2.getTrackbarPos('R','image')
        g = cv2.getTrackbarPos('G','image')
        b = cv2.getTrackbarPos('B','image')
        s = cv2.getTrackbarPos(switch,'image')

        if s == 0:
            img[:] = 0
        else:
            img[:] = [b,g,r]

    cv2.destroyAllWindows()


def zad2():
    img = cv2.imread('img.png', 0)
    ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    ret, thresh2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    ret, thresh3 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
    ret, thresh4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
    ret, thresh5 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)

    titles = ['Original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
    images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

    for i in range(6):
        plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])

    plt.show()

def zad2_todo():
    def trackbar1(x):
        ret, thresh1 = cv2.threshold(img, x, 255, cv2.THRESH_BINARY)
        cv2.imshow('image', thresh1)

    img = cv2.imread('img.png', 0)
    cv2.namedWindow('image')
    cv2.createTrackbar('BINARY', 'image', 0, 255, trackbar1)
    ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
    cv2.imshow('image', thresh)

    while(1):
        bin_ = cv2.getTrackbarPos('BINARY', 'image')
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()

def zad3():
    img = cv2.imread('img.jpg', 0)
    img_scaled = img
    inter1 = cv2.resize(img, dsize=(100,100), interpolation=cv2.INTER_LINEAR)
    inter2 = cv2.resize(img, dsize=(500,500), interpolation=cv2.INTER_AREA)
    inter3 = cv2.resize(img, dsize=(500,500), interpolation=cv2.INTER_CUBIC)
    inter4 = cv2.resize(img, dsize=(500,500), interpolation=cv2.INTER_LANCZOS4)

    titles = ['Original Image', 'INTER_LINEAR', 'INTER_NEAREST', 'INTER_AREA', 'INETR_LANCZOS4']
    images = [img_scaled, inter1, inter2, inter3, inter4]

    for i in range(5):
        cv2.imshow('{}'.format(titles[i]), images[i])


    cv2.waitKey(0)
    cv2.destroyAllWindows()


def zad4():
    img1 = cv2.imread('img.png')
    img2 = cv2.imread('1.png')

    i1 = cv2.resize(img1, dsize=(400, 400), interpolation=cv2.INTER_LINEAR)
    i2 = cv2.resize(img2, dsize=(400, 400), interpolation=cv2.INTER_LINEAR)

    dst = cv2.addWeighted(i1, 0.7, i2, 0.3, 0)

    cv2.imshow('dst', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def zad5():
    def trackbar1(x):
        ret, thresh1 = cv2.threshold(img, x, 255, cv2.THRESH_BINARY)
        cv2.imshow('image', thresh1)

    img1 = cv2.imread('img.png')
    img2 = cv2.imread('1.png')

    i1 = cv2.resize(img1, dsize=(400, 400), interpolation=cv2.INTER_LINEAR)
    i2 = cv2.resize(img2, dsize=(400, 400), interpolation=cv2.INTER_LINEAR)

    dst = cv2.addWeighted(i1, 0.7, i2, 0.3, 0)

    cv2.imshow('dst', dst)

    cv2.namedWindow('image')
    cv2.createTrackbar('BINARY', 'image', 0, 255, trackbar1)


    while(1):
        bin_ = cv2.getTrackbarPos('BINARY', 'image')
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    #zad1()
    #zad2()
    #zad2_todo()
    #zad3()
    zad4()