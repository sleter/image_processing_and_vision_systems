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


def check_time(func):
    def fun_wrapper(interpolation_, img, window_name):
        e1 = cv2.getTickCount()
        func(interpolation_, img, window_name)
        e2 = cv2.getTickCount()
        time = (e2 - e1) / cv2.getTickFrequency()
        print(window_name+": "+str(time)+" seconds")
    return fun_wrapper

@check_time
def resize_(interpolation_, img, window_name):
    pom = cv2.resize(img, dsize=(500,500), interpolation=interpolation_)
    cv2.imshow(window_name,pom)


def zad3():
    img = cv2.imread('img.png', 0)

    resize_(cv2.INTER_LINEAR, img, "INTER_LINEAR")
    resize_(cv2.INTER_AREA, img, "INTER_AREA")
    resize_(cv2.INTER_CUBIC, img, "INTER_CUBIC")
    resize_(cv2.INTER_LANCZOS4, img, "INTER_LANCZOS4")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def zad4():
    img1 = cv2.imread('img.png')
    img2 = cv2.imread('1.png')

    i1 = cv2.resize(img1, dsize=(400, 400))
    i2 = cv2.resize(img2, dsize=(400, 400))

    dst = cv2.addWeighted(i1, 0.7, i2, 0.3, 0)

    cv2.imshow('dst', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def nwm():
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
    input = input("$> ")
    input = str(input)
    if(input == '1'):
        zad1()
    elif(input == '2'):
        zad2()
    elif(input == '2todo'):
        zad2_todo()
    elif(input == '3'):
        zad3()
    elif(input == '4'):
        zad4()
    elif(input == 'nwm'):
        nwm()
    else:
        print("nope")
