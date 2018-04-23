import cv2
import numpy as np
from matplotlib import pyplot as plt

def zad1():
    img = cv2.imread('lena_noise.bmp')
    img2 = cv2.imread('lena_salt_and_pepper.bmp')

    kernel = np.ones((5, 5), np.float32) / 25
    dst = cv2.filter2D(img, -1, kernel)
    dst2 = cv2.filter2D(img2, -1, kernel)

    plt.subplot(131), plt.imshow(img), plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(132), plt.imshow(dst), plt.title('Averaging')
    plt.xticks([]), plt.yticks([])
    plt.subplot(133), plt.imshow(dst2), plt.title('Averaging2')
    plt.xticks([]), plt.yticks([])
    plt.show()

def zad2_erosion():
    def trackbar1(x):
        ret, thresh1 = cv2.threshold(img, x, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        erosion = cv2.erode(thresh1, kernel, iterations=1)
        cv2.imshow('image', erosion)

    img = cv2.imread('lena_noise.bmp', 0)
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

def zad2_dilation():
    def trackbar1(x):
        ret, thresh1 = cv2.threshold(img, x, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        dilation = cv2.dilate(thresh1, kernel, iterations=1)
        cv2.imshow('image', dilation)

    img = cv2.imread('lena_noise.bmp', 0)
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
    img = cv2.imread('img.png', 0)
    img2 = cv2.imread('img.png')
    plt.subplot(131), plt.hist(img.ravel(), 256, [0, 256]), plt.title('Greyscale')
    plt.xticks([]), plt.yticks([])
    plt.subplot(132), plt.hist(img2.ravel(), 256, [0, 256]), plt.title('Colour')
    plt.xticks([]), plt.yticks([])
    plt.subplot(133), plt.imshow(img2), plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.show()


def zad3_():
    img = cv2.imread('img.png', 0)

    hist, bins = np.histogram(img.flatten(), 256, [0, 256])

    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()

    plt.plot(cdf_normalized, color='b')
    plt.hist(img.flatten(), 256, [0, 256], color='r')
    plt.xlim([0, 256])
    plt.legend(('cdf', 'histogram'), loc='upper left')
    plt.show()

def zad4():
    img = cv2.imread('img.png', 0)

    # create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(img)

    cv2.imwrite('clahe.jpg', cl1)

def zad5():
    pom=[]
    def point(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            pom.append([x,y])
            print('click')

    img = cv2.imread('road.jpg')
    img = cv2.resize(img, (1000, 1000))
    rows, cols, ch = img.shape
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', point)
    while (1):
        cv2.imshow('image', img)
        if cv2.waitKey(20) & 0xFF == 27:
            break
    pom = np.array(pom)
    pts1 = np.float32(pom)
    pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])

    print(pts1)

    M = cv2.getPerspectiveTransform(pts1, pts2)

    dst = cv2.warpPerspective(img, M, (300, 300))
    cv2.imshow('image',dst)
    cv2.destroyAllWindows()





if __name__ == "__main__":
    #zad1()
    #zad2_erosion()
    #zad2_dilation()
    #zad3()
    #zad3_()
    #zad4()
    zad5()
