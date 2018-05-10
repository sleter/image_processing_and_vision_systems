import os, sys, cv2
import tkinter as tk
from tkinter import filedialog as fd

def load_images(path):
    images = []
    for filename in os.listdir(path):
        img = cv2.imread(filename)
        if img is not None:
            images.append(img)
            print(filename)
    return images

def showcase(images):
    cv2.namedWindow("result", cv2.WINDOW_NORMAL)
    i = 0
    cv2.imshow('result', images[i])
    while True:
        k = cv2.waitKey(1) & 0xFF
        if k == ord('='):
            i+=1
            if(i >= len(images)):
                i = 0
            cv2.imshow('result', images[i])
        elif k == ord('-'):
            i-=1
            if (i < 0):
                i=len(images)-1
            cv2.imshow('result', images[i])
        elif k == 27:
            sys.exit(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    root = tk.Tk()
    root.withdraw()
    path = fd.askdirectory()
    root.destroy()
    images = load_images(path)
    showcase(images)
    sys.exit(0)