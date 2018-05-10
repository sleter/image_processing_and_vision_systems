#Canny algorithm
import cv2

img = cv2.imread('img.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(img, 100, 200)
cv2.imshow("window", edges)
cv2.waitKey(0)


