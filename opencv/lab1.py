import cv2

def zad1():
    cap = cv2.VideoCapture(0)
    key = ord('a')
    while key != ord('q'):
        ret, frame = cap.read()
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img_filtered = cv2.GaussianBlur(img_gray, (7, 7), 1.5)
        img_edges = cv2.Canny(img_filtered, 0, 30, 3)
        cv2.imshow('result', img_edges)
        key = cv2.waitKey(30)
    cap.release()
    cv2.destroyAllWindows()

def zad2():
    img = cv2.imread('img.png', 0)
    cv2.imshow('image', img)
    cv2.imwrite('imggrey.png', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def zad3():
    img = cv2.imread('img.png')

    px = img[220, 270]
    print("Pixel {} ".format(px))

    img = cv2.imread('img.png',0)

    px = img[220, 270]
    print("Pixel value [220, 270] {} ".format(px))

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def zad4():
    img = cv2.imread('AdditiveColor.png')
    b, g, r = cv2.split(img)
    cv2.imshow('image1', b)
    cv2.imshow('image2', g)
    cv2.imshow('image3', r)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def zad5():
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))
    pom =0
    while (True):
        pom+=1
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        if(pom==3):
            out.write(frame)
            pom=0
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def zad6():
    cap = cv2.VideoCapture('Wildlife.mp4')
    while (cap.isOpened()):
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        while cv2.waitKey(1) & 0xFF != ord(' '):
            pass
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def zad7():
    img = cv2.imread('img.png')
    img = cv2.rectangle(img, (384, 0), (510, 128), (0, 255, 0), 3)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, 'Umim obrazy!', (10, 500), font, 4, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    zad2()