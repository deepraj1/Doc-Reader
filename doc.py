import cv2
import numpy as np

w,h=480,640

def reOrder(points):
    points = points.reshape((4,2))
    # print(points)
    newpoints = np.zeros((4,1,2),np.int32)
    add= points.sum(1)
    # print(points[np.argmax(add)])
    newpoints[0] = points[np.argmin(add)]
    newpoints[3] = points[np.argmax(add)]
    diff = np.diff(points,axis=1)
    newpoints[1] = points[np.argmin(diff)]
    newpoints[2] = points[np.argmax(diff)]
    # print(newpoints)
    return newpoints


def wrapPars(img,points):
    # print(points.shape)
    pt1=np.float32(points)
    pt2=np.float32([[0,0],[w,0],[0,h],[w,h]])
    matrix = cv2.getPerspectiveTransform(pt1,pt2)
    output = cv2.warpPerspective(img,matrix,(w,h))
    return output


def getContour(img):
    contour,hei =cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    maxArea=0;
    points =np.array([])
    for cvt in contour:
        area = cv2.contourArea(cvt)

        pari = cv2.arcLength(cvt,True)
        approx = cv2.approxPolyDP(cvt,0.02*pari,True)
        # print(len(approx))
        if len(approx)==4 and area>maxArea:
            points = approx
            maxArea = area

    print(points,len(points))
    cv2.drawContours(imgcopy, points, -1, (0, 255, 0), 3)
    return points


def imgWrap(img):
    kernel = np.ones((5,5),np.uint8)
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray,(7,7),1)
    imgCanny = cv2.Canny(imgBlur,150,150)
    imgDiat = cv2.dilate(imgCanny,kernel,iterations=1)
    return imgDiat


img = cv2.imread("src/doc1.jpg")
imgcopy = img.copy()
cv2.resize(img,(640,480))
res = imgWrap(img)
points = getContour(res)
newpoints = reOrder(points)
final = wrapPars(img,newpoints)
cv2.imshow("Img",img)
cv2.imshow("Final",final)
cv2.waitKey(0)