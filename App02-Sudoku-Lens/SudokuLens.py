import cv2, sys
import numpy as np

# img = cv2.imread('sudoku-flat.png')
img = cv2.imread('sudoku.jpg')

# Resizing Image (max dim = size)
size = 640
h, w = img.shape[:2]
if h > w:
    hnew = size
    wnew = (size*w)//h
else:
    wnew = size
    hnew = (size*h)//w
    
img = cv2.resize(img, (wnew, hnew))
img_cpy = img.copy()

corners = []

def click_and_crop(event, x, y, flags, param):
    global corners
    
    if event == cv2.EVENT_LBUTTONDBLCLK:
        print(x, y)
        corners.append([x, y])
        cv2.circle(img_cpy, (x, y), 2, (0, 0, 255), 2)
        cv2.imshow("image", img_cpy)
                
        
cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)

while True:
    cv2.imshow('image',img_cpy)
    key = cv2.waitKey(1) & 0xFF
    if key == 13:
        break
    if key == 27:
        sys.exit(0)

cv2.destroyWindow('image')

if len(corners) == 4:
    corners = np.float32(corners)
    M = cv2.getPerspectiveTransform(corners,np.float32([[0,0],[size,0],[size,size],[0,size]]))
    cropped = cv2.warpPerspective(img,M,(size,size))
elif len(corners) == 0:
    cropped = img.copy()
cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)     
cv2.imshow("crop_img", cropped)
cv2.waitKey(0)
cv2.destroyWindow("crop_img")

region_1d = [(i*70+i+i//3, (i+1)*70+i+i//3) for i in range(9)]
region_2d = [(x, y) for x in region_1d for y in region_1d]

for i, dim in enumerate(region_2d):
    p, q = dim
    x1, x2 = p
    y1, y2 = q
    print(x1, x2, y1, y2)
    num_i = cropped[y1:y2, x1:x2]
    cv2.imshow(f"num_{i}", num_i)
    
cv2.waitKey(0)    
cv2.destroyAllWindows()