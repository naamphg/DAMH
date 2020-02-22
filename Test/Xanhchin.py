""" T.P.Nam - 13:39 - 5/21/2019 - Đồ án """

import numpy as np
import cv2
import time

pathR = 'D:/Study/HCMUT/HK182/DA/Images/Samples/2Color/Red/'
pathG = 'D:/Study/HCMUT/HK182/DA/Images/Samples/2Color/Green/'

size = 20
train_ids = np.arange(1, 901)


def buildfilename(path, pre, im_ids):
    filename = []
    for im_id in im_ids:
        fn = path + pre + '-' + str(im_id) + '.png'
        filename.append(fn)
    return filename


def resize(filename):
    img = cv2.imread(filename)
    img = cv2.resize(img, (size, size))
    return img


def trainsamples(img_ids):
    filenamegreen = buildfilename(pathG, 'G', img_ids)
    filenamered = buildfilename(pathR, 'R', img_ids)
    filenamefull = filenamegreen + filenamered
    resized = []
    feature = []
    for i in range(len(filenamefull)):
        x = resize(filenamefull[i])
        resized.append(x)
    for i in range(len(resized)):
        x = resized[i]
        f = []
        # blue = 0
        green = 0
        red = 0
        for j in range(len(x)):
            cell = x[j]
            for z in range(len(cell)):
                # blue = blue + cell[j][0]
                green = green + cell[j][1]
                red = red + cell[j][2]
        # f.append(int(blue/(size*size)))
        f.append(int(green / (size * size)))
        f.append(int(red / (size * size)))
        feature.append(f)
    feature = np.array(feature).astype(np.float32)
    return feature


x = trainsamples(train_ids)
k = np.arange(2)
lable = np.repeat(k, len(x) / 2)[:, np.newaxis]
knn = cv2.ml.KNearest_create()
knn.train(x, 0, lable)


def test(filename):
    f = []
    t = []
    img = cv2.imread(filename)
    img = cv2.resize(img, (size, size))
    # blue = 0
    green = 0
    red = 0
    for i in range(len(img)):
        cell = img[i]
        for j in range(len(cell)):
            # blue = blue + cell[j][0]
            green = green + cell[j][1]
            red = red + cell[j][2]
    # f.append(int(blue/(size*size)))
    f.append(int(green / (size * size)))
    f.append(int(red / (size * size)))
    t.append(f)
    sp = np.array(t).astype(np.float32)
    kq1, kq2, kq3, kq4 = knn.findNearest(sp, 1)
    return kq2

soridetect = cv2.CascadeClassifier("cascade1.xml")

cap = cv2.VideoCapture(1)
count = 1
tt = 0
xanh = 0
chin = 0
while(count<351):
    st = time.time()
    '''ret, frame = cap.read()
    if ret != 1:
        print("Video finished")
        break'''
    frame = cv2.imread("D:/Study/HCMUT/HK182/DA/Images/Test/X/TestG-{}.jpg".format(count))
    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sori = soridetect.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=10,
    )
    for (x, y, w, h) in sori:
        img_crop = frame[y + 2:y + h - 2, x + 2:x + w - 2, :]
        cv2.imwrite('object.jpg', img_crop)
        c = test('object.jpg')
        font = cv2.FONT_HERSHEY_SIMPLEX
        if c == 0:
            xanh = xanh + 1
            print(count)
            print("Xanh")
        else:
            chin = chin + 1
            print(count)
            print("Chin")
        print("###############")
    #cv2.imshow('Dectection and classification Acerola based on color using Haar Cascade and KNN', frame)
    fn = time.time()
    tt = tt + (fn-st)
    count = count + 1
    
'''    if cv2.waitKey(1) == 27:
        print('Bye bye !!!')
        break'''
tt = tt/(count-1)
print(tt)
print(chin)
print(xanh)
cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()
