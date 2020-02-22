import numpy as np
import cv2
import time

t = time.time()
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
    kq1, kq2, kq3, kq4 = knn.findNearest(sp, 10)
    return kq2
chin = 0
xanh = 0
count = 1
while (count < 351):
    c = test('D:\Study\HCMUT\HK182\DA\Images\DetectedG\Object{}.png'.format(count))
    if c == 0:
        xanh = xanh + 1
        print(count)
        print("Xanh")
    else:
        chin = chin + 1
        print(count)
        print("Chin")
    print("###############")
    count = count + 1
s = time.time()
tt = s-t
print(tt)
print(chin)
print(xanh)
