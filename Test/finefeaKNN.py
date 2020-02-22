import cv2
import numpy as np

path = 'D:/Study/HCMUT/HK182/DA/Images/Samples/2Color/Green/'

size = 20

count = 1
r = 0
g = 0
x = 0
y = 0
z = 0
while(count < 101):
    img = cv2.imread(path + 'G' + '-' + str(count)+ '.png')
    img = cv2.resize(img, (size, size))
    #print(img)
    red = 0
    green = 0
    blue = 0
    for i in range(len(img)):
        cell = img[i]
        for j in range(len(cell)):
            red = red + cell[j][2]
            green = green + cell[j][1]
            blue = blue + cell[j][0]
    print(blue/(size*size))
    print(red/(size*size))
    print(green/(size*size))
    x = x + blue/(size*size)
    y = y + green/(size*size)
    z = z + red/(size*size)
    print("================")
    
    count = count + 1
    
print(x/100)
print(y/100)
print(z/100)
