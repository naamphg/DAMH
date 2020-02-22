import cv2
import time

st = time.time()
soridetect = cv2.CascadeClassifier("cascade1.xml")

img = cv2.imread("3.jpg")
img = cv2.resize(img, (640, 480))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
sori = soridetect.detectMultiScale(
gray,
scaleFactor  = 1.1,
minNeighbors = 10,
)
for (x, y, w, h) in sori:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
fn = time.time()
print(st)
print(fn)
cv2.imshow("test", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
