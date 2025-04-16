import cv2 as cv
import numpy as np
import skimage.morphology

mp4 = cv.VideoCapture('mouse_video.mp4')
print(mp4.get(5))
if not mp4.isOpened():
    print("Video 0")
for i in range(1, 7):
    ret, frame = mp4.read()
    if not ret:
        print('video end')
        break
    mask = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cv.imwrite(f'frame_{i}.png', mask)
mp4.release()
cv.destroyAllWindows()
