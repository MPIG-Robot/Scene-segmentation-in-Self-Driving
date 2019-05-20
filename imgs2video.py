import cv2
import os

img_root = r'./result'
fps = 5
size=(480, 360)
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
videoWriter = cv2.VideoWriter(r"./result.mp4", cv2.VideoWriter_fourcc('I', '4', '2', '0'), fps, size)

images = os.listdir(img_root)
for img in images:
    frame = cv2.imread(os.path.join(img_root, img))
    videoWriter.write(frame)
videoWriter.release()