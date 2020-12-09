import cv2 
import os

def make_video():
    EPISODES = 70_000
    SAVES_EP = 500
    fourcc = cv2.cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.cv2.VideoWriter('qlearn.mp4', fourcc, 20, (1200, 900))

    for i in range(0, EPISODES+1, SAVES_EP):
        img_path = f"charts/try{i}.png"
        frame = cv2.cv2.imread(img_path)
        out.write(frame)
    
    out.release()

make_video()