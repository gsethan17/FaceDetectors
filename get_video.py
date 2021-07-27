import glob
import os
import cv2
from detectors import face_detector, draw_detection

VIDEO_PATH = '/home/gsethan/Documents/DMS/FrontView_IR_Samples/'
FOURCC = cv2.VideoWriter_fourcc(*'XVID')
COLOR = (0, 0, 0)
STROKE = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX

MODE = 'mmod' # Haar, dlib, opencv_cnn, mmod, mtcnn
THRESHOLD = 0.50

if __name__ == '__main__' :
    video_path = glob.glob(os.path.join(VIDEO_PATH, '*.avi'))

    video_file = video_path[2]
    print(video_file)

    cap = cv2.VideoCapture(video_file)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)

    save_path = os.path.join(VIDEO_PATH, os.path.basename(video_file).split('.')[0] + '_' + str(MODE) + '_' + str(THRESHOLD) + '.avi')
    out = cv2.VideoWriter(filename=save_path,
                          fourcc=FOURCC,
                          fps=fps,
                          frameSize=(int(width), int(height)))

    face_detector = face_detector(MODE, THRESHOLD)

    while cap.isOpened() :
        num_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cur_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        print('{} / {}'.format(int(cur_frame), int(num_frame)))
        ret, img = cap.read()

        if not ret :
            print('Video file is not available anymore')
            break

        input_img = img.copy()

        if MODE == 'Haar' or MODE == 'dlib' :
            input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)

        rectangles, duration = face_detector.get_detection(input_img)

        img = draw_detection(img, rectangles, COLOR, STROKE, FONT)
        cv2.putText(img, 'FPS : %.2f' % (1/duration), (10, 30), FONT, 1, COLOR, STROKE, cv2.LINE_AA)

        out.write(img)

    cap.release()
    out.release()