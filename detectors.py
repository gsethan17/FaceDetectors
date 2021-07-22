import cv2
import cv2.dnn
import dlib
import time
from mtcnn import MTCNN

class face_detector() :
    def __init__(self, mode, THRESHOLD = 0.0):
        self.mode = mode
        self.threshold = THRESHOLD
        self.detector = self.load_detector()
        self.st_time = 0
        self.duration = 0
        self.rectangles = {}
        self.rectangles['count'] = 0
        self.rectangles['lefttop'] = []
        self.rectangles['rightbottom'] = []
        self.rectangles['confidence'] = []

    def reset(self):
        self.rectangles = {}
        self.rectangles['count'] = 0
        self.rectangles['lefttop'] = []
        self.rectangles['rightbottom'] = []
        self.rectangles['confidence'] = []
        self.st_time = 0
        self.duration = 0

    def load_detector(self) :
        if self.mode == 'Haar' :
            PRETRAIND_PATH = './src/haarcascades/haarcascade_frontalface_default.xml'
            detector = cv2.CascadeClassifier(PRETRAIND_PATH)

        elif self.mode == 'opencv_cnn' :
            model_path = './src/opencv_cnn/opencv_face_detector_uint8.pb'
            config_path = './src/opencv_cnn/opencv_face_detector.pbtxt'

            detector = cv2.dnn.readNetFromTensorflow(model_path, config_path)
            detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        elif self.mode == 'dlib' :
            detector = dlib.get_frontal_face_detector()

        elif self.mode == 'mmod' :
            config_path = './src/mmod/mmod_human_face_detector.dat'
            detector = dlib.cnn_face_detection_model_v1(config_path)

        elif self.mode == 'mtcnn' :
            detector = MTCNN()

        else :
            print('mode is not valid.')
            return -1

        return detector

    def get_detection(self, img):
        self.reset()
        self.st_time = time.time()

        if self.mode == 'Haar' :
            detections = self.detector.detectMultiScale(img)

            for x, y, w, h in detections :
                self.rectangles['count'] += 1
                self.rectangles['lefttop'].append((x, y))
                self.rectangles['rightbottom'].append((x+w, y+h))

        elif self.mode == 'opencv_cnn' :
            blob = cv2.dnn.blobFromImage(img, size=(300,300))
            self.detector.setInput(blob)
            detections = self.detector.forward()

            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > self.threshold:
                    self.rectangles['count'] += 1
                    self.rectangles['lefttop'].append(
                        (int(detections[0, 0, i, 3] * img.shape[1]), int(detections[0, 0, i, 4] * img.shape[0]))
                    )
                    self.rectangles['rightbottom'].append(
                        (int(detections[0, 0, i, 5] * img.shape[1]), int(detections[0, 0, i, 6] * img.shape[0]))
                    )
                    self.rectangles['confidence'].append(confidence)

        elif self.mode == 'dlib' :
            detections = self.detector(img)

            for detection in detections :
                self.rectangles['count'] += 1
                self.rectangles['lefttop'].append((detection.left(), detection.top()))
                self.rectangles['rightbottom'].append((detection.right(), detection.bottom()))

        elif self.mode == 'mmod' :
            detections = self.detector(img, 0)

            for detection in detections :
                confidence = detection.confidence
                if confidence > self.threshold:
                    self.rectangles['count'] += 1
                    self.rectangles['lefttop'].append((detection.rect.left(), detection.rect.top()))
                    self.rectangles['rightbottom'].append((detection.rect.right(), detection.rect.bottom()))
                    self.rectangles['confidence'].append(confidence)

        elif self.mode == 'mtcnn' :
            detections = self.detector.detect_faces(img)

            for detection in detections :
                confidence = detection['confidence']
                if confidence > self.threshold:
                    x, y, w, h = detection['box']
                    self.rectangles['count'] += 1
                    self.rectangles['lefttop'].append((x,y))
                    self.rectangles['rightbottom'].append((x+w, y+h))
                    self.rectangles['confidence'].append(confidence)


        duration = time.time() - self.st_time
        return self.rectangles, duration


def draw_detection(img, rectangles, color = (255, 255, 255), stroke = 2, font = cv2.FONT_HERSHEY_SIMPLEX, show=False) :
    for i in range(rectangles['count']) :
        cv2.rectangle(img, rectangles['lefttop'][i], rectangles['rightbottom'][i], color, stroke, cv2.LINE_AA)
        if rectangles['confidence'] :
            cv2.putText(img, '%.2f%%' % (rectangles['confidence'][i] * 100), (rectangles['lefttop'][i][0], rectangles['lefttop'][i][1] - 10), font, 1, color, stroke, cv2.LINE_AA)

    if show :
        cv2.imshow('aa', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return img



if __name__ == '__main__' :
    # THRESHOLD = 0.9
    # MODE = 'Haar'
    # MODE = 'opencv_cnn'
    MODE = 'mtcnn'

    face_detector = face_detector(MODE)

    img_bgr = cv2.imread('./images/sample.jpg')
    img = img_bgr.copy()
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    rectangles, duration = face_detector.get_detection(img)
    print(duration)

    draw_detection(img_bgr, rectangles, show=True)
