import glob
import os
import cv2
import matplotlib.pyplot as plt
from detectors import face_detector, draw_detection

IMAGE_PATH = '/home/gsethan/Documents/DMS/FrontView_IR_Samples/test_images'

MODE = 'mtcnn' # opencv_cnn, mmod, mtcnn
THRESHOLD = 0.1
AP_RATE = 0.95    # 0.95, 0.75, 0.50

def get_gt(image_path, shape) :
    name = os.path.basename(image_path).split('.')[0]
    gt_path = os.path.join(os.path.dirname(image_path), name+'.txt')

    f = open(gt_path, 'r')
    line = f.readline()
    context = line.split(' ')
    x = float(context[1])
    y = float(context[2])
    w = float(context[3])
    h = float(context[4].split('\\')[0])
    f.close()

    x_pt = int(shape[1] * x)
    y_pt = int(shape[0] * y)
    w_pt = int((shape[1] * w) / 2)
    h_pt = int((shape[0] * h) / 2)

    return x_pt, y_pt, w_pt, h_pt

def get_iou(gt, detections) :
    ious = {}
    ious['count'] = 0
    ious['value'] = []
    ious['lefttop'] = []
    ious['rightbottom'] = []
    ious['confidence'] = []

    detections['results'] = []

    for i in range(detections['count']) :
        # top_edge = min([gt['lefttop'][0][1], detections['lefttop'][i][1]])
        top_inside = max([gt['lefttop'][0][1], detections['lefttop'][i][1]])

        # left_edge = min([gt['lefttop'][0][0], detections['lefttop'][i][0]])
        left_inside = max([gt['lefttop'][0][0], detections['lefttop'][i][0]])

        # bottom_edge = max([gt['rightbottom'][0][1], detections['rightbottom'][i][1]])
        bottom_inside = min([gt['rightbottom'][0][1], detections['rightbottom'][i][1]])

        # right_edge = max([gt['rightbottom'][0][0], detections['rightbottom'][i][0]])
        right_inside = min([gt['rightbottom'][0][0], detections['rightbottom'][i][0]])

        num = (bottom_inside - top_inside) * (right_inside - left_inside)
        gt_area = (gt['rightbottom'][0][1] - gt['lefttop'][0][1]) * (gt['rightbottom'][0][0] - gt['lefttop'][0][0])
        detection_area = (detections['rightbottom'][i][1] - detections['lefttop'][i][1]) * (detections['rightbottom'][i][0] - detections['lefttop'][i][0])

        det = gt_area + detection_area - num

        if num / det >= AP_RATE :
            ious['count'] += 1
            ious['value'].append(num / det)
            ious['lefttop'].append((left_inside, top_inside))
            ious['rightbottom'].append((right_inside, bottom_inside))
            detections['results'].append('TP')
        else :
            detections['results'].append('FP')

    return ious, detections

def draw_pr_curve(AP, MODE, AP_RATE) :
    recall = AP['recall']
    precision = AP['precision']
    plt.figure(figsize=(9, 6))
    plt.plot(recall, precision)
    plt.scatter(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve 2D [{}] : AP{}={:.3f}'.format(MODE, AP_RATE, AP['AP']))
    save_dir = os.path.join(os.getcwd(), 'eval_results')
    if not os.path.isdir(save_dir) :
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, 'pr_curve_{}_{}.png'.format(MODE, AP_RATE)))

if __name__ == '__main__' :
    results = []
    confidence = []

    test_images = glob.glob(os.path.join(IMAGE_PATH, '*.jpg'))

    face_detector = face_detector(MODE, THRESHOLD)

    for i, image in enumerate(test_images) :
        name = os.path.basename(image).split('.')[0]
        input_img = cv2.imread(image)

        gt = {}
        gt['count'] = 1
        gt['confidence'] = []
        gt['lefttop'] = []
        gt['rightbottom'] = []
        x_pt, y_pt, w_pt, h_pt = get_gt(image, input_img.shape)

        gt['lefttop'].append((x_pt - w_pt, y_pt - h_pt))
        gt['rightbottom'].append((x_pt + w_pt, y_pt + h_pt))

        gt_img = draw_detection(input_img, gt, color = (0, 0, 255))

        # if MODE == 'Haar' or MODE == 'dlib' :
        #     input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)

        rectangles, duration = face_detector.get_detection(input_img)
        result_img = draw_detection(gt_img, rectangles, color=(0, 255, 0))

        IOUs, result = get_iou(gt, rectangles)
        results += result['results']
        confidence += result['confidence']

        iou_img = draw_detection(result_img, IOUs, color = (255, 0, 0))

    print(MODE)
    print("Number of test images : ", i+1)
    print("AP rate : ", AP_RATE)
    print("Minimum confidence", THRESHOLD)
    confidence_sorted = sorted(confidence, reverse = True)

    TP = 0
    AP = {}
    AP['precision'] = []
    AP['recall'] = []
    AP['AP'] = 0

    flag = 0
    prev_recall = 0

    for num_detect, conf in enumerate(confidence_sorted) :

        idx = confidence.index(conf)
        TPorFP = results[idx]
        if TPorFP == 'TP' :
            TP += 1
            flag = 1

        if TPorFP == 'FP' and flag :
            cur_prec = AP['precision'][-1]
            cur_recall = AP['recall'][-1]

            if not prev_recall :
                prev_recall = AP['recall'][0]

            AP['AP'] += cur_prec * (cur_recall - prev_recall)
            prev_recall = cur_recall

        precision = TP / (num_detect+1)
        recall = TP / (i+1)

        AP['precision'].append(precision)
        AP['recall'].append(recall)

    draw_pr_curve(AP, MODE, AP_RATE)
    print("Number of True Positive : ", TP)
    print("AP : ", AP['AP'])