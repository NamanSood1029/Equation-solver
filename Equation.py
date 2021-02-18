import cv2
import numpy as np

np.random.seed(1212)
import keras
import json
import sys

my_dict = {'(': 0, ')': 1, '+': 2, '-': 3, '0': 4, '1': 5, '2': 6, '3': 7, '4': 8, '5': 9, '6': 10, '7': 11, '8': 12,
           '9': 13, 'forward_slash': 14, 'times': 15}
model_predictor = keras.models.load_model(r"weights-improvement-10-0.97.hdf5")
binary_coords = []
img = ""
n = len(sys.argv)
# print("Total:", n)

# Arguments passed
# print("\nName of Python script:", sys.argv[0])

# If no image
if n == 1:
    img = cv2.imread(r"t1.jpeg", cv2.IMREAD_GRAYSCALE)
else:
    img = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)


# Function to map keys from values: Predict function will return values and we will find the corresponding label
def get_key(val):
    for key, value in my_dict.items():
        if val == value:
            return key


# Function to remove overlapping bounding boxes
def overlap_remover(boxes, contours):
    for r in boxes:
        l = []
        for rec in boxes:
            flag = 0
            if rec != r:
                if r[0] < (rec[0] + rec[2] + 10) and rec[0] < (r[0] + r[2] + 10) and r[1] < (rec[1] + rec[3] + 10) and \
                        rec[1] < (r[1] + r[3] + 10):
                    flag = 1
                l.append(flag)
            if rec == r:
                l.append(0)
        binary_coords.append(l)
        # print(binary_coords)
    dump_rect = []
    for i in range(0, len(contours)):
        for j in range(0, len(contours)):
            # print(i, j)
            if binary_coords[i][j] == 1:
                area1 = boxes[i][2] * boxes[i][3]
                area2 = boxes[j][2] * boxes[j][3]
                if area1 == min(area1, area2):
                    dump_rect.append(boxes[i])
    # print(len(dump_rect))
    final_rect = [i for i in boxes if i not in dump_rect]
    return final_rect


def getContours(img):
    img = ~img
    ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    cv2.imshow("Thresh", thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
    bounding_boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        bb = [x, y, w, h]
        bounding_boxes.append(bb)
    bounding_boxes = overlap_remover(bounding_boxes, contours)
    s = ""
    for (x, y, w, h) in bounding_boxes:
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
        imgRoi = thresh[y:y + h, x:x + w]
        img_x = cv2.resize(imgRoi, (28, 28))
        cv2.imshow("ROI of image", img_x)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        img_x = np.resize(img_x, (1, 28, 28, 1))
        # print("Shape: ", img_x.shape)
        y_prob = model_predictor.predict(img_x)
        y_classes = y_prob.argmax(axis=-1)
        result = get_key(y_classes)
        if result == 'forward_slash':
            s = s + '/'
        elif result == 'times':
            s = s + '*'
        else:
            s = s + str(get_key(y_classes))
        cv2.putText(img_copy, get_key(y_classes), (x, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 0), 2)
    print("Equation is: ", s)
    try:
        print("Result of the equation: ", eval(s))
        eq = {
            'equation': s,
            'answer': eval(s)
        }
        with open('solution.json', 'w') as file:
            json.dump(eq, file)
    except:
        print("This string cannot be evaluated")
    cv2.imshow("Final image", img_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# IF no image
img_copy = img.copy()
# img_copy = img.copy()
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img_blur = cv2.GaussianBlur(img_gray, (3, 3), 1)
# kernel = np.ones((7, 7), np.uint8)
# img_blur = cv2.erode(img_blur, kernel)
# img_can = cv2.Canny(img_blur, 100, 100)
cv2.imshow("Gray Scale", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
getContours(img)
