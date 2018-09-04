import argparse
import cv2
from lib.yolov2_predictor import Predictor

# argument parse
parser = argparse.ArgumentParser(description="重さファイル")
parser.add_argument('model_path', help="重さファイルへのパスを指定")
args = parser.parse_args()

cap = cv2.VideoCapture(0)
coco_predictor = Predictor(args.model_path, ["ok", "five"],0.2 )

is_inner = False
is_inner_prev = False

gesture_box = [
    int(640 / 2 - 70),  # left
    int(640 / 2 + 70),  # right
    int(480 / 2 - 70),  # top
    int(480 / 2 + 70)  # bottom
]

max_prob_point = 0
max_prob_label = ""


is_takeoff = False
takeoff_counter = 0
land_counter = 0


# tello prepare ------
import threading
import socket
from time import sleep

host = ''
port = 9000
locaddr = (host,port)

# Create a UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
tello_address = ('192.168.10.1', 8889)
sock.bind(locaddr)

def recv():
    count = 0
    while True:
        try:
            data, server = sock.recvfrom(1518)
            print(data.decode(encoding="utf-8"))
        except Exception:
            print ('\nExit . . .\n')
            break

#recvThread create
recvThread = threading.Thread(target=recv)
recvThread.start()

sleep(1)

sent = sock.sendto("command".encode(encoding="utf-8"), tello_address)

# tello prepare end ------

while (True):
    ret, orig_img = cap.read()
    nms_results = coco_predictor(orig_img)

    cv2.rectangle(orig_img, (gesture_box[0], gesture_box[2]), (gesture_box[1], gesture_box[3]), (230, 230, 230), 1)

    max_prob = 0
    max_prob_label = ""

    # draw result & select max probs
    for result in nms_results:
        left, top = result["box"].int_left_top()
        right, bottom = result["box"].int_right_bottom()
        cv2.rectangle(orig_img, (left, top), (right, bottom), (255, 255, 100), 2)
        prob = result["probs"].max() * result["conf"] * 100
        text = '%s(%2d%%)' % (result["label"], prob)
        cv2.putText(orig_img, text, (left, top - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if max_prob < prob:
            max_prob = prob

            max_prob_point = (int(left + (right - left) / 2), int(top + (bottom - top) / 2))
            max_prob_label = result["label"]
            cv2.circle(orig_img, max_prob_point, 10, (255, 0, 0))

    is_inner = False
    if max_prob_label == "five":
        is_in_x = gesture_box[0] < max_prob_point[0] and max_prob_point[0] < gesture_box[1]
        is_in_y = gesture_box[2] < max_prob_point[1] and max_prob_point[1] < gesture_box[3]
        is_inner = is_in_x and is_in_y

    # takeoff check
    if not(is_takeoff) and is_inner:
        takeoff_counter += 1
        if(takeoff_counter > 20):
            takeoff_counter = 0
            sent = sock.sendto("takeoff".encode(encoding="utf-8"), tello_address)
            is_takeoff = True
            print("take off")

    # flip check
    if is_takeoff and max_prob_label == "five" and not(is_inner) and is_inner_prev:
        if max_prob_point[0] > gesture_box[1]:
            sent = sock.sendto("flip l".encode(encoding="utf-8"), tello_address)
            print("filp left")

        if max_prob_point[0] < gesture_box[0]:
            sent = sock.sendto("flip r".encode(encoding="utf-8"), tello_address)
            print("filp right")

        if max_prob_point[1] < gesture_box[2]:
            sent = sock.sendto("flip b".encode(encoding="utf-8"), tello_address)
            print("filp top")

        if max_prob_point[1] > gesture_box[3]:
            sent = sock.sendto("flip b".encode(encoding="utf-8"), tello_address)
            # sent = sock.sendto("land".encode(encoding="utf-8") , tello_address)
            print("filp bottom")

    is_inner_prev = is_inner

    # land check
    if is_takeoff:
        if max_prob_label != "five" and max_prob != 0:
            land_counter += 1
        else:
            land_counter -= 1
            if land_counter < 0:
                land_counter = 0

        if land_counter > 40:
            print("land")
            sent = sock.sendto("land".encode(encoding="utf-8") , tello_address)
            land_counter = 0
            takeoff_counter = 0
            is_takeoff = False

    window_name = "w"
    cv2.imshow(window_name, cv2.resize(orig_img, (600, 400)))
    cv2.moveWindow(window_name, 1000, 500)
    cv2.waitKey(1)
