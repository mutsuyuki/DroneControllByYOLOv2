import argparse
import cv2
from lib.yolov2_predictor import Predictor

# argument parse
parser = argparse.ArgumentParser(description="指定したパスの画像を読み込み、bbox及びクラスの予測を行う")
parser.add_argument('model_path',default="",help="重さファイルへのパスを指定")
parser.add_argument('--thresh',default=0.3, required=False,help="")
args = parser.parse_args()

labels = []
if "coco" in args.model_path :
    labels = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"] 
    
if "hand" in args.model_path:
    labels = ["ok", "five"] 

if len(labels) == 0:
    print("You can use only coco weights or hand weights")

# main process ---------
cap = cv2.VideoCapture(0)
predictor = Predictor(args.model_path, labels, args.thresh)

while (True):

    ret, orig_img = cap.read()
    orig_img = cv2.resize(orig_img,(256,192))
    nms_results = predictor(orig_img)

    # draw result
    for result in nms_results:
        left, top = result["box"].int_left_top()
        right, bottom = result["box"].int_right_bottom()
        cv2.rectangle(orig_img, (left, top), (right, bottom), (255, 0, 255), 1)
        text = '%s(%2d%%)' % (result["label"], result["probs"].max() * result["conf"] * 100)
        cv2.putText(orig_img, text, (left, top - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        print(text)

    window_name = "w"
    cv2.imshow(window_name, cv2.resize(orig_img,(256*2, 192*2)))
    cv2.moveWindow(window_name, 150, 150)
    cv2.waitKey(1)
