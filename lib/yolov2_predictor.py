import chainer
from chainer import serializers
from lib.yolov2_model import *
import cupy as xp

class Predictor:
    def __init__(self, __path, __labels, __thresh):
        # hyper parameters
        weight_file = __path
        self.n_classes = len(__labels)
        self.n_boxes = 5
        self.detection_thresh = __thresh 
        self.iou_thresh = 0.5
        self.labels = __labels
        anchors = [[0.738768, 0.874946], [2.42204, 2.65704], [4.30971, 7.04493], [10.246, 4.59428], [12.6868, 11.8741]]

        # load model
        print("loading  model...")
        yolov2 = YOLOv2(n_classes=self.n_classes, n_boxes=self.n_boxes)
        serializers.load_hdf5(weight_file, yolov2)  # load saved model
        model = YOLOv2Predictor(yolov2)
        model.init_anchor(anchors)
        model.predictor.train = False
        model.predictor.finetune = False
        model.to_gpu(0)
        self.model = model

    def __call__(self, orig_img):
        orig_input_height, orig_input_width, _ = orig_img.shape
        img = reshape_to_yolo_size(orig_img)
        input_height, input_width, _ = img.shape
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = xp.asarray(img, dtype=xp.float32) / 255.0
        img = img.transpose(2, 0, 1)

        # forward
        x_data = img[xp.newaxis, :, :, :]
        x = Variable(x_data)
        x, y, w, h, conf, prob = self.model.predict(x)

        # parse results
        _, _, _, grid_h, grid_w = x.shape
        x = F.reshape(x, (self.n_boxes, grid_h, grid_w)).data
        y = F.reshape(y, (self.n_boxes, grid_h, grid_w)).data
        w = F.reshape(w, (self.n_boxes, grid_h, grid_w)).data
        h = F.reshape(h, (self.n_boxes, grid_h, grid_w)).data
        conf = F.reshape(conf, (self.n_boxes, grid_h, grid_w)).data
        prob = F.transpose(F.reshape(prob, (self.n_boxes, self.n_classes, grid_h, grid_w)), (1, 0, 2, 3)).data
        detected_indices = (conf * prob).max(axis=0) > self.detection_thresh

        conf_cpu = chainer.cuda.to_cpu(conf)
        prob_cpu = chainer.cuda.to_cpu(prob)
        detected_indices_cpu = chainer.cuda.to_cpu(detected_indices)

        results = []
        for i in range(detected_indices_cpu.sum()):
            prob_transpose = prob_cpu.transpose(1, 2, 3, 0)
            results.append({
                "class_id": prob_transpose[detected_indices_cpu][i].argmax(),
                "label": self.labels[prob_transpose[detected_indices_cpu][i].argmax()],
                "probs": prob_transpose[detected_indices_cpu][i],
                "conf": conf_cpu[detected_indices_cpu][i],
                "objectness": conf[detected_indices_cpu][i] * prob_transpose[detected_indices_cpu][i].max(),
                "box": Box(
                    x[detected_indices_cpu][i] * orig_input_width,
                    y[detected_indices_cpu][i] * orig_input_height,
                    w[detected_indices_cpu][i] * orig_input_width,
                    h[detected_indices_cpu][i] * orig_input_height).crop_region(orig_input_height, orig_input_width)
            })

        # nms
        nms_results = nms(results, self.iou_thresh)
        return nms_results
