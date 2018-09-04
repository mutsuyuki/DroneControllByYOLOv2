# トイドローンTelloをYOLOを使って操作するDEMO

Yoloで手を認識して、手の位置をフリックするジェスチャーによって[Tello](https://store.dji.com/jp/shop/tello-series?from=menu_products)にAPIコマンドを送ります。
手のジェスチャーでTelloの離陸・着陸と、左右前後のフリップを操作できます。



![デモ](readme_media/tello_yolo.gif "デモ")



### クオリティは全体的に子供だましです。

実際に子供を喜ばせるために作ったので質は期待しちゃいけません。

Yoloのchainer版をベースにGPUを使うようにちょっぴり改造しています。

Yolo部分のオリジナルは[こちら](https://github.com/leetenki/YOLOv2)です！


## 試した環境
- Windows 10
- Python 3.6.3
- opencv-python 3.4.0.12
- Chainer 4.1.0
- CUDA V9.0



## 準備

手を認識する学習済みモデルをダウンロードしてChainer用にパースします。
```
cd weights
wget http://mtool.dip.jp/DroneControllByYOLOv2/hand_15000.weights
python weights_parser.py hand_15000.weights
```
hand_15000.weights.modelというファイルができます。

## 実行
DroneControllByYOLOv2直下で以下を実行
```
python demo_tello.py weights/hand_15000.weights.model
```

![デモ](readme_media/tello_yolo_camera.gif "デモ")


## 通常の認識もできます
カメラ
cocoのモデルで
```
cd weights
wget http://mtool.dip.jp/DroneControllByYOLOv2/yolo_coco.weights
python weights_parser.py yolo_coco.weights
cd ..
python demo_camera.py weights/yolo_coco.weights.model
```

