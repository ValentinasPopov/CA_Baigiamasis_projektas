
# Medinių paviršių defektų aptikimas (YOLOv8 / Faster R-CNN)

Šio projekto tikslas yra sukurti ir įvertinti medinių paviršių defektų aptikimo sistemą, pasitelkiant deep learning tinklus R-CNN ir Yolov8
* Paruošti ir pažymėti anotacijas medinių defektų vaizdų rinkinį kaip įtrūkimus, puvinius. 
* Išmokyti R-CNN ir Yolov8 architektūrų modelius, kad galėtų efektyviai lokalizuoti bei klasifikuoti įvairius medinių paviršių defektus.
## Projekto struktūra
```
├── config/                    
│   ├── defect_labels.yaml     
│   ├── rcnn.yaml       
│   └── yolo.yaml     
├── dataset/   
│   ├── anomaly/     
│   ├── good/       
│   ├── raw/
│   ├── test/
│   │   ├──  images/
│   │   ├──  labels/
│   │   ├──  annotations.json/
│   ├── train/
│   │   ├──  images/
│   │   ├──  labels/
│   │   ├──  annotations.json/           
├── helper/  
│   ├── load_detection_labels.py
│   ├── rcnn_label_convert.py               
├── models/         
│   ├── RCNN.py     
│   ├── Yolo.py       
│   ├── yolov8s.pt
│   ├── yolov8s-seg.pt/ 
├── scripts/         
│   ├── dataLoader_rcnn.py     
│   ├── image_dataset_splitter.py       
│   ├── inference_rcnn.py
│   ├── labeling.py
│   ├── train_rcnn.py
├── Yolo_Output/   
│   ├── run_yolov8/     
│   │   ├── best.pt
├── RCNN_Output/   
│   ├── run_rcnn/     
│   │   ├── best_r50fpn.pth

```

## Instaliavimas
``` pip install -r requirements.txt ```

## Naudojimas
1. Paleidžiama  ```python main.py```
2. Pasirenkama: 
   * Label photos ⎯ 1
   * Training YOLO ⎯ 2
   * Training RCNN ⎯ 3
   * Run inference YOLO ⎯ 4
   * Run inference RCNN ⎯ 5
   


## Rezultatai
* Yolo modelio inference išvestis: ```Yolo_Output/inference/```
* Yolo modelio weights išvestis: ```Yolo_Output/run_yolov8/weights/best.pt```
* RCNN modelio inference išvestis: ```RCNN_Output/inference```
* RCNN modelio weights išvestis: ```RCNN_Output/run_yolov8/weights/best_resnet.pth```


