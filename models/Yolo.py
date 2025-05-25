import os
import tempfile
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO


SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
ITER_GRABCUT = 5

data_yaml_path    = PROJECT_ROOT / 'config' / 'yolo.yaml'
cfg               = yaml.safe_load(open(data_yaml_path))
dataset_config    = cfg['dataset_params']
train_config      = cfg['train_params']

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')


def train_custom_yolo():
    print(f"Using device: {DEVICE}")
    if not data_yaml_path.exists():
        print(f"ERROR: Data YAML file not found at '{data_yaml_path}'.")
        return

    data_subset = {
        'path':  str(PROJECT_ROOT / dataset_config['path']),
        'train': dataset_config['train'],
        'val':   dataset_config['val'],
        'nc':    dataset_config['nc'],
        'names': dataset_config['names']
    }
    with tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False) as tf:
        yaml.safe_dump(data_subset, tf)
        temp_data_yaml = tf.name

    try:
        model = YOLO(train_config['model_variant'])
        model.model.to(DEVICE)
    except Exception as e:
        print(f"Error loading model {train_config['model_variant']}: {e}")
        return

    try:
        model.train(
            data=temp_data_yaml,
            epochs=train_config['epochs'],
            imgsz=train_config['img_size'],
            batch=train_config['batch_size'],
            project=str(PROJECT_ROOT / dataset_config['project_name']),
            name=dataset_config['experiment_name'],
            device=DEVICE,
            exist_ok=train_config['exist_ok'],
            val=train_config['val'],


            # --- augmentation & preprocessing ---
            augment=train_config['augment'],
            auto_augment=train_config['auto_augment'],
            mosaic=train_config['mosaic'],
            mixup=train_config['mixup'],
            copy_paste=train_config['copy_paste'],
            erasing=train_config['erasing'],
            hsv_h=train_config['hsv_h'],
            hsv_s=train_config['hsv_s'],
            hsv_v=train_config['hsv_v'],
            fliplr=train_config['fliplr'],
            flipud=train_config['flipud'],
            degrees=train_config['degrees'],
            translate=train_config['translate'],
            scale=train_config['scale'],
            shear=train_config['shear'],
            perspective=train_config['perspective'],
            close_mosaic=train_config['close_mosaic'])

        print("Training completed.")
    except Exception as e:
        print(f"An error occurred during training: {e}")
        traceback.print_exc()


def inference(trained_model_path_str, image_to_detect_path_str, confidence_threshold=0.25):
    trained_model_path   = Path(trained_model_path_str)
    image_to_detect_path = Path(image_to_detect_path_str)

    if not trained_model_path.exists():
        print(f"Error: Model not found at {trained_model_path}")
        return
    if not image_to_detect_path.exists():
        print(f"Error: Source not found at {image_to_detect_path}")
        return


    print(f"Loaded YOLO model from {trained_model_path} on {DEVICE}")

    # Gather image files
    if image_to_detect_path.is_dir():
        sources = []
        for p in image_to_detect_path.iterdir():
            if p.suffix.lower() in ('.jpg', '.png', '.jpeg'):
                sources.append(str(p))
        sources = sorted(sources)

    else:
        sources = [str(image_to_detect_path)]

    base_out = PROJECT_ROOT / dataset_config['project_name'] / dataset_config['inference_output']
    base_out.mkdir(parents=True, exist_ok=True)

    for src in sources:
        print(f"\nProcessing {src}")
        img = cv2.imread(src)
        h, w = img.shape[:2]

        results = model.predict(source=img, conf=confidence_threshold, device=DEVICE)

        # Mask konstravimas per GrabcUT
        full_mask = np.zeros((h, w), dtype=np.uint8)
        for r in results:
            for box, cls_idx, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
                x1, y1, x2, y2 = map(int, box.cpu().numpy())
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)
                roi = img[y1:y2, x1:x2]
                if roi.size == 0:
                    continue

                mask_gc  = np.zeros(roi.shape[:2], np.uint8)
                bgdModel = np.zeros((1, 65), np.float64)
                fgdModel = np.zeros((1, 65), np.float64)
                rect     = (1, 1, roi.shape[1] - 2, roi.shape[0] - 2)

                cv2.grabCut(roi, mask_gc, rect,
                            bgdModel, fgdModel,
                            ITER_GRABCUT, cv2.GC_INIT_WITH_RECT)

                mask_fg = np.where(
                    (mask_gc == cv2.GC_FGD) | (mask_gc == cv2.GC_PR_FGD),
                    255, 0
                ).astype('uint8')

                full_mask[y1:y2, x1:x2] = np.maximum(
                    full_mask[y1:y2, x1:x2], mask_fg
                )

        # Piešiama prediction bbox ir label tekstus
        vis = img.copy()
        for r in results:
            for box, cls_idx, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
                x1, y1, x2, y2 = map(int, box.cpu().numpy())
                name = model.names[int(cls_idx)]
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(vis, f"{name} {conf:.2f}",
                            (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        # Uždedama maską ant prediction
        color_mask = cv2.merge([
            full_mask,
            np.zeros_like(full_mask),
            np.zeros_like(full_mask)
        ])
        overlay = cv2.addWeighted(vis, 1.0, color_mask, 0.5, 0)

        stem, ext = Path(src).stem, Path(src).suffix
        cv2.imwrite(str(base_out / f"{stem}_pred{ext}"),      vis)
        cv2.imwrite(str(base_out / f"{stem}_overlay{ext}"),  overlay)

        print(f"Saved to {base_out}:")
        print(f"  • {stem}_pred{ext}")
        print(f"  • {stem}_overlay{ext}")


def inference(trained_model_path_str, image_to_detect_path_str,
              confidence_threshold=0.25):
    trained_model_path = Path(trained_model_path_str)
    image_to_detect_path = Path(image_to_detect_path_str)

    if not trained_model_path.exists():
        print(f"Error: Model not found at {trained_model_path}")
        return
    if not image_to_detect_path.exists():
        print(f"Error: Source not found at {image_to_detect_path}")
        return

    model = YOLO(str(trained_model_path))
    model.model.to(DEVICE)
    print(f"Loaded YOLO model from {trained_model_path} on {DEVICE}")

    base_out = PROJECT_ROOT / dataset_config['project_name'] / dataset_config['inference_output']
    base_out.mkdir(parents=True, exist_ok=True)

    if image_to_detect_path.is_dir():
        sources = sorted(
            str(p) for p in image_to_detect_path.iterdir()
            if p.suffix.lower() in ('.jpg', '.png', '.jpeg')
        )
    else:
        sources = [str(image_to_detect_path)]

    for src in sources:
        print(f"\nProcessing {src}")
        img = cv2.imread(src)
        h, w = img.shape[:2]

        results = model.predict(source=img, conf=confidence_threshold, device=DEVICE)

        full_mask = np.zeros((h, w), dtype=np.uint8)
        for r in results:
            for box in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
                (x1, y1, x2, y2), cls_idx, conf = box
                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                # clamp
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)
                roi = img[y1:y2, x1:x2]
                if roi.size == 0:
                    continue

                mask_gc = np.zeros(roi.shape[:2], np.uint8)
                bgdModel = np.zeros((1, 65), np.float64)
                fgdModel = np.zeros((1, 65), np.float64)
                rect = (1, 1, roi.shape[1] - 2, roi.shape[0] - 2)

                cv2.grabCut(roi, mask_gc, rect,
                            bgdModel, fgdModel,
                            ITER_GRABCUT, cv2.GC_INIT_WITH_RECT)

                mask_fg = np.where(
                    (mask_gc == cv2.GC_FGD) | (mask_gc == cv2.GC_PR_FGD),
                    255, 0
                ).astype('uint8')

                full_mask[y1:y2, x1:x2] = np.maximum(
                    full_mask[y1:y2, x1:x2], mask_fg
                )

        vis_pred = img.copy()
        for r in results:
            for box_tensor, cls_idx, conf in zip(
                    r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
                x1, y1, x2, y2 = map(int, box_tensor.cpu().numpy())
                name = model.names[int(cls_idx)]
                cv2.rectangle(vis_pred, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(vis_pred, f"{name} {float(conf):.2f}",
                            (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        color_mask = cv2.merge([
            full_mask,
            np.zeros_like(full_mask),
            np.zeros_like(full_mask)
        ])
        overlay = cv2.addWeighted(vis_pred, 1.0, color_mask, 0.5, 0)

        vis_gt = img.copy()
        label_file = Path(dataset_config['path']) / Path(dataset_config['val_labels']) / Path(src).with_suffix('.txt').name
        print(f"Ground-truth label file: {label_file}")
        if label_file.exists():
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls, x_c, y_c, w_rel, h_rel = map(float, parts)
                    x_center = x_c * w
                    y_center = y_c * h
                    bw = w_rel * w
                    bh = h_rel * h
                    x1 = int(x_center - bw/2)
                    y1 = int(y_center - bh/2)
                    x2 = int(x_center + bw/2)
                    y2 = int(y_center + bh/2)
                    cv2.rectangle(vis_gt, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    label = model.names[int(cls)] if int(cls) < len(model.names) else str(int(cls))
                    cv2.putText(vis_gt, label,
                                (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        else:
            print(f"Ground-truth label failas nerastas: {label_file}")

        # --- 6) Save outputs ---
        stem, ext = Path(src).stem, Path(src).suffix
        cv2.imwrite(str(base_out / f"{stem}_pred{ext}"), vis_pred)
        cv2.imwrite(str(base_out / f"{stem}_overlay{ext}"), overlay)
        cv2.imwrite(str(base_out / f"{stem}_gt{ext}"), vis_gt)

        print(f"Išsaugomas į {base_out}:")
        print(f"  • {stem}_pred{ext}")
        print(f"  • {stem}_overlay{ext}")
        print(f"  • {stem}_gt{ext}")

def run_inference_yolo(confidence_threshold=0.2):
    best_pt = PROJECT_ROOT / dataset_config['project_name'] / dataset_config['experiment_name'] / 'weights' / 'best.pt'
    img_src = PROJECT_ROOT / 'dataset' / 'test' / 'images'
    inference(str(best_pt), str(img_src), confidence_threshold=confidence_threshold)

