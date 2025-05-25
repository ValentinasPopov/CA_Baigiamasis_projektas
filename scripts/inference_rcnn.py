import argparse
import os
import random
from pathlib import Path
import yaml

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from sklearn.metrics import ConfusionMatrixDisplay
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from models.RCNN import FasterRCNN
from scripts.dataLoader_rcnn import RCNNDataset



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_iou(det, gt):
    det_x1, det_y1, det_x2, det_y2 = det
    gt_x1, gt_y1, gt_x2, gt_y2 = gt

    # Apskaičiuojama IoU
    # kairysis taškas – didesnė x reikšmė, viršutinis – didesnė y
    x_left = max(det_x1, gt_x1)
    y_top = max(det_y1, gt_y1)
    x_right = min(det_x2, gt_x2)
    y_bottom = min(det_y2, gt_y2)

    # Jei susikirtimo nėra, IoU = 0
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # Susikirtimo srities plotas
    area_intersection = (x_right - x_left) * (y_bottom - y_top)

    # Atitinkamai aptikimo ir ground-truth plotai
    det_area = (det_x2 - det_x1) * (det_y2 - det_y1)
    gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)

    area_union = float(det_area + gt_area - area_intersection + 1E-6)

    # IoU – susikirtimo dalis iš jungtinio ploto
    iou = area_intersection / area_union
    return iou


def compute_map(det_boxes, gt_boxes, iou_threshold=0.4, method='area'):

    # Surenkame visas klases, kurios pasitaiko ground-truth duomenyse
    gt_labels = {cls_key for im_gt in gt_boxes for cls_key in im_gt.keys()}
    gt_labels = sorted(gt_labels)

    all_aps = {}
    aps = []

    # Einame per kiekvieną klasę atskirai
    for label in gt_labels:
        # Surenkame visus detections šiai klasei:
        # kiekvienas įrašas: [vaizdo indeksas, vienas aptikimas su score pabaigoje]
        cls_dets = [
            [im_idx, det]
            for im_idx, im_dets in enumerate(det_boxes)
            if label in im_dets
            for det in im_dets[label]
        ]
        # Rūšiuojame pagal score mažėjimo tvarka
        cls_dets = sorted(cls_dets, key=lambda x: -x[1][-1])
        gt_matched = [
            [False] * len(im_gt.get(label, []))
            for im_gt in gt_boxes
        ]
        # Suskaičiuojame visų GT dėžučių skaičių
        num_gts = sum(len(im_gt.get(label, [])) for im_gt in gt_boxes)

        # Paruošiama sąrašus TP ir FP
        tp = [0] * len(cls_dets)
        fp = [0] * len(cls_dets)

        # Einame per kiekvieną spėjimą ir žiūrime, ar jis TP ar FP
        for det_idx, (im_idx, det_pred) in enumerate(cls_dets):
            im_gts = gt_boxes[im_idx].get(label, [])
            max_iou = -1
            max_iou_gt_idx = -1

            # Ieškome geriausiai sutampančios GT dėžutės pagal IoU
            for gt_idx, gt_box in enumerate(im_gts):
                iou = get_iou(det_pred[:-1], gt_box)
                if iou > max_iou:
                    max_iou = iou
                    max_iou_gt_idx = gt_idx

            # Jeigu IoU per mažas arba GT jau atitiktas → FP, kitu atveju → TP
            if max_iou < iou_threshold or gt_matched[im_idx][max_iou_gt_idx]:
                fp[det_idx] = 1
            else:
                tp[det_idx] = 1
                gt_matched[im_idx][max_iou_gt_idx] = True

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)

        eps = np.finfo(np.float32).eps
        recalls = tp_cum / max(num_gts, eps)
        precisions = tp_cum / np.maximum(tp_cum + fp_cum, eps)

        # 5. Apskaičiuojame AP
        if method == 'area':
            recalls = np.concatenate(([0.0], recalls, [1.0]))
            precisions = np.concatenate(([0.0], precisions, [0.0]))
            for i in range(precisions.size - 1, 0, -1):
                precisions[i-1] = max(precisions[i-1], precisions[i])
            changes = np.where(recalls[1:] != recalls[:-1])[0]
            ap = np.sum((recalls[changes+1] - recalls[changes]) * precisions[changes+1])

        elif method == 'interp':
            ap = 0.0
            for r in np.arange(0, 1 + 1e-3, 0.1):
                # randame max precision, kur recall >= r
                p = precisions[recalls >= r]
                ap += p.max() if p.size > 0 else 0.0
            ap /= 11.0

        else:
            raise ValueError("method gali būti tik 'area' arba 'interp'")

        # Jeigu yra bent vienas GT → saugoma AP, kitu atveju NaN
        if num_gts > 0:
            aps.append(ap)
            all_aps[label] = ap
        else:
            all_aps[label] = np.nan

    # mAP – vidurkis per visas klases
    mean_ap = sum(aps) / len(aps) if aps else 0.0
    return mean_ap, all_aps


def load_model_and_dataset(args):
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)

    dataset_config = config['dataset_params']
    model_config = config['model_params']
    train_config = config['train_params']
    infer_config = config['inference_params']

    seed = train_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)

    rcnn_data = RCNNDataset(
        split='test',
        im_dir=dataset_config['im_test_path'],
        annotation_json_path=dataset_config['ann_test_path']
    )



    test_dataset = DataLoader(rcnn_data, batch_size=1, shuffle=False)

    if args.use_resnet50_fpn:
        faster_rcnn_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True,
                                                                                 min_size=model_config['min_im_size'],
                                                                                 max_size=model_config['max_im_size'],
                                                                                 box_score_thresh=infer_config['box_score_thresh'],
                                                                                 )
        faster_rcnn_model.roi_heads.box_predictor = FastRCNNPredictor(
            faster_rcnn_model.roi_heads.box_predictor.cls_score.in_features,
            num_classes=dataset_config['num_classes'])
    else:
        backbone = torchvision.models.resnet34(pretrained=True, norm_layer=torchvision.ops.FrozenBatchNorm2d)
        backbone = torch.nn.Sequential(*list(backbone.children())[:-3])
        backbone.out_channels = 256
        roi_align = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)
        rpn_anchor_generator = AnchorGenerator()
        faster_rcnn_model = torchvision.models.detection.FasterRCNN(backbone,
                                                                    num_classes=dataset_config['num_classes'],
                                                                    min_size=model_config['min_im_size'],
                                                                    max_size=model_config['max_im_size'],
                                                                    rpn_anchor_generator=rpn_anchor_generator,
                                                                    rpn_pre_nms_top_n_train=model_config[
                                                                        'rpn_pre_nms_top_n_train'],
                                                                    rpn_pre_nms_top_n_test=model_config[
                                                                        'rpn_pre_nms_top_n_test'],
                                                                    box_batch_size_per_image=model_config[
                                                                        'box_batch_size_per_image'],
                                                                    rpn_post_nms_top_n_test=model_config[
                                                                        'rpn_post_nms_top_n_test'],
                                                                    box_score_thresh=infer_config['box_score_thresh'])

    faster_rcnn_model.eval()
    faster_rcnn_model.to(device)
    if args.use_resnet50_fpn:
        faster_rcnn_model.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                                  'weight_frcnn_resnet_' + train_config['ckpt_name']),
                                                     map_location=device))
    else:
        faster_rcnn_model.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                                  'weight_frcnn_' + train_config['ckpt_name']),
                                                     map_location=device))
    return faster_rcnn_model, rcnn_data, test_dataset


def infer(args):

    # Pasirenkame katalogo pavadinimą pagal modelio tipą
    suffix = "resnet_" if args.use_resnet50_fpn else ""
    output_dir = f"tv_frcnn_{suffix}"
    os.makedirs(output_dir, exist_ok=True)

    # Užkrauname modelį ir duomenų rinkinį
    model, dataset, _ = load_model_and_dataset(args)
    model.to(device).eval()

    for _ in tqdm(range(10), desc="Inference"):
        # Pasirenkame atsitiktinį nuotrauką
        idx = random.randrange(len(dataset))
        img_tensor, target, fname = dataset[idx]
        with torch.no_grad():
            output = model([img_tensor.to(device)])[0]

        # Paruošiama nuotraukų vizualizacija
        orig = cv2.imread(fname)
        if orig is None:
            raise RuntimeError(f"Nepavyko įkelti paveikslo {fname}")
        pred_vis = orig.copy()
        gt_vis   = orig.copy()

        # Nubrėžiama predictions (raudonos dėžutės)
        boxes  = output['boxes'].cpu().numpy()
        labels = output['labels'].cpu().numpy()
        scores = output['scores'].cpu().numpy()
        for (x1,y1,x2,y2), lbl, sc in zip(boxes, labels, scores):
            if sc < 0.05:
                continue
            x1,y1,x2,y2 = map(int, (x1,y1,x2,y2))
            cv2.rectangle(pred_vis, (x1,y1), (x2,y2), (0,0,255), 2)
            text = f"{dataset.idx2label[lbl]}:{sc:.2f}"
            cv2.putText(pred_vis, text, (x1, y1-5),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)

        # Nubrėžiame ground-truth (žalios dėžutės)
        for box, lbl in zip(target['bboxes'], target['labels']):
            x1,y1,x2,y2 = box.int().cpu().numpy()
            cv2.rectangle(gt_vis, (x1,y1), (x2,y2), (0,255,0), 2)
            label_text = dataset.idx2label[int(lbl)]
            tw, th = cv2.getTextSize(label_text,
                                     cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
            cv2.rectangle(gt_vis,
                          (x1, y1-th-4),
                          (x1+tw+4, y1),
                          (255,255,255), -1)
            cv2.putText(gt_vis, label_text,
                        (x1+2, y1-2),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)

        # Išsaugome rezultatus
        src = Path(fname)
        stem, ext = src.stem, src.suffix
        out_pred = Path(output_dir) / f"{stem}_pred{ext}"
        out_gt   = Path(output_dir) / f"{stem}_gt{ext}"
        cv2.imwrite(str(out_pred), pred_vis)
        cv2.imwrite(str(out_gt),   gt_vis)

    print(f"Inference baigta. Išsaugota į {output_dir}")




def evaluate_map(args):

    """
    Įvertina modelio rezultatus: skaičiuoja confusion_matrix, IoU ir mAP.
    """
    output_dir = 'tv_frcnn_resnet_' if args.use_resnet50_fpn else 'tv_frcnn_'
    os.makedirs(output_dir, exist_ok=True)

    # Užkrauna modelį ir test duomenų rinkinį
    faster_rcnn_model, rcnn_data, test_dataset = load_model_and_dataset(args)
    faster_rcnn_model.eval().to(device)

    num_classes = len(rcnn_data.idx2label)
    conf_mat = np.zeros((num_classes, num_classes), dtype=int)
    sum_iou  = np.zeros(num_classes, dtype=float)
    cnt_iou  = np.zeros(num_classes, dtype=int)
    all_gts   = []
    all_preds = []

    # Eina per pirmus 10 test paveikslėlių
    for idx, (im_batch, target_batch, _) in enumerate(tqdm(test_dataset, desc="Eval")):
        if idx >= 10:
            break

        # Išpakuojama vieno elemento batch’ą
        img_tensor = im_batch[0].to(device)               # [C,H,W]
        gt_boxes   = target_batch['bboxes'][0].cpu().numpy()
        gt_labels  = target_batch['labels'][0].cpu().numpy()

        # Vykdoma inference
        with torch.no_grad():
            out = faster_rcnn_model([img_tensor])[0]

        # Paruošiama mAP skaičiavimui
        pred_per_image = {cls: [] for cls in rcnn_data.idx2label.values()}
        gt_per_image   = {cls: [] for cls in rcnn_data.idx2label.values()}

        # threshold predictions
        boxes  = out['boxes'].cpu().numpy()
        labels = out['labels'].cpu().numpy()
        scores = out['scores'].cpu().numpy()
        keep   = scores > 0.05
        boxes, labels, scores = boxes[keep], labels[keep], scores[keep]

        # Surenka predictions pagal klases
        for b, l, s in zip(boxes, labels, scores):
            name = rcnn_data.idx2label[l]
            pred_per_image[name].append([*b.tolist(), s])
        # Surenka ground-truth
        for b, l in zip(gt_boxes, gt_labels):
            name = rcnn_data.idx2label[l]
            gt_per_image[name].append(b.tolist())

        all_preds.append(pred_per_image)
        all_gts.append(gt_per_image)

        # Atnaujina confusion matrix ir IoU
        used_gt = np.zeros(len(gt_boxes), dtype=bool)
        order   = np.argsort(-scores)

        for pi in order:
            pb = boxes[pi]
            pl = labels[pi]
            # Jei IoU ≥ 0.3 ir GT dar nenaudotas → TP
            ious = np.array([get_iou(pb, gb) for gb in gt_boxes])
            best = ious.argmax()
            best_iou = ious[best]
            if best_iou >= 0.3 and not used_gt[best]:
                gl = gt_labels[best]
                conf_mat[gl, pl] += 1
                sum_iou[gl] += best_iou
                cnt_iou[gl] += 1
                used_gt[best] = True
            else:
                # false positive vs background
                conf_mat[0, pl] += 1

        # Neatitikę GT → FN
        for gi, matched in enumerate(used_gt):
            if not matched:
                gl = gt_labels[gi]
                conf_mat[gl, 0] += 1

    # compute & print mAP
    mean_ap, class_aps = compute_map(all_preds, all_gts, iou_threshold=0.0, method='area')
    print("\n=== mAP ===")
    for cls, ap in class_aps.items():
        print(f"  AP[{cls:12s}] = {ap:.4f}")
    print(f"  mAP = {mean_ap:.4f}\n")

    labels = [rcnn_data.idx2label[i] for i in range(num_classes)]
    # Apskaičiuojama precision, recall, F1 ir vid. IoU kiekvienai klasei
    print("=== Precision  Recall  F1-score  Avg IoU ===")
    for i, cls in enumerate(labels):
        tp = conf_mat[i, i]
        fp = conf_mat[:, i].sum() - tp
        fn = conf_mat[i, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        avg_iou = sum_iou[i] / cnt_iou[i] if cnt_iou[i] > 0 else 0.0
        print(f"{cls:12s}  {prec:0.3f}     {rec:0.3f}      {f1:0.3f}     {avg_iou:0.3f}")

    # 7) plot & save confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=labels)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(cmap="Blues", ax=ax, xticks_rotation=45)
    ax.set_title("Aptikimo Confusion Matrix\n(rows = GT, cols = Pred)")
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    fig.savefig(cm_path)
    print(f" Išsaugotas confusion matrix į {cm_path}")
    plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config_path',
                        default='config/rcnn.yaml', type=str)
    parser.add_argument('--evaluate', dest='evaluate',
                        default=True, type=bool)
    parser.add_argument('--infer_samples', dest='infer_samples',
                        default=True, type=bool)
    parser.add_argument('--use_resnet50_fpn', dest='use_resnet50_fpn',
                        default=True, type=bool)
    args = parser.parse_args()

    if args.infer_samples:
        infer(args)
    else:
        print('Neišvedami pavyzdžiai, nes argumentas `infer_samples` yra False')

    if args.evaluate:
        evaluate_map(args)
    else:
        print('Nevertinama, nes argumentas `evaluate` yra False')
