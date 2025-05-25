
import torch
import numpy as np
import cv2
import torchvision
import argparse
import random
import os
import yaml
from tqdm import tqdm
from models.RCNN import FasterRCNN
from scripts.dataLoader_rcnn import RCNNDataset
from torch.utils.data.dataloader import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.anchor_utils import AnchorGenerator
import os
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from pathlib import Path


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_iou(det, gt):
    det_x1, det_y1, det_x2, det_y2 = det
    gt_x1, gt_y1, gt_x2, gt_y2 = gt

    x_left = max(det_x1, gt_x1)
    y_top = max(det_y1, gt_y1)
    x_right = min(det_x2, gt_x2)
    y_bottom = min(det_y2, gt_y2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    area_intersection = (x_right - x_left) * (y_bottom - y_top)
    det_area = (det_x2 - det_x1) * (det_y2 - det_y1)
    gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
    area_union = float(det_area + gt_area - area_intersection + 1E-6)
    iou = area_intersection / area_union
    return iou


def compute_map(det_boxes, gt_boxes, iou_threshold=0.4, method='area'):


    gt_labels = {cls_key for im_gt in gt_boxes for cls_key in im_gt.keys()}
    gt_labels = sorted(gt_labels)
    all_aps = {}
    # average precisions for ALL classes
    aps = []
    for idx, label in enumerate(gt_labels):
        # Get detection predictions of this class
        cls_dets = [
            [im_idx, im_dets_label] for im_idx, im_dets in enumerate(det_boxes)
            if label in im_dets for im_dets_label in im_dets[label]
        ]
        # Sort them by confidence score
        cls_dets = sorted(cls_dets, key=lambda k: -k[1][-1])

        # For tracking which gt boxes of this class have already been matched
        gt_matched = [[False for _ in im_gts[label]] for im_gts in gt_boxes]
        # Number of gt boxes for this class for recall calculation
        num_gts = sum([len(im_gts[label]) for im_gts in gt_boxes])
        tp = [0] * len(cls_dets)
        fp = [0] * len(cls_dets)

        # For each prediction
        for det_idx, (im_idx, det_pred) in enumerate(cls_dets):
            # Get gt boxes for this image and this label
            im_gts = gt_boxes[im_idx][label]
            max_iou_found = -1
            max_iou_gt_idx = -1

            # Get best matching gt box
            for gt_box_idx, gt_box in enumerate(im_gts):
                gt_box_iou = get_iou(det_pred[:-1], gt_box)
                if gt_box_iou > max_iou_found:
                    max_iou_found = gt_box_iou
                    max_iou_gt_idx = gt_box_idx
            # TP only if iou >= threshold and this gt has not yet been matched
            if max_iou_found < iou_threshold or gt_matched[im_idx][max_iou_gt_idx]:
                fp[det_idx] = 1
            else:
                tp[det_idx] = 1
                # If tp then we set this gt box as matched
                gt_matched[im_idx][max_iou_gt_idx] = True
        # Cumulative tp and fp
        tp = np.cumsum(tp)
        fp = np.cumsum(fp)

        eps = np.finfo(np.float32).eps
        recalls = tp / np.maximum(num_gts, eps)
        precisions = tp / np.maximum((tp + fp), eps)

        if method == 'area':
            recalls = np.concatenate(([0.0], recalls, [1.0]))
            precisions = np.concatenate(([0.0], precisions, [0.0]))

            # Replace precision values with recall r with maximum precision value
            # of any recall value >= r
            # This computes the precision envelope
            for i in range(precisions.size - 1, 0, -1):
                precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])
            # For computing area, get points where recall changes value
            i = np.where(recalls[1:] != recalls[:-1])[0]
            # Add the rectangular areas to get ap
            ap = np.sum((recalls[i + 1] - recalls[i]) * precisions[i + 1])
        elif method == 'interp':
            ap = 0.0
            for interp_pt in np.arange(0, 1 + 1E-3, 0.1):
                # Get precision values for recall values >= interp_pt
                prec_interp_pt = precisions[recalls >= interp_pt]

                # Get max of those precision values
                prec_interp_pt = prec_interp_pt.max() if prec_interp_pt.size > 0.0 else 0.0
                ap += prec_interp_pt
            ap = ap / 11.0
        else:
            raise ValueError('Method can only be area or interp')
        if num_gts > 0:
            aps.append(ap)
            all_aps[label] = ap
        else:
            all_aps[label] = np.nan
    # compute mAP at provided iou threshold
    mean_ap = sum(aps) / len(aps)
    return mean_ap, all_aps


def load_model_and_dataset(args):
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    ########################

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
    # 1) Prepare output directory
    suffix = "resnet_" if args.use_resnet50_fpn else ""
    output_dir = f"tv_frcnn_{suffix}"
    os.makedirs(output_dir, exist_ok=True)

    # 2) Load model & data
    model, dataset, _ = load_model_and_dataset(args)
    model.to(device).eval()

    # 3) Inference loop
    for _ in tqdm(range(10), desc="Inference"):
        # ---- pick a random sample ----
        idx = random.randrange(len(dataset))
        img_tensor, target, fname = dataset[idx]

        # ---- run the model ----
        with torch.no_grad():
            output = model([img_tensor.to(device)])[0]

        # ---- prepare visualizations ----
        orig = cv2.imread(fname)
        if orig is None:
            raise RuntimeError(f"Failed to load image {fname}")
        pred_vis = orig.copy()
        gt_vis   = orig.copy()

        # ---- draw predictions ----
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

        # ---- draw ground-truth ----
        for box, lbl in zip(target['bboxes'], target['labels']):
            x1,y1,x2,y2 = box.int().cpu().numpy()
            # solid green box
            cv2.rectangle(gt_vis, (x1,y1), (x2,y2), (0,255,0), 2)
            # label background + text
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

        # ---- save outputs ----
        src = Path(fname)
        stem, ext = src.stem, src.suffix
        out_pred = Path(output_dir) / f"{stem}_pred{ext}"
        out_gt   = Path(output_dir) / f"{stem}_gt{ext}"
        cv2.imwrite(str(out_pred), pred_vis)
        cv2.imwrite(str(out_gt),   gt_vis)

    print(f"Inference complete. Saved to {output_dir}")




def evaluate_map(args):

    # 0) prepare output directory for saving confusion matrix
    output_dir = 'tv_frcnn_resnet_' if args.use_resnet50_fpn else 'tv_frcnn_'
    os.makedirs(output_dir, exist_ok=True)

    # 1) load model + data
    faster_rcnn_model, rcnn_data, test_dataset = load_model_and_dataset(args)
    faster_rcnn_model.eval().to(device)

    # 2) prepare accumulators
    num_classes = len(rcnn_data.idx2label)
    conf_mat = np.zeros((num_classes, num_classes), dtype=int)
    sum_iou  = np.zeros(num_classes, dtype=float)
    cnt_iou  = np.zeros(num_classes, dtype=int)
    all_gts   = []
    all_preds = []

    # 3) loop over up to 10 test images
    for idx, (im_batch, target_batch, _) in enumerate(tqdm(test_dataset, desc="Eval")):
        if idx >= 10:
            break

        # unpack the single‐item batch
        img_tensor = im_batch[0].to(device)               # [C,H,W]
        gt_boxes   = target_batch['bboxes'][0].cpu().numpy()
        gt_labels  = target_batch['labels'][0].cpu().numpy()

        # run inference
        with torch.no_grad():
            out = faster_rcnn_model([img_tensor])[0]

        # --- prepare data for mAP calculation ---
        pred_per_image = {cls: [] for cls in rcnn_data.idx2label.values()}
        gt_per_image   = {cls: [] for cls in rcnn_data.idx2label.values()}

        # threshold predictions
        boxes  = out['boxes'].cpu().numpy()
        labels = out['labels'].cpu().numpy()
        scores = out['scores'].cpu().numpy()
        keep   = scores > 0.05
        boxes, labels, scores = boxes[keep], labels[keep], scores[keep]

        # collect predictions
        for b, l, s in zip(boxes, labels, scores):
            name = rcnn_data.idx2label[l]
            pred_per_image[name].append([*b.tolist(), s])
        # collect ground truth
        for b, l in zip(gt_boxes, gt_labels):
            name = rcnn_data.idx2label[l]
            gt_per_image[name].append(b.tolist())

        all_preds.append(pred_per_image)
        all_gts.append(gt_per_image)

        # --- update confusion matrix & IoU stats ---
        used_gt = np.zeros(len(gt_boxes), dtype=bool)
        order   = np.argsort(-scores)

        for pi in order:
            pb = boxes[pi]
            pl = labels[pi]
            # compute IoUs against all GT
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

        # any unmatched GT → false negatives
        for gi, matched in enumerate(used_gt):
            if not matched:
                gl = gt_labels[gi]
                conf_mat[gl, 0] += 1

    # 4) compute & print mAP
    mean_ap, class_aps = compute_map(all_preds, all_gts, iou_threshold=0.0, method='area')
    print("\n=== mAP ===")
    for cls, ap in class_aps.items():
        print(f"  AP[{cls:12s}] = {ap:.4f}")
    print(f"  mAP = {mean_ap:.4f}\n")

    labels = [rcnn_data.idx2label[i] for i in range(num_classes)]
    # 6) precision / recall / F1 / avg IoU per class
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
    ax.set_title("Detection Confusion Matrix\n(rows = GT, cols = Pred)")
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    fig.savefig(cm_path)
    print(f"→ Saved confusion matrix plot to {cm_path}")
    plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for inference using torchvision code faster rcnn')
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
        print('Not Inferring for samples as `infer_samples` argument is False')

    if args.evaluate:
        evaluate_map(args)
    else:
        print('Not Evaluating as `evaluate` argument is False')
