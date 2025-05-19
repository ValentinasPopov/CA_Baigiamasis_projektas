import os
import argparse
import random
import yaml
import torch
import numpy as np
import cv2
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image


from dataLoader import RCNNDataset  # your dataset class

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def collate_function(batch):
    imgs, targets, fnames = zip(*batch)
    return imgs, targets, fnames

def load_model_and_dataset(args):
    # 1) read config
    with open(args.config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    ds_conf    = cfg['dataset_params']
    mdl_conf   = cfg.get('model_params', {})
    train_conf = cfg['train_params']

    # 2) fix seeds
    seed = train_conf.get('seed', 42)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)

    # 3) dataset & loader
    voc_test = VOCDataset(
        split='test',
        im_dir=ds_conf['im_test_path'],
        annotation_json_path=ds_conf['ann_test_path']
    )
    test_loader = DataLoader(
        voc_test, batch_size=1, shuffle=False, collate_fn=collate_function
    )

    # 4) build a 3-class Faster R-CNN
    num_classes  = len(voc_test.idx2label)            # background + knok + scratchs
    #score_thresh padidinti
    score_thresh = mdl_conf.get('box_score_thresh', 0.1)
    min_sz       = mdl_conf.get('min_size', 600)
    max_sz       = mdl_conf.get('max_size', 1000)

    if args.use_resnet50_fpn:

        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            pretrained=True,
            box_score_thresh=score_thresh,
            min_size=min_sz,
            max_size=max_sz
        )
        in_feats = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_feats, num_classes)
    else:
        backbone = torchvision.models.resnet34(
            pretrained=True,
            norm_layer=torchvision.ops.FrozenBatchNorm2d
        )
        modules = list(backbone.children())[:-3]
        backbone = torch.nn.Sequential(*modules)
        backbone.out_channels = 256
        anchor_gen = AnchorGenerator()
        roi_pool   = MultiScaleRoIAlign(
            featmap_names=['0'], output_size=7, sampling_ratio=2
        )
        model = torchvision.models.detection.FasterRCNN(
            backbone,
            num_classes=num_classes,
            box_score_thresh=score_thresh,
            rpn_anchor_generator=anchor_gen,
            box_roi_pool=roi_pool,
            min_size=min_sz,
            max_size=max_sz
        )

    model.to(device).eval()

    # 5) load checkpoint but DROP the old VOC‐21 head weights
    ckpt_dir  = train_conf.get('task_name', 'checkpoints')
    ckpt_name = train_conf.get('ckpt_name', '')
    ckpt_file = ('tv_frcnn_r50fpn_' if args.use_resnet50_fpn else 'tv_frcnn_') + ckpt_name
    path      = os.path.join(ckpt_dir, ckpt_file)
    print(f"Loading weights from {path}")
    raw_state = torch.load(path, map_location=device)

    # filter out any 'roi_heads.box_predictor' keys so our 3‑class head isn't overwritten
    filtered_state = {
        k: v for k, v in raw_state.items()
        if not k.startswith('roi_heads.box_predictor')
    }
    model.load_state_dict(filtered_state, strict=False)

    return model, voc_test, test_loader

def infer(args):
    model, voc_test, _ = load_model_and_dataset(args)
    os.makedirs(args.output_dir, exist_ok=True)

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
    ])

    for i in range(args.num_samples):
        # Correct sampling
        idx = random.randint(0, len(voc_test) - 1)
        img_t, target, fname = voc_test[idx]

        # Reload the original image for drawing
        bgr_image = cv2.imread(fname)
        if bgr_image is None:
            print(f"[Sample {i}] Skipping, couldn't read: {fname}")
            continue

        # Draw GT boxes
        gt = bgr_image.copy()
        for box, lbl in zip(target['bboxes'], target['labels']):
            x1, y1, x2, y2 = map(int, box.tolist())
            cv2.rectangle(gt, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imwrite(os.path.join(args.output_dir, f"gt_{i}.png"), gt)

        # Run prediction
        tensor = transform(Image.fromarray(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB))).to(device)
        with torch.no_grad():
            out = model([tensor])[0]

        # Debug print of raw predictions
        print(f"[Sample {i}] raw preds:",
              [(voc_test.idx2label.get(l.item(), str(l.item())), float(s))
               for l, s in zip(out['labels'], out['scores'])])

        # Draw predicted boxes above threshold
        pred = bgr_image.copy()
        for box, lbl, score in zip(out['boxes'], out['labels'], out['scores']):
            if score < args.score_thr:
                continue
            x1, y1, x2, y2 = map(int, box.tolist())
            cls_name = voc_test.idx2label.get(lbl.item(), f"class_{lbl.item()}")
            text = f"{cls_name}:{score:.2f}"
            cv2.rectangle(pred, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(pred, text, (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        cv2.imwrite(os.path.join(args.output_dir, f"pred_{i}.png"), pred)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config',          dest='config_path',
                   default='config/voc.yaml', type=str)
    p.add_argument('--infer_samples',   dest='infer_samples',
                   default=True, type=bool)
    p.add_argument('--use_resnet50_fpn',dest='use_resnet50_fpn',
                   default=True, type=bool)
    p.add_argument('--num_samples',     dest='num_samples',
                   default=10, type=int)
    p.add_argument('--score_thr',       dest='score_thr',
                   default=0.3, type=float)
    p.add_argument('--output_dir',      dest='output_dir',
                   default='samples_frcnn', type=str)
    args = p.parse_args()

    if args.infer_samples:
        infer(args)
    else:
        print("Skipping inference (--infer_samples=False)")
