from os import mkdir
from os.path import exists

import torch
import argparse
import os
import numpy as np
import yaml
import random
from tqdm import tqdm
import torchvision
from scripts.dataLoader_rcnn import RCNNDataset
from torch.utils.data.dataloader import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.anchor_utils import AnchorGenerator
from pathlib import Path


SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def collate_function(data):
    return tuple(zip(*data))


def train(args):
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    ########################

    dataset_config = config['dataset_params']
    train_config = config['train_params']
    model_config = config['model_params']


    seed = train_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)

    rcnn_data = RCNNDataset('train',
                     im_dir=dataset_config['im_train_path'],
                     annotation_json_path=dataset_config['ann_train_path'])

    total_images = len(rcnn_data)
    num_images = train_config['num_train_images']
    if num_images < total_images:
        idxs = list(range(total_images))
        random.shuffle(idxs)
        indices = idxs[:num_images]
        rcnn_data = torch.utils.data.Subset(rcnn_data, indices)
    else:
        rcnn_data = torch.utils.data.Subset(rcnn_data, list(range(num_images)))


    train_dataset = DataLoader(rcnn_data,
                               batch_size=train_config['batch_size'],
                               shuffle=train_config['shuffle'],
                               num_workers=train_config['num_workers'],
                               collate_fn=collate_function)

    if args.use_resnet50_fpn:
        faster_rcnn_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True,
                                                                                 min_size=model_config['min_im_size'],
                                                                                 max_size=model_config['max_im_size'],
        )
        faster_rcnn_model.roi_heads.box_predictor = FastRCNNPredictor(
            faster_rcnn_model.roi_heads.box_predictor.cls_score.in_features,
            num_classes=dataset_config['num_classes'])
    else:
        backbone = torchvision.models.resnet34(pretrained=True, norm_layer=torchvision.ops.FrozenBatchNorm2d)
        backbone = torch.nn.Sequential(*list(backbone.children())[:-3])
        backbone.out_channels = model_config['backbone_out_channels']
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
                                                                    rpn_post_nms_top_n_test=model_config[
                                                                        'rpn_post_nms_top_n_test'],
                                                                    box_batch_size_per_image=model_config[
                                                                        'box_batch_size_per_image'],
                                                                    )

    faster_rcnn_model.train()
    faster_rcnn_model.to(device)
    if not os.path.exists(train_config['task_name']):
        os.mkdir(train_config['task_name'])

    optimizer = torch.optim.SGD(lr=5e-3,
                                params=filter(lambda p: p.requires_grad, faster_rcnn_model.parameters()),
                                weight_decay=5E-5,
                                momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    num_epochs = train_config['num_epochs']
    step_count = 0

    base_out = PROJECT_ROOT / dataset_config['output'] / train_config['task_name']
    base_out.mkdir(parents=True, exist_ok=True)

    for i in range(num_epochs):
        rpn_classification_losses = []
        rpn_localization_losses = []
        frcnn_classification_losses = []
        frcnn_localization_losses = []
        for ims, targets, _ in tqdm(train_dataset):

            for target in targets:
                target['boxes'] = target['bboxes'].float().to(device)
                del target['bboxes']
                target['labels'] = target['labels'].long().to(device)
            images = [im.float().to(device) for im in ims]
            batch_losses = faster_rcnn_model(images, targets)
            loss = batch_losses['loss_classifier']
            loss += batch_losses['loss_box_reg']
            loss += batch_losses['loss_rpn_box_reg']
            loss += batch_losses['loss_objectness']

            rpn_classification_losses.append(batch_losses['loss_objectness'].item())
            rpn_localization_losses.append(batch_losses['loss_rpn_box_reg'].item())
            frcnn_classification_losses.append(batch_losses['loss_classifier'].item())
            frcnn_localization_losses.append(batch_losses['loss_box_reg'].item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step_count +=1

        scheduler.step()
        print('Finished epoch {}'.format(i))

        if args.use_resnet50_fpn:
            torch.save(faster_rcnn_model.state_dict(), os.path.join(train_config['task_name'],
                                                                    'weight_frcnn_resnet_' + train_config['ckpt_name']))
        else:
            torch.save(faster_rcnn_model.state_dict(), os.path.join(train_config['task_name'],
                                                                    'weight_frcnn_' + train_config['ckpt_name']))
        loss_output = ''
        loss_output += 'RPN Classification Loss : {:.4f}'.format(np.mean(rpn_classification_losses))
        loss_output += ' | RPN Localization Loss : {:.4f}'.format(np.mean(rpn_localization_losses))
        loss_output += ' | FRCNN Classification Loss : {:.4f}'.format(np.mean(frcnn_classification_losses))
        loss_output += ' | FRCNN Localization Loss : {:.4f}'.format(np.mean(frcnn_localization_losses))
        print(loss_output)
    print('Done Training...')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for faster rcnn using torchvision code training')
    parser.add_argument('--config', dest='config_path',
                        default='config/rcnn.yaml', type=str)
    parser.add_argument('--use_resnet50_fpn', dest='use_resnet50_fpn',
                        default=True, type=bool)
    args = parser.parse_args()
    train(args)
