import torch
import torch.nn as nn
import torchvision
import math

import yaml

CONFIG_PATH = "config/rcnn.yaml"

# Auto-select device
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


with open(CONFIG_PATH, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

model_config = config['model_params']

def get_iou(boxes1, boxes2):
    """
    Apskaičiuoja IoU:
      - boxes1: N x 4
      - boxes2: M x 4
    Grąžina N x M matricą, kur elementas [i,j] yra IoU tarp boxes1[i] ir boxes2[j].
    """

    # Apskaičiuojama kiekvieno rėmelio plotus
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Randame IoU koordinates
    x_left = torch.max(boxes1[:, None, 0], boxes2[:, 0])
    y_top = torch.max(boxes1[:, None, 1], boxes2[:, 1])
    x_right = torch.min(boxes1[:, None, 2], boxes2[:, 2])
    y_bottom = torch.min(boxes1[:, None, 3], boxes2[:, 3])

    # Apskaičiuojame intersection plotą, užtikrindami neigiamas reikšmės kaip 0
    intersection_area = (x_right - x_left).clamp(min=0) * (y_bottom - y_top).clamp(min=0)

    # Apskaičiuojame jungtinį plotą ir galutinį IoU
    union = area1[:, None] + area2 - intersection_area
    iou = intersection_area / union
    return iou

def boxes_to_transformation_targets(ground_truth_boxes, anchors_or_proposals):
    r"""
    Given ground-truth boxes and proposals (or anchors), compute the
    transformation targets (tx, ty, tw, th).
    """

    # Išskaičiuojame pločius, aukščius ir centru koordinates
    widths = anchors_or_proposals[:, 2] - anchors_or_proposals[:, 0]
    heights = anchors_or_proposals[:, 3] - anchors_or_proposals[:, 1]
    center_x = anchors_or_proposals[:, 0] + 0.5 * widths
    center_y = anchors_or_proposals[:, 1] + 0.5 * heights

    gt_widths = ground_truth_boxes[:, 2] - ground_truth_boxes[:, 0]
    gt_heights = ground_truth_boxes[:, 3] - ground_truth_boxes[:, 1]
    gt_center_x = ground_truth_boxes[:, 0] + 0.5 * gt_widths
    gt_center_y = ground_truth_boxes[:, 1] + 0.5 * gt_heights

    targets_dx = (gt_center_x - center_x) / widths
    targets_dy = (gt_center_y - center_y) / heights
    targets_dw = torch.log(gt_widths / widths)
    targets_dh = torch.log(gt_heights / heights)
    regression_targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)
    return regression_targets

def apply_regression_pred_to_anchors_or_proposals(box_transform_pred, anchors_or_proposals):
    box_transform_pred = box_transform_pred.reshape(box_transform_pred.size(0), -1, 4)
    w = anchors_or_proposals[:, 2] - anchors_or_proposals[:, 0]
    h = anchors_or_proposals[:, 3] - anchors_or_proposals[:, 1]
    center_x = anchors_or_proposals[:, 0] + 0.5 * w
    center_y = anchors_or_proposals[:, 1] + 0.5 * h

    dx = box_transform_pred[..., 0]
    dy = box_transform_pred[..., 1]
    dw = box_transform_pred[..., 2]
    dh = box_transform_pred[..., 3]

    dw = torch.clamp(dw, max=math.log(1000.0 / 16))
    dh = torch.clamp(dh, max=math.log(1000.0 / 16))

    pred_center_x = dx * w[:, None] + center_x[:, None]
    pred_center_y = dy * h[:, None] + center_y[:, None]
    pred_w = torch.exp(dw) * w[:, None]
    pred_h = torch.exp(dh) * h[:, None]

    pred_box_x1 = pred_center_x - 0.5 * pred_w
    pred_box_y1 = pred_center_y - 0.5 * pred_h
    pred_box_x2 = pred_center_x + 0.5 * pred_w
    pred_box_y2 = pred_center_y + 0.5 * pred_h

    pred_boxes = torch.stack((pred_box_x1, pred_box_y1, pred_box_x2, pred_box_y2), dim=2)
    return pred_boxes

def sample_positive_negative(labels, positive_count, total_count):

    # Randam visus teigiamų labels >= 1 ir neigiamų labels == 0)indeksus
    positive = torch.where(labels >= 1)[0]
    negative = torch.where(labels == 0)[0]

    # Nustatome, kiek iš jų naudosime
    num_pos = min(positive.numel(), positive_count)
    num_neg = min(negative.numel(), total_count - num_pos)

    perm_positive_idxs = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
    perm_negative_idxs = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

    # Gaunam galutinius pasiriktų indeksų masyvus
    pos_idxs = positive[perm_positive_idxs]
    neg_idxs = negative[perm_negative_idxs]


    sampled_pos_idx_mask = torch.zeros_like(labels, dtype=torch.bool)
    sampled_neg_idx_mask = torch.zeros_like(labels, dtype=torch.bool)

    sampled_pos_idx_mask[pos_idxs] = True
    sampled_neg_idx_mask[neg_idxs] = True

    return sampled_neg_idx_mask, sampled_pos_idx_mask

def clamp_boxes_to_image_boundary(boxes, image_shape):
    """
    Užtikrina, kad visos dėžutės koordinates būtų paveikslo ribose.
    """

    boxes_x1 = boxes[..., 0]
    boxes_y1 = boxes[..., 1]
    boxes_x2 = boxes[..., 2]
    boxes_y2 = boxes[..., 3]

    # Nustato paveikslo aukštį ir plotį
    height, width = image_shape[-2:]

    # Riboja x koordinates intervale [0, width]
    boxes_x1 = boxes_x1.clamp(min=0, max=width)
    boxes_x2 = boxes_x2.clamp(min=0, max=width)

    # Riboja y koordinates intervale [0, height]
    boxes_y1 = boxes_y1.clamp(min=0, max=height)
    boxes_y2 = boxes_y2.clamp(min=0, max=height)

    # Sujungiame atgal į Tensor su keturiomis stulpelėmis
    boxes = torch.cat((
                    boxes_x1[..., None],
                    boxes_y1[..., None],
                    boxes_x2[..., None],
                    boxes_y2[..., None]),
                    dim=-1)
    return boxes

def transform_boxes_to_original_size(boxes, new_size, original_size):
    """
    Konvertuoja dėžučių koordinates iš naujo (resized) paveikslėlio atgal į pradinį dydį.
    """

    # Apskaičiuoja aukščio ir pločio santykius: orig / new
    ratios = [
        torch.tensor(s_orig, dtype=torch.float32, device=boxes.device) /
        torch.tensor(s, dtype=torch.float32, device=boxes.device)
        for s, s_orig in zip(new_size, original_size)
    ]
    ratio_height, ratio_width = ratios
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    xmin = xmin * ratio_width
    xmax = xmax * ratio_width
    ymin = ymin * ratio_height
    ymax = ymax * ratio_height

    # Sujungia atgal į Tensor N x 4 formatu
    return torch.stack((xmin, ymin, xmax, ymax), dim=1)

class RegionProposalNetwork(nn.Module):
    """
        Generuoja anchor’us ant feature map sluoksnio
        Prognozuoja objectness score ir bbox offset’us kiekvienam anchor’iui
    """
    def __init__(self, in_channels, scales, aspect_ratios):
        super(RegionProposalNetwork, self).__init__()
        self.scales = scales
        self.low_iou_threshold = model_config['rpn_bg_threshold']
        self.high_iou_threshold = model_config['rpn_fg_threshold']
        self.rpn_nms_threshold = model_config['rpn_nms_threshold']
        self.rpn_batch_size = model_config['rpn_batch_size']
        self.rpn_pos_count = int(model_config['rpn_pos_fraction'] * self.rpn_batch_size)
        self.rpn_topk = model_config['rpn_train_topk'] if self.training else model_config['rpn_test_topk']
        self.rpn_prenms_topk = model_config['rpn_train_prenms_topk'] if self.training else model_config['rpn_test_prenms_topk']
        self.aspect_ratios = aspect_ratios
        self.num_anchors = len(self.scales) * len(self.aspect_ratios)

        self.rpn_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.cls_layer = nn.Conv2d(in_channels, self.num_anchors, kernel_size=1, stride=1)
        self.bbox_reg_layer = nn.Conv2d(in_channels, self.num_anchors * 4, kernel_size=1, stride=1)

        for layer in [self.rpn_conv, self.cls_layer, self.bbox_reg_layer]:
            torch.nn.init.normal_(layer.weight, std=0.01)
            torch.nn.init.constant_(layer.bias, 0)

    def generate_anchors(self, image, feat):

        """
         Generuoja visus anchor’us visiems pozicijų taškams: Base anchors pagal scales+aspect_ratios
         """

        # Feature map matmenys
        grid_h, grid_w = feat.shape[-2:]

        # Pradinio vaizdo matmenys
        image_h, image_w = image.shape[-2:]

        # Stride tarp pikselių feature ir originalaus vaizdo
        stride_h = torch.tensor(image_h // grid_h, dtype=torch.int64, device=feat.device)
        stride_w = torch.tensor(image_w // grid_w, dtype=torch.int64, device=feat.device)

        # Konvertuoja scales ir aspect_ratios į Tensor’us
        scales = torch.as_tensor(self.scales, dtype=feat.dtype, device=feat.device)
        aspect_ratios = torch.as_tensor(self.aspect_ratios, dtype=feat.dtype, device=feat.device)

        # Apskaičiuoja w ir h
        h_ratios = torch.sqrt(aspect_ratios)
        w_ratios = 1 / h_ratios
        ws = (w_ratios[:, None] * scales[None, :]).view(-1)
        hs = (h_ratios[:, None] * scales[None, :]).view(-1)

        # Base anchors viename centre, su puse pločio/aukščio
        base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2
        base_anchors = base_anchors.round()

        # Generuojame shift‘us tinklelyje
        shifts_x = torch.arange(0, grid_w, dtype=torch.int32, device=feat.device) * stride_w
        shifts_y = torch.arange(0, grid_h, dtype=torch.int32, device=feat.device) * stride_h
        shifts_y, shifts_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
        shifts_x = shifts_x.reshape(-1)
        shifts_y = shifts_y.reshape(-1)
        shifts = torch.stack((shifts_x, shifts_y, shifts_x, shifts_y), dim=1)

        # Pridedame base anchors prie kiekvieno shift‘o
        anchors = (shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4)
        return anchors

    def assign_targets_to_anchors(self, anchors, gt_boxes):
        """
            Priskiria ground-truth dėžutes anchor’iui pagal IoU:
            - ≥ high_threshold → teigiamas
            - < low_threshold  → fonas
            - Užtikrina, kad kiekvienai GT yra bent vienas anchor
        """

        # IoU matrica tarp gt ir anchor’ų
        iou_matrix = get_iou(gt_boxes, anchors)

        # Kiekvienam anchor randame geriausią GT indeksą
        best_match_iou, best_match_gt_idx = iou_matrix.max(dim=0)
        best_match_gt_idx_pre_thresholding = best_match_gt_idx.clone()

        # Žymi mažo/tarp ribų atvejus
        below_low_threshold = best_match_iou < self.low_iou_threshold
        between_thresholds = (best_match_iou >= self.low_iou_threshold) & (best_match_iou < self.high_iou_threshold)
        best_match_gt_idx[below_low_threshold] = -1
        best_match_gt_idx[between_thresholds] = -2

        # Užtikriname, kad kiekvienas GT turi bent vieną anchor
        best_anchor_iou_for_gt, _ = iou_matrix.max(dim=1)
        gt_pred_pair_with_highest_iou = torch.where(iou_matrix == best_anchor_iou_for_gt[:, None])
        pred_inds_to_update = gt_pred_pair_with_highest_iou[1]
        best_match_gt_idx[pred_inds_to_update] = best_match_gt_idx_pre_thresholding[pred_inds_to_update]

        # Sukuriame matched GT boxes ir labels
        matched_gt_boxes = gt_boxes[best_match_gt_idx.clamp(min=0)]
        labels = (best_match_gt_idx >= 0).to(dtype=torch.float32)
        background_anchors = best_match_gt_idx == -1
        labels[background_anchors] = 0.0
        ignored_anchors = best_match_gt_idx == -2
        labels[ignored_anchors] = -1.0
        return labels, matched_gt_boxes

    def filter_proposals(self, proposals, cls_scores, image_shape):
        """
           Atrinkti geriausius proposals pagal:
           - objectness score
           - minimalaus dydžio filtravimą
           - NMS
           - top-K
        """

        # Pertvarkoe scores ir taikome sigmoid
        cls_scores = cls_scores.reshape(-1)
        cls_scores = torch.sigmoid(cls_scores)

        # Išrenkame top prieš-NMS
        _, top_n_idx = cls_scores.topk(min(self.rpn_prenms_topk, len(cls_scores)))
        cls_scores = cls_scores[top_n_idx]
        proposals = proposals[top_n_idx]
        proposals = clamp_boxes_to_image_boundary(proposals, image_shape)
        min_size = 16

        # Patikriname minimalų plotį/aukštį
        ws = proposals[:, 2] - proposals[:, 0]
        hs = proposals[:, 3] - proposals[:, 1]
        keep = (ws >= min_size) & (hs >= min_size)
        keep = torch.where(keep)[0]
        proposals = proposals[keep]
        cls_scores = cls_scores[keep]
        keep_mask = torch.zeros_like(cls_scores, dtype=torch.bool)
        keep_indices = torch.ops.torchvision.nms(proposals, cls_scores, self.rpn_nms_threshold)
        keep_mask[keep_indices] = True
        keep_indices = torch.where(keep_mask)[0]
        post_nms_keep_indices = keep_indices[cls_scores[keep_indices].sort(descending=True)[1]]
        proposals, cls_scores = (proposals[post_nms_keep_indices[:self.rpn_topk]],
                                 cls_scores[post_nms_keep_indices[:self.rpn_topk]])
        return proposals, cls_scores

    def forward(self, image, feat, target=None):

        # RPN feature su ReLU
        rpn_feat = nn.ReLU()(self.rpn_conv(feat))

        # Prognozuojam objectness ir bbox deltas
        cls_scores = self.cls_layer(rpn_feat)
        box_transform_pred = self.bbox_reg_layer(rpn_feat)

        # Generuojam anchor’us tinklelyje
        anchors = self.generate_anchors(image, feat)

        # Pertvarkom formatus ir taikome deltas
        number_of_anchors_per_location = cls_scores.size(1)
        cls_scores = cls_scores.permute(0, 2, 3, 1).reshape(-1, 1)
        box_transform_pred = box_transform_pred.view(
            box_transform_pred.size(0),
            number_of_anchors_per_location,
            4,
            rpn_feat.shape[-2],
            rpn_feat.shape[-1]
        ).permute(0, 3, 4, 1, 2).reshape(-1, 4)

        # Atrenkam geriausius proposals
        proposals = apply_regression_pred_to_anchors_or_proposals(box_transform_pred.detach().reshape(-1, 1, 4), anchors)
        proposals = proposals.reshape(proposals.size(0), 4)
        proposals, scores = self.filter_proposals(proposals, cls_scores.detach(), image.shape)
        rpn_output = {'proposals': proposals, 'scores': scores}
        if not self.training or target is None:
            return rpn_output
        else:
            # If ground-truth boxes are empty, create dummy targets.
            if target['bboxes'].numel() == 0:
                labels_for_anchors = torch.zeros(anchors.shape[0], dtype=torch.float32, device=anchors.device)
                matched_gt_boxes_for_anchors = torch.zeros((anchors.shape[0], 4), dtype=anchors.dtype, device=anchors.device)
            else:
                labels_for_anchors, matched_gt_boxes_for_anchors = self.assign_targets_to_anchors(
                    anchors, target['bboxes']
                )
            regression_targets = boxes_to_transformation_targets(matched_gt_boxes_for_anchors, anchors)
            sampled_neg_idx_mask, sampled_pos_idx_mask = sample_positive_negative(
                labels_for_anchors, positive_count=self.rpn_pos_count, total_count=self.rpn_batch_size
            )
            sampled_idxs = torch.where(sampled_pos_idx_mask | sampled_neg_idx_mask)[0]
            localization_loss = (
                torch.nn.functional.smooth_l1_loss(
                    box_transform_pred[sampled_pos_idx_mask],
                    regression_targets[sampled_pos_idx_mask],
                    beta=1 / 9,
                    reduction="sum",
                ) / (sampled_idxs.numel())
            )
            cls_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                cls_scores[sampled_idxs].flatten(),
                labels_for_anchors[sampled_idxs].flatten()
            )
            rpn_output['rpn_classification_loss'] = cls_loss
            rpn_output['rpn_localization_loss'] = localization_loss
            return rpn_output

class ROIHead(nn.Module):
    """
     ROI head that performs:
      1) atlieka ROI pooling ant pasiūlytų regionų
      2) Apdoroja per du pilnai sujungtus sluoksnius (fc6, fc7)
      3) Prognozuoja galutines klasės tikimybes ir dėžučių offset’us
    """
    def __init__(self, in_channels):
        super(ROIHead, self).__init__()
        self.num_classes = model_config['num_classes']
        self.roi_batch_size = model_config['roi_batch_size']
        self.roi_pos_count = int(model_config['roi_pos_fraction'] * self.roi_batch_size)
        self.iou_threshold = model_config['roi_iou_threshold']
        self.low_bg_iou = model_config['roi_low_bg_iou']
        self.nms_threshold = model_config['roi_nms_threshold']
        self.topK_detections = model_config['roi_topk_detections']
        self.low_score_threshold = model_config['roi_score_threshold']
        self.pool_size = model_config['roi_pool_size']
        self.fc_inner_dim = model_config['fc_inner_dim']

        # --- Pilnai sujungti sluoksniai po ROI pooling ---
        # fc6: įėjimas = flattened pooled features, išėjimas = inner dim
        self.fc6 = nn.Linear(in_channels * self.pool_size * self.pool_size, self.fc_inner_dim)
        # fc7: antras MLP sluoksnis
        self.fc7 = nn.Linear(self.fc_inner_dim, self.fc_inner_dim)
        # Klasifikacijos sluoksnis: prognozuoja klases
        self.cls_layer = nn.Linear(self.fc_inner_dim, self.num_classes)
        # BBox regresijos sluoksnis: prognozuoja off­set’us
        self.bbox_reg_layer = nn.Linear(self.fc_inner_dim, self.num_classes * 4)

        # --- Inicijuojame weights ir bias’us ---
        torch.nn.init.normal_(self.cls_layer.weight, std=0.01)
        torch.nn.init.constant_(self.cls_layer.bias, 0)
        torch.nn.init.normal_(self.bbox_reg_layer.weight, std=0.001)
        torch.nn.init.constant_(self.bbox_reg_layer.bias, 0)

    def assign_target_to_proposals(self, proposals, gt_boxes, gt_labels):

        # Pritaikome gt_boxes ir gt_labels formą
        if gt_boxes.ndim > 2:
            gt_boxes = gt_boxes.squeeze(0)
        if gt_labels.ndim > 1:
            gt_labels = gt_labels.squeeze(0)

        # Jei nėra GT dėžučių – pažymime visus pasiūlymus kaip foną
        if gt_boxes.numel() == 0:
            labels = torch.zeros(proposals.shape[0], dtype=torch.int64, device=proposals.device)
            dummy_boxes = torch.zeros((proposals.shape[0], 4), dtype=gt_boxes.dtype, device=gt_boxes.device)
            return labels, dummy_boxes

        # Apskaičiuojame IoU matricą tarp GT ir proposals.
        iou_matrix = get_iou(gt_boxes, proposals)
        best_match_iou, best_match_gt_idx = iou_matrix.max(dim=0)

        # Nustatyti proposals su maža ir labai maža IOU.
        background_proposals = (best_match_iou < self.iou_threshold) & (best_match_iou >= self.low_bg_iou)
        ignored_proposals = best_match_iou < self.low_bg_iou

        best_match_gt_idx[background_proposals] = -1
        best_match_gt_idx[ignored_proposals] = -2

        # Gautas GT dėžutes priskiriame pagal clamp(min=0)
        matched_gt_boxes_for_proposals = gt_boxes[best_match_gt_idx.clamp(min=0)]

        # Parinka atitinkamas labels.
        labels = gt_labels[best_match_gt_idx.clamp(min=0)].to(dtype=torch.int64)

        # Nustato labels: fono proposals gauna 0 ir ignoruojama -1
        labels[background_proposals] = 0
        labels[ignored_proposals] = -1

        return labels, matched_gt_boxes_for_proposals

    def forward(self, feat, proposals, image_shape, target):
        if self.training and target is not None:
            # If there are ground-truth boxes, use them.
            if target['bboxes'].numel() > 0:
                proposals = torch.cat([proposals, target['bboxes']], dim=0)
            gt_boxes = target['bboxes']
            gt_labels = target['labels']
            labels, matched_gt_boxes_for_proposals = self.assign_target_to_proposals(proposals, gt_boxes, gt_labels)
            sampled_neg_idx_mask, sampled_pos_idx_mask = sample_positive_negative(
                labels, positive_count=self.roi_pos_count, total_count=self.roi_batch_size)
            sampled_idxs = torch.where(sampled_pos_idx_mask | sampled_neg_idx_mask)[0]
            proposals = proposals[sampled_idxs]
            labels = labels[sampled_idxs]
            matched_gt_boxes_for_proposals = matched_gt_boxes_for_proposals[sampled_idxs]
            regression_targets = boxes_to_transformation_targets(matched_gt_boxes_for_proposals, proposals)
        size = feat.shape[-2:]
        possible_scales = []
        for s1, s2 in zip(size, image_shape):
            approx_scale = float(s1) / float(s2)
            scale = 2 ** float(torch.tensor(approx_scale).log2().round())
            possible_scales.append(scale)
        assert possible_scales[0] == possible_scales[1]

        # ROI pooling
        proposal_roi_pool_feats = torchvision.ops.roi_pool(feat, [proposals],
                                                           output_size=self.pool_size,
                                                           spatial_scale=possible_scales[0])
        #ROI feature prieš fc sluoksnius
        proposal_roi_pool_feats = proposal_roi_pool_feats.flatten(start_dim=1)

        # FC sluoksniai su ReLU aktyvacija
        box_fc_6 = torch.nn.functional.relu(self.fc6(proposal_roi_pool_feats))
        box_fc_7 = torch.nn.functional.relu(self.fc7(box_fc_6))

        # Galutiniai sluoksniai
        cls_scores = self.cls_layer(box_fc_7)
        box_transform_pred = self.bbox_reg_layer(box_fc_7)
        num_boxes, num_classes = cls_scores.shape
        box_transform_pred = box_transform_pred.reshape(num_boxes, num_classes, 4)
        frcnn_output = {}
        if self.training and target is not None:
            classification_loss = torch.nn.functional.cross_entropy(cls_scores, labels)
            fg_proposals_idxs = torch.where(labels > 0)[0]
            fg_cls_labels = labels[fg_proposals_idxs]
            localization_loss = torch.nn.functional.smooth_l1_loss(
                box_transform_pred[fg_proposals_idxs, fg_cls_labels],
                regression_targets[fg_proposals_idxs],
                beta=1 / 9,
                reduction="sum",
            )
            localization_loss = localization_loss / labels.numel()
            frcnn_output['frcnn_classification_loss'] = classification_loss
            frcnn_output['frcnn_localization_loss'] = localization_loss
        if self.training:
            return frcnn_output
        else:
            device = cls_scores.device
            pred_boxes = apply_regression_pred_to_anchors_or_proposals(box_transform_pred, proposals)
            pred_scores = torch.nn.functional.softmax(cls_scores, dim=-1)
            pred_boxes = clamp_boxes_to_image_boundary(pred_boxes, image_shape)
            pred_labels = torch.arange(num_classes, device=device).view(1, -1).expand_as(pred_scores)
            pred_boxes = pred_boxes[:, 1:]
            pred_scores = pred_scores[:, 1:]
            pred_labels = pred_labels[:, 1:]
            pred_boxes = pred_boxes.reshape(-1, 4)
            pred_scores = pred_scores.reshape(-1)
            pred_labels = pred_labels.reshape(-1)
            pred_boxes, pred_labels, pred_scores = self.filter_predictions(pred_boxes, pred_labels, pred_scores)
            frcnn_output['boxes'] = pred_boxes
            frcnn_output['scores'] = pred_scores
            frcnn_output['labels'] = pred_labels
            return frcnn_output

    def filter_predictions(self, pred_boxes, pred_labels, pred_scores):
        keep = torch.where(pred_scores > self.low_score_threshold)[0]
        pred_boxes, pred_scores, pred_labels = pred_boxes[keep], pred_scores[keep], pred_labels[keep]
        min_size = 16
        ws = pred_boxes[:, 2] - pred_boxes[:, 0]
        hs = pred_boxes[:, 3] - pred_boxes[:, 1]
        keep = torch.where((ws >= min_size) & (hs >= min_size))[0]
        pred_boxes, pred_scores, pred_labels = pred_boxes[keep], pred_scores[keep], pred_labels[keep]
        keep_mask = torch.zeros_like(pred_scores, dtype=torch.bool)
        for class_id in torch.unique(pred_labels):
            curr_indices = torch.where(pred_labels == class_id)[0]
            curr_keep_indices = torch.ops.torchvision.nms(pred_boxes[curr_indices],
                                                          pred_scores[curr_indices],
                                                          self.nms_threshold)
            keep_mask[curr_indices[curr_keep_indices]] = True
        keep_indices = torch.where(keep_mask)[0]
        post_nms_keep_indices = keep_indices[pred_scores[keep_indices].sort(descending=True)[1]]
        keep = post_nms_keep_indices[:self.topK_detections]
        pred_boxes, pred_scores, pred_labels = pred_boxes[keep], pred_scores[keep], pred_labels[keep]
        return pred_boxes, pred_labels, pred_scores

class FasterRCNN(nn.Module):
    def __init__(self):
        super(FasterRCNN, self).__init__()
        self.model_config = model_config
        vgg16 = torchvision.models.vgg16(pretrained=True)
        self.backbone = vgg16.features[:-1]
        self.rpn = RegionProposalNetwork(model_config['backbone_out_channels'],
                                         scales=model_config['scales'],
                                         aspect_ratios=model_config['aspect_ratios'],
                                         model_config=model_config)
        self.roi_head = ROIHead(in_channels=model_config['backbone_out_channels'])
        for layer in self.backbone[:10]:
            for p in layer.parameters():
                p.requires_grad = False
        self.image_mean = [0.485, 0.456, 0.406]
        self.image_std = [0.229, 0.224, 0.225]
        self.min_size = model_config['min_im_size']
        self.max_size = model_config['max_im_size']

    def normalize_resize_image_and_boxes(self, image, bboxes):
        dtype, device = image.dtype, image.device
        # Normalize the image.
        mean = torch.as_tensor(self.image_mean, dtype=dtype, device=device)
        std = torch.as_tensor(self.image_std, dtype=dtype, device=device)
        image = (image - mean[:, None, None]) / std[:, None, None]

        h, w = image.shape[-2:]
        im_shape = torch.tensor(image.shape[-2:], dtype=torch.float32, device=device)
        min_size = torch.min(im_shape)
        max_size = torch.max(im_shape)
        scale = torch.min(float(self.min_size) / min_size, float(self.max_size) / max_size)
        scale_factor = scale.item()

        image = torch.nn.functional.interpolate(
            image,
            size=None,
            scale_factor=scale_factor,
            mode="bilinear",
            recompute_scale_factor=True,
            align_corners=False,
        )

        if bboxes is not None:
            # If bboxes are provided as a list, extract the first element.
            if isinstance(bboxes, list):
                if len(bboxes) == 0:
                    bboxes = torch.empty((0, 4), dtype=torch.float32, device=device)
                    return image, bboxes
                else:
                    bboxes = bboxes[0]
            # If bboxes is 3D with a leading batch dimension of 1, squeeze it.
            if bboxes.ndim == 3 and bboxes.shape[0] == 1:
                bboxes = bboxes.squeeze(0)
            if bboxes.numel() == 0 or (bboxes.ndim > 1 and bboxes.shape[0] == 0):
                bboxes = torch.empty((0, 4), dtype=bboxes.dtype, device=bboxes.device)
                return image, bboxes
            # Compute scaling ratios.
            ratio_height = image.shape[-2] / float(h)
            ratio_width = image.shape[-1] / float(w)
            if bboxes.ndim == 1:
                bboxes = bboxes.unsqueeze(0)
            if bboxes.shape[1] != 4:
                raise ValueError("Expected bounding boxes with 4 values per box, got shape: {}".format(bboxes.shape))
            xmin, ymin, xmax, ymax = bboxes.unbind(1)
            xmin = xmin * ratio_width
            xmax = xmax * ratio_width
            ymin = ymin * ratio_height
            ymax = ymax * ratio_height
            bboxes = torch.stack((xmin, ymin, xmax, ymax), dim=1)
        return image, bboxes

    def forward(self, image, target=None):
        old_shape = image.shape[-2:]
        if self.training:
            image, bboxes = self.normalize_resize_image_and_boxes(image, target['bboxes'])
            target['bboxes'] = bboxes
        else:
            image, _ = self.normalize_resize_image_and_boxes(image, None)
        feat = self.backbone(image)
        rpn_output = self.rpn(image, feat, target)
        proposals = rpn_output['proposals']
        frcnn_output = self.roi_head(feat, proposals, image.shape[-2:], target)
        if not self.training:
            frcnn_output['boxes'] = transform_boxes_to_original_size(frcnn_output['boxes'],
                                                                     image.shape[-2:],
                                                                     old_shape)
        return rpn_output, frcnn_output

if __name__ == "__main__":

    faster_rcnn_model = FasterRCNN()
    faster_rcnn_model.train()
    faster_rcnn_model.to(device)
    # Dummy image: 3 x 800 x 600
    dummy_image = torch.rand(3, 800, 600)
    # Dummy boxes and labels, provided as a tensor of shape [1, N, 4] (for a single-image batch)
    dummy_boxes = torch.tensor([[[50, 60, 200, 220], [200, 150, 250, 200], [300, 300, 350, 350]]],
                               dtype=torch.float32)
    dummy_labels = torch.tensor([[1, 1, 1]], dtype=torch.int64)
    target = {'bboxes': dummy_boxes, 'labels': dummy_labels}
    out = faster_rcnn_model(dummy_image, target)
    print("Model output:", out)

