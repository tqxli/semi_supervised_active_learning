import torch
import torch.nn.functional as F
from torchvision.models.detection.roi_heads import fastrcnn_loss
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.ops import boxes as box_ops

from torch.jit.annotations import List, Dict

class RoIHeads(RoIHeads):
    def postprocess_detections(self, class_logits, box_regression, proposals, image_shapes):
        device = class_logits.device
        num_classes = class_logits.shape[-1]

        boxes_per_image = [len(boxes_in_image) for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        pred_scores = F.softmax(class_logits, -1)
        # split boxes and scores per image
        pred_boxes = pred_boxes.split(boxes_per_image, 0)
        pred_scores = pred_scores.split(boxes_per_image, 0)
        all_boxes = []
        all_scores = []
        all_labels = []
        all_props = []
        all_prob_max = []
        all_scores_cls = []

        for boxes, scores, props, image_shape in zip(pred_boxes, pred_scores, proposals, image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)
            scores_cls = scores.unsqueeze(1).expand(scores.shape[0], scores.shape[1] - 1, scores.shape[1])
            scores_cls = scores_cls.reshape(-1, scores.shape[1])
            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]
            props = props.unsqueeze(1).expand(props.shape[0], boxes.shape[1], props.shape[1])
            # batch everything, by making every class prediction be a separate instance
            prob_max = torch.max(scores, 1)[0]
            prob_max = prob_max.unsqueeze(1).expand(prob_max.shape[0], scores.shape[1])
            boxes = boxes.reshape(-1, 4)
            scores = scores.flatten()
            prob_max = prob_max.flatten()
            labels = labels.flatten()
            props = props.reshape(-1, 4)
            # remove low scoring boxes
            inds = torch.nonzero(scores > self.score_thresh).squeeze(1)
            boxes, scores, labels, props, prob_max, scores_cls = boxes[inds], scores[inds], labels[inds], props[inds], \
                                                                 prob_max[inds], scores_cls[inds]
            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            # keep only top k scoring predictions
            keep = keep[:self.detections_per_img]
            boxes, scores, labels, props, prob_max, scores_cls = \
                boxes[keep], scores[keep], labels[keep], props[keep], prob_max[keep], scores_cls[keep]
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)
            all_props.append(props)
            all_prob_max.append(prob_max)
            all_scores_cls.append(scores_cls)
        return all_boxes, all_scores, all_labels, all_props, all_prob_max, all_scores_cls

    def forward(self, features, proposals, image_shapes, targets=None):
        """
        Arguments:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        if targets is not None:
            for t in targets:
                floating_point_types = (torch.float, torch.double, torch.half)
                assert t["boxes"].dtype in floating_point_types, 'target boxes must of float type'
                assert t["labels"].dtype == torch.int64, 'target labels must of int64 type'
                if self.has_keypoint():
                    assert t["keypoints"].dtype == torch.float32, 'target keypoints must of float type'
        if self.training:
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None
            matched_idxs = None
        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)
        result = torch.jit.annotate(List[Dict[str, torch.Tensor]], [])
        losses = {}
        if self.training:
            assert labels is not None and regression_targets is not None
            loss_classifier, loss_box_reg = fastrcnn_loss(
                class_logits, box_regression, labels, regression_targets)
            losses = {
                "loss_classifier": loss_classifier,
                "loss_box_reg": loss_box_reg
            }
        else:
            boxes, scores, labels, props, prob_max, scores_cls = self.postprocess_detections(class_logits,
                                                                                             box_regression,
                                                                                             proposals, image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    {
                        "boxes": boxes[i],
                        "labels": labels[i],
                        "scores": scores[i],
                        "props": props[i],
                        "prob_max": prob_max[i],
                        "scores_cls": scores_cls[i],
                    }
                )

        return result, losses