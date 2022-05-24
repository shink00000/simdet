import torch
import torch.nn as nn
from torchvision.ops import batched_nms


class SingleLabelNMS(nn.Module):
    def __init__(self, min_score: float = 0.01, select_top: int = 200, nms_iou: float = 0.45):
        super().__init__()
        self.min_score = min_score
        self.select_top = select_top
        self.nms_iou = nms_iou

    def forward(self, batched_bboxes: torch.Tensor, batched_scores: torch.Tensor) -> tuple:
        """
        Args:
            batched_bboxes (torch.Tensor): (batch_size, n_preds, 4)
            batched_scores (torch.Tensor): (batch_size, n_preds, n_classes)

        Returns:
            Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
                0: pred_bboxes    [(n_preds_keep, 4), (n_preds_keep, 4), ...]
                1: pred_scores    [(n_preds_keep,), (n_preds_keep,), ...]
                2: pred_class_ids [(n_preds_keep,), (n_preds_keep,), ...]
        """
        batch_size = batched_bboxes.size(0)
        pred_bboxes, pred_scores, pred_class_ids = [], [], []
        for batch_id in range(batch_size):
            bboxes = batched_bboxes[batch_id]
            scores = batched_scores[batch_id]

            # select top k
            max_scores, max_ids = scores.max(dim=1)
            top_scores, top_ids = max_scores.topk(self.select_top, dim=0)
            bboxes = bboxes[top_ids]
            scores = top_scores
            class_ids = max_ids[top_ids] + 1

            # remove less than min_score
            keep = scores > self.min_score
            bboxes = bboxes[keep]
            scores = scores[keep]
            class_ids = class_ids[keep]

            # nms
            keep = batched_nms(bboxes, scores, class_ids, iou_threshold=self.nms_iou)
            bboxes = bboxes[keep]
            scores = scores[keep]
            class_ids = class_ids[keep]

            pred_bboxes.append(bboxes)
            pred_scores.append(scores)
            pred_class_ids.append(class_ids)

        return pred_bboxes, pred_scores, pred_class_ids


class MultiLabelNMS(SingleLabelNMS):
    def forward(self, batched_bboxes: torch.Tensor, batched_scores: torch.Tensor) -> tuple:
        """
        Args:
            batched_bboxes (torch.Tensor): (batch_size, n_preds, 4)
            batched_scores (torch.Tensor): (batch_size, n_preds, n_classes)

        Returns:
            Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
                0: pred_bboxes    [(n_preds_keep, 4), (n_preds_keep, 4), ...]
                1: pred_scores    [(n_preds_keep,), (n_preds_keep,), ...]
                2: pred_class_ids [(n_preds_keep,), (n_preds_keep,), ...]
        """
        batch_size = batched_bboxes.size(0)
        n_classes = batched_scores.size(-1)
        pred_bboxes, pred_scores, pred_class_ids = [], [], []
        for batch_id in range(batch_size):
            bboxes = batched_bboxes[batch_id]
            scores = batched_scores[batch_id]

            # select top k
            top_scores, top_ids = scores.topk(self.select_top, dim=0)
            bboxes = bboxes[top_ids.flatten()]
            scores = top_scores.flatten()
            class_ids = torch.arange(1, n_classes+1, dtype=torch.long, device=bboxes.device).repeat(self.select_top)

            # remove less than min_score
            keep = scores > self.min_score
            bboxes = bboxes[keep]
            scores = scores[keep]
            class_ids = class_ids[keep]

            # nms
            keep = batched_nms(bboxes, scores, class_ids, iou_threshold=self.nms_iou)
            bboxes = bboxes[keep]
            scores = scores[keep]
            class_ids = class_ids[keep]

            pred_bboxes.append(bboxes)
            pred_scores.append(scores)
            pred_class_ids.append(class_ids)

        return pred_bboxes, pred_scores, pred_class_ids
