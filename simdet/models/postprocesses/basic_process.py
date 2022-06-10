import torch
import torch.nn as nn


class SingleLabelBasicProcess(nn.Module):
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

            scores, class_ids = scores.max(dim=1)
            class_ids += 1

            pred_bboxes.append(bboxes)
            pred_scores.append(scores)
            pred_class_ids.append(class_ids)

        return pred_bboxes, pred_scores, pred_class_ids


class MultiLabelBasicProcess(nn.Module):
    def __init__(self, min_score: float = 0.01):
        super().__init__()
        self.min_score = min_score

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
        n_preds = batched_bboxes.size(1)
        n_classes = batched_scores.size(-1)
        pred_bboxes, pred_scores, pred_class_ids = [], [], []
        for batch_id in range(batch_size):
            bboxes = batched_bboxes[batch_id]
            scores = batched_scores[batch_id]

            bboxes = bboxes.unsqueeze(1).repeat(1, n_classes, 1)
            class_ids = torch.arange(
                1, n_classes+1, dtype=torch.long, device=bboxes.device
            ).unsqueeze(0).repeat(n_preds, 1)

            # remove less than min_score
            keep = scores > self.min_score
            bboxes = bboxes[keep]
            scores = scores[keep]
            class_ids = class_ids[keep]

            pred_bboxes.append(bboxes)
            pred_scores.append(scores)
            pred_class_ids.append(class_ids)

        return pred_bboxes, pred_scores, pred_class_ids

    def extra_repr(self) -> str:
        return f'min_score={self.min_score}'
