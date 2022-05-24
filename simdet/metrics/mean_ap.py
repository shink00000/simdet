from torchmetrics import Metric
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class BBoxMeanAP(Metric):
    """
    Inputs:
        preds: Tuple[List[Tensor], List[Tensor], List[Tensor]]
            - [bboxes_batch1, bboxes_batch2, ...]
            - [scores_batch1, scores_batch2, ...]
            - [class_ids_batch1, class_ids_batch2, ...]
        metas: Tuple[Dict[str, Any]]
            contains 'image_id'

    Outputs:
        Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = x.xxx
        Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = x.xxx
        Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = x.xxx
        Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = x.xxx
        Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = x.xxx
        Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = x.xxx
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = x.xxx
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = x.xxx
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = x.xxx
        Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = x.xxx
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = x.xxx
        Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = x.xxx
    """

    def __init__(self, anno_path: str, pred_size: list):
        super().__init__()

        self.cocoGt = COCO(anno_path)
        self.h, self.w = pred_size
        self.add_state('coco_dets', default=[], dist_reduce_fx=None)

    def update(self, preds: tuple, metas: tuple):
        pred_bboxes, pred_scores, pred_class_ids = preds

        for batch_id in range(len(metas)):
            bboxes = pred_bboxes[batch_id].cpu().numpy()
            scores = pred_scores[batch_id].cpu().numpy()
            class_ids = pred_class_ids[batch_id].cpu().numpy()
            meta = metas[batch_id]
            h_ratio = meta['height'] / self.h
            w_ratio = meta['width'] / self.w
            dets = [{
                'image_id': meta['image_id'],
                'category_id': class_id,
                'bbox': [
                    xmin * w_ratio,
                    ymin * h_ratio,
                    (xmax - xmin) * w_ratio,
                    (ymax - ymin) * h_ratio],
                'score': score
            } for (xmin, ymin, xmax, ymax), score, class_id in zip(bboxes, scores, class_ids)]
            self.coco_dets.extend(dets)

    def compute(self) -> dict:
        cocoGt = self.cocoGt
        cocoDt = cocoGt.loadRes(self.coco_dets)
        cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        return {
            'mAP@IoU=0.50:0.95': cocoEval.stats[0],
            'mAP@IoU=0.50': cocoEval.stats[1]
        }
