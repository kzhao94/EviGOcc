from __future__ import annotations

import numpy as np


class OccupancyMetricAccumulator:
    """Pooled semantic occupancy metrics used by the paper tables."""

    def __init__(self, class_count: int, occupied_class_ids: tuple[int, ...], empty_label: int):
        if class_count < 2 or empty_label < 0 or empty_label >= class_count:
            raise ValueError("invalid metric taxonomy")
        self.class_count = class_count
        self.occupied_class_ids = np.asarray(occupied_class_ids, dtype=np.int64)
        self.empty_label = empty_label
        self.confusion = np.zeros((class_count, class_count), dtype=np.int64)
        self.completion_tp = 0
        self.completion_fp = 0
        self.completion_fn = 0
        self.samples = 0

    def update(
        self,
        prediction: np.ndarray,
        ground_truth: np.ndarray,
        valid_mask: np.ndarray | None = None,
        ignore_label: int = 255,
    ) -> None:
        prediction = np.asarray(prediction)
        ground_truth = np.asarray(ground_truth)
        if prediction.shape != ground_truth.shape:
            raise ValueError("prediction and ground truth must share shape")
        valid = ground_truth != ignore_label
        if valid_mask is not None:
            valid &= np.asarray(valid_mask, dtype=bool)
        pred = prediction[valid].astype(np.int64)
        target = ground_truth[valid].astype(np.int64)
        if ((pred < 0) | (pred >= self.class_count)).any():
            raise ValueError("prediction contains labels outside the taxonomy")
        target_occ = np.isin(target, self.occupied_class_ids)
        pred_occ = np.isin(pred, self.occupied_class_ids)
        self.completion_tp += int((target_occ & pred_occ).sum())
        self.completion_fp += int((~target_occ & pred_occ).sum())
        self.completion_fn += int((target_occ & ~pred_occ).sum())
        selected = (target >= 0) & (target < self.class_count)
        self.confusion += np.bincount(
            self.class_count * target[selected] + pred[selected],
            minlength=self.class_count**2,
        ).reshape(self.class_count, self.class_count)
        self.samples += 1

    def compute(self) -> dict[str, float | int | list[float]]:
        true_positive = np.diag(self.confusion)
        denominator = self.confusion.sum(axis=0) + self.confusion.sum(axis=1) - true_positive
        class_iou = np.divide(
            true_positive,
            denominator,
            out=np.zeros(self.class_count, dtype=np.float64),
            where=denominator > 0,
        )
        completion_denominator = self.completion_tp + self.completion_fp + self.completion_fn
        return {
            "samples": self.samples,
            "IoU": self.completion_tp / max(completion_denominator, 1),
            "mIoU": float(class_iou[self.occupied_class_ids].mean()),
            "per_class_IoU": class_iou.tolist(),
        }
