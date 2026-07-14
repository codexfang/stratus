from __future__ import annotations
import logging
from pathlib import Path
from typing import Any
import numpy as np
from ultralytics import YOLO

from stratus.core.classifier import Classifier

logger = logging.getLogger(__name__)

GRADE_A = {"cup", "marker", "pen", "bottle", "phone", "keyboard", "mouse", "cable", "box", "toy"}
GRADE_C = {"broken", "cracked", "damaged", "rust", "burn", "scratched"}


class YOLOClassifier(Classifier):
    def __init__(self, model_path: str = "models/best.pt",
                 conf_threshold: float = 0.5):
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path.resolve()}")
        self._model = YOLO(str(path))
        self._conf = conf_threshold
        logger.info("YOLO classifier loaded from %s (%d classes)",
                    path, len(self._model.names))

    def classify(self, image: np.ndarray, **kwargs: Any) -> dict[str, Any]:
        if not isinstance(image, np.ndarray) or image.size == 0 or image.ndim != 3:
            logger.warning("Invalid image for YOLO: type=%s shape=%s", type(image).__name__,
                          getattr(image, 'shape', 'N/A'))
            return {"labels": [], "grade": "B", "bin": "B"}

        img = image[..., :3] if image.shape[2] > 3 else image

        results = self._model(img, conf=self._conf, verbose=False)[0]
        labels = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = results.names[cls_id]
            labels.append({"label": label, "confidence": conf})

        if not labels:
            return {"labels": [], "grade": "B", "bin": "B"}

        top = max(labels, key=lambda x: x["confidence"])
        grade = "A" if top["label"] in GRADE_A else \
                "C" if top["label"] in GRADE_C else "B"
        bin_ = grade

        return {
            "labels": labels,
            "grade": grade,
            "bin": bin_,
        }
