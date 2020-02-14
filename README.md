# detectron2-helpers
Helpers for detectron2

`DummyAlbuMapper` allows you to use albumentations during fitting detectron2 for detection tasks. 

It requires improvements to work with keypoints and masks.

To use custom dataset_mapper you also need to define new `class Trainer` with modified `build_train_loader`.
Smth like this:
```python3
from detectron2.engine import DefaultTrainer


class Trainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DummyAlbuMapper(cfg, is_train=True)
        return build_detection_train_loader(cfg, mapper=mapper)
```