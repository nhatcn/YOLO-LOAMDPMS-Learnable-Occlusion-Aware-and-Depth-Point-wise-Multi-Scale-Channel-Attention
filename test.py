import torch
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.torch_utils import de_parallel
yaml_path = r'D:\yolov8LOAMDPMS\ultralytics\cfg\models\v8\yolov8_DPMS.yaml'
model = DetectionModel(cfg=yaml_path)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"✅ Tổng số tham số: {total_params:,}")
print(f"🧠 Trong đó có thể train được: {trainable_params:,}")
