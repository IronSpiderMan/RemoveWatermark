import torch
from yolo.remove_watermark import remove_all
from models.common import DetectMultiBackend

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

if __name__ == '__main__':
    model_path = "best.pt"
    model = DetectMultiBackend(
        weights=model_path,
        device=device,
        dnn=False,
        data='data\coco.yaml',
        fp16=False
    )
    remove_all(model, "videos", "outputs")
