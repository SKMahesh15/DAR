from ultralytics import YOLO

model = YOLO("yolo26s.pt")

model.train(data="datasets/data.yaml",
    epochs=50,
    imgsz=510,
    batch=4,
    device=0
)

