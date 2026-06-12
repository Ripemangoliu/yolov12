from ultralytics import YOLO

model = YOLO("yolov12n_dg_hva2_area8_res.yaml")
model.load("yolov12n.pt")

model.train(
    data="CDLA.yaml",
    imgsz=640,
    epochs=300,
    batch=32,
    device=0,
    workers=8,
    project="runs_cdla",
    name="yolov12n_dg_hva2_area8_res_640_b32",
    exist_ok=False
)