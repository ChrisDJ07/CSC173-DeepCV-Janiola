import cv2
from ultralytics import YOLO

model = YOLO("coin_yolo_training/yolov8n_coins2/weights/best.pt")

values = {0: 10, 1: 1, 2: 20, 3: 5}

def compute_total(result):
    counts = {k: 0 for k in values}
    total = 0
    for cls in result.boxes.cls.tolist():
        cls = int(cls)
        counts[cls] += 1
        total += values[cls]
    return counts, total

cap = cv2.VideoCapture(0)  # 0 = default webcam
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(
        frame,
        imgsz=640,
        conf=0.6,
        iou=0.5,
        verbose=False
    )

    result = results[0]

    annotated = result.plot()

    counts, total = compute_total(result)
    text = f"Total: ₱{total}"
    cv2.putText(annotated, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Coin Detector", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()