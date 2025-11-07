from jetson_inference import detectNet
from jetson_utils import videoSource, videoOutput
import jetson.inference
import jetson.utils

net = detectNet("ssd-mobilenet-v2", threshold=0.5)
camera = videoSource("/dev/video0")
display = videoOutput("display://0")

while display.IsStreaming():
    img = camera.Capture()
    if img is None:
        continue

    detections = net.Detect(img)

    if detections:
        print("="*50)
        for idx, detection in enumerate(detections, 1):
            class_id = detection.ClassID
            confidence = detection.Confidence
            left = detection.Left
            top = detection.Top
            right = detection.Right
            bottom = detection.Bottom

            width = right - left
            height = bottom - top
            area = width * height
            center_x = (left + right) / 2
            center_y = (top + bottom) / 2

            print(f"【Object {idx}】")
            print(f"ClassID：{class_id}")
            print(f"Confidence：{confidence:.2f}")
            print(f"box（Left/Top/Right/Bottom）：{left:.1f} / {top:.1f} / {right:.1f} / {bottom:.1f}")
            print(f"width/height：{width:.1f} / {height:.1f}")
            print(f"area：{area:.1f}")
            print(f"center：({center_x:.1f}, {center_y:.1f})")
            print("-"*30)

    display.Render(img)
    display.SetStatus("Object Detection | FPS: {:.0f}".format(net.GetNetworkFPS()))
