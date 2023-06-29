from io import BytesIO
from logging import info, warn
from typing_extensions import override
from mitmproxy import ctx
from os import listdir
from os.path import basename
from PIL import Image, ImageDraw
from ultralytics import YOLO
from ultralytics.yolo.utils import set_logging
from urllib.parse import urlparse

import pillow_avif
import random

set_logging("ultralytics", False)

class MIMIMITM:
    def __init__(self):
        self.model = None

    def load(self, loader):
        loader.add_option("weights", str, "weights.pt", "weights path (.pt)")
        loader.add_option("debug", bool, False, "enable debug")
        loader.add_option("imgsize", int, 640, "inference img size")
        loader.add_option("conf", float, 0.5, "confidence threshold")
        loader.add_option("iou", float, 0.5, "IOU threshold")
        loader.add_option("dev", str, "cpu", "inference device")
        loader.add_option("augment", bool, True, "inference device")

    def running(self):
        try:
            self.model = YOLO(ctx.options.weights)
            self.overlays = [Image.open(f"overlays/{name}") \
                for name in listdir("overlays")]
            info("MIMIMITM is running!")
        except:
            addons.clear()
            raise

    def response(self, flow):
        content_type = flow.response.headers.get("content-type", "")

        info(f"TYPE {content_type}")

        if "image/" not in content_type:
            return

        if ctx.options.debug:
            warn("RUNNING")

        try:
            io = BytesIO(flow.response.content)
            io.name = basename(urlparse(flow.request.url).path)
            img = Image.open(io, formats=None)
        except:
            raise Exception(f"load img of content type {content_type}")

        results = self.model.predict(source=img, imgsz=ctx.options.imgsize,
            conf=ctx.options.conf, iou=ctx.options.iou,
            augment=ctx.options.augment, device=ctx.options.dev)
        result = results[0].cpu().numpy()

        draw = ImageDraw.Draw(img)
        if ctx.options.debug:
            warn(f"FOUND {len(result.boxes)}")
        for box in result.boxes:
            x1, y1, x2, y2 = [int(v + 0.5) for v in box.xyxy[0]]
            if ctx.options.debug:
                conf = box.conf[0]
                info(f"BOX ({conf}) {x1} {y1} {x2} {y2}")
                draw.line((x1, y1, x2, y1), fill=(255, 0, 0), width=2)
                draw.line((x2, y1, x2, y2), fill=(255, 0, 0), width=2)
                draw.line((x2, y2, x1, y2), fill=(255, 0, 0), width=2)
                draw.line((x1, y2, x1, y1), fill=(255, 0, 0), width=2)
            w, h = abs(x1 - x2), abs(y1 - y2)
            if w < h / 2: # facing away approximator
                continue
            y1, y2 = (y1 - h / 2), (y2 - h / 2)
            bx, by = (int(x1 - w / 2), int(y1 - h / 2))
            overlay = random.choice(self.overlays).resize((w * 2, h * 2))
            img.paste(overlay, box=(bx, by), mask=overlay.split()[-1])

        io = BytesIO()
        img.save(io, format=img.format)
        flow.response.content = io.getvalue()

addons = [
    MIMIMITM()
]
