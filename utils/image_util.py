import io

import cv2
import requests
import numpy as np
from PIL import Image


def url2image(url: str) -> np.ndarray:
    capture = cv2.VideoCapture(url)
    _, image = capture.read()
    return image


def url2image2(url: str, url_type='remote'):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36"
    }
    try:
        image = None
        if url_type == 'local':
            image = Image.open(url)
        elif url_type == 'remote':
            resp = requests.get(url, headers=headers)
            image = Image.open(io.BytesIO(resp.content))
        image = np.array(image)
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    except Exception as e:
        return None


def crop_image(image: np.ndarray) -> np.ndarray:
    # 把图片缩放到n×240
    h, w, _ = image.shape
    fx = round(240 / w, 2)
    image = cv2.resize(image, (0, 0), fx=fx, fy=fx)
    nh, nw, _ = image.shape
    # 截取图片中心180×180区域
    x = (nw - 180) // 2
    y = (nh - 180) // 2
    image = Image.fromarray(image)
    image = image.crop((x, y, x + 180, y + 180))
    return np.array(image)


def preprocess(image):
    return image


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


if __name__ == '__main__':
    path = "https://img2.baidu.com/it/u=272103827,29441899&fm=253&fmt=auto&app=138&f=JPEG?w=500&h=625"
    url2image2(path)
    # path = r"C:\Users\86185\Desktop\111.jpg"
    # image = cv2.imread(path)
    # image = crop_image(image)
    # print(image.shape)
    # cv2.imshow("img", image)
    # cv2.waitKey()
