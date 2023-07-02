import os
import re

import cv2
import torch
from tqdm import tqdm
import numpy as np
import moviepy.editor as mp
from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_boxes

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


class WatermarkRemoverCV:
    def __init__(self, video_path, save_path, model, watermark_path=None, background_color=(255, 255, 255)):
        self.video_path = video_path
        self.save_path = save_path
        self.model = model
        self.watermark = None
        if watermark_path:
            self.watermark = cv2.imread(watermark_path)
        self.background_color = background_color
        self.old_rect = None
        self.sift = cv2.SIFT_create()
        index_params = dict(algorithm=0, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

    @staticmethod
    def preprocess(image, auto=True):
        """
        把图片处理成网络需要的形式
        :param image: numpy数组
        :param auto:
        :return:
        """
        # 转换
        image = letterbox(image, 640, stride=32, auto=auto)[0]
        image = image.transpose((2, 0, 1))[::-1]
        image = np.ascontiguousarray(image)
        image = torch.from_numpy(image.astype(np.float32))
        image /= 255.0
        if len(image.shape) == 3:
            image = image[None]
        return image

    def locate_watermark_shift(self, image, watermark, expend=0, stabilize=True):
        """
        使用sift特征匹配，获取watermark在image中的位置
        :param image:
        :param watermark:
        :param expend:
        :param stabilize:
        :return:
        """
        # Convert the logo to grayscale
        watermark_gray = cv2.cvtColor(watermark, cv2.COLOR_BGR2GRAY)
        # Find the keypoints and descriptors for the logo
        kp_watermark, des_watermark = self.sift.detectAndCompute(watermark_gray, None)
        # Find the keypoints and descriptors for the image
        kp_image, des_image = self.sift.detectAndCompute(image, None)
        matches = self.flann.knnMatch(des_watermark, des_image, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        MIN_MATCH_COUNT = 10
        if len(good_matches) >= MIN_MATCH_COUNT:
            src_pts = np.float32([kp_watermark[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_image[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            h, w = watermark_gray.shape
            logo_corners = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
            dst_corners = cv2.perspectiveTransform(logo_corners, M)
            x, y, w, h = cv2.boundingRect(dst_corners)
            x1, y1 = x - expend, y - expend
            x2, y2 = x + w + expend, y + h + expend
            if not self.old_rect or not stabilize:
                self.old_rect = (x1 * 5, y1 * 5, x2 * 5, y2 * 5)
                return self.old_rect
            else:
                self.old_rect = self.stabilization(self.old_rect, (x1 * 5, y1 * 5, x2 * 5, y2 * 5))
                return self.old_rect
        else:
            return

    def locate_watermark(self, image, expend=15, stabilize=True):
        """
        定位视频中的水印位置
        :param stabilize: 是否防抖
        :param expend: 扩大定位框
        :param image: numpy数组
        :return: 是否有水印, 水印位置
        """
        image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
        origin = image.copy()
        image = self.preprocess(image).to(device)
        pred = self.model(image, augment=False)
        pred = non_max_suppression(pred, 0.25, 0.25, None, False, max_det=1000)[0]
        pred[:, :4] = scale_boxes(image.shape[2:], pred[:, :4], origin.shape).round()
        try:
            pred = pred[0].cpu().tolist()
            *xyxy, conf, _ = map(int, pred)
            x1, y1, x2, y2 = map(lambda x: x * 2, xyxy)
            x1, y1 = x1 - expend, y1 - expend
            x2, y2 = x2 + expend, y2 + expend
            if not self.old_rect or not stabilize:
                self.old_rect = (x1, y1, x2, y2)
                return self.old_rect
            else:
                self.old_rect = self.stabilization(self.old_rect, (x1, y1, x2, y2))
                return self.old_rect
        except Exception as e:
            return None

    @staticmethod
    def stabilization(old, new, alpha=0.8):
        """
        防抖，当old和new的iou大于alpha时，使用old，否则使用new
        :param old:
        :param new:
        :param alpha:
        :return:
        """
        # 计算iou
        x1 = max(old[0], new[0])
        y1 = max(old[1], new[1])
        x2 = min(old[0] + old[2], new[0] + new[2])
        y2 = min(old[1] + old[3], new[1] + new[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = old[2] * old[3]
        area_b = new[2] * new[3]
        union_area = area_a + area_b - intersection
        if intersection / union_area < alpha:
            return new
        else:
            return old

    def remove_watermark_frame(self, frame):
        """
        去除图片水印
        :param frame: numpy数组
        :return: 去除水印后的numpy数组
        """
        if isinstance(self.watermark, np.ndarray):
            box = self.locate_watermark_shift(frame, self.watermark)
        else:
            box = self.locate_watermark(frame)
        if box:
            x1, y1, x2, y2 = box
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), self.background_color, -1)
        return frame

    def remove_watermark_video(self):
        """
        去除视频水印
        :return:
        """
        video_capture = cv2.VideoCapture(self.video_path)
        # 获取视频信息
        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # 创建写入对象
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(self.save_path, fourcc, fps, (width, height))
        # 进度条
        bar = tqdm(total=total_frames)
        bar.pandas(desc="正在去除水印")
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break
            # 图片去水印
            frame = self.remove_watermark_frame(frame)
            bar.update(1)
            writer.write(frame)
        video_capture.release()
        writer.release()

    def add_audio(self):
        """
        给视频添加音频
        :return:
        """
        video = mp.VideoFileClip(self.video_path)
        audio = video.audio
        output = mp.VideoFileClip(self.save_path)
        output = output.set_audio(audio)
        new_save_path = re.sub(r'\.mp4$', '_tmp.mp4', self.save_path)
        output.write_videofile(new_save_path, audio_codec='aac')
        video.close()
        output.close()
        os.remove(self.save_path)
        os.rename(new_save_path, self.save_path)


def remove_all(model, basepath, savepath):
    for root, dirs, files in os.walk(basepath):
        for file in files:
            video_path = os.path.join(root, file)
            if not os.path.exists(root.replace(basepath, savepath)):
                os.makedirs(root.replace(basepath, savepath))
            print(f"正在处理：{video_path}")
            output_path = video_path.replace(basepath, savepath)
            remover = WatermarkRemoverCV(video_path, output_path, model)
            remover.remove_watermark_video()
            remover.add_audio()


if __name__ == '__main__':
    model_path = "../best.pt"
    model = DetectMultiBackend(
        weights=model_path,
        device=device,
        dnn=False,
        data='data\coco.yaml',
        fp16=False
    )
    remove_all(model, "../videos", "outputs")
