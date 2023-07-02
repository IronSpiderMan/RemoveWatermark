import os
import subprocess


def remove_watermark(input_file, output_file, box):
    x, y, w, h = box
    # 使用FFmpeg去除视频中的水印
    command = ['ffmpeg', '-i', input_file, '-vf', f'delogo=x={x}:y={y}:w={w}:h={h}:show=0', '-c:a', 'copy', output_file]
    subprocess.call(command)


def remove_batch(basedir, box):
    savebase = "outputs"
    if not os.path.exists(savebase):
        os.mkdir(savebase)
    for cpt in os.listdir(basedir):
        cpt_dir = os.path.join(basedir, cpt)
        savecpt = os.path.join(savebase, cpt)
        if not os.path.exists(savecpt):
            os.mkdir(savecpt)
        for file in os.listdir(cpt_dir):
            filepath = os.path.join(cpt_dir, file)
            savepath = os.path.join(savebase, cpt, file)
            remove_watermark(filepath, savepath, box)


if __name__ == '__main__':
    x1, y1 = 1910, 1150
    x2, y2 = 2208, 1295
    x = x1
    y = y1
    w = x2 - x1
    h = y2 - y1
    remove_batch('videos', (x, y, w, h))
