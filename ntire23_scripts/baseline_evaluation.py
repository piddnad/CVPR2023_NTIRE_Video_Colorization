# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path as osp

import cv2
import glob
import json
import os
import shutil
import time
import numpy as np
import tqdm

from modelscope.metainfo import Trainers
from modelscope.msdatasets import MsDataset
from modelscope.pipelines import pipeline
from modelscope.outputs import OutputKeys
from modelscope.trainers import build_trainer
from modelscope.utils.constant import Tasks
from modelscope.utils.cv.image_utils import draw_face_detection_result
from modelscope.utils.demo_utils import DemoCompatibilityCheck
from modelscope.utils.test_utils import test_level
from modelscope.utils.constant import DownloadMode

from fid import calculate_fid
from cdc import calculate_cdc


class Evaluater(object):

    def __init__(self):
        self.task = Tasks.image_colorization
        self.model_id = 'damo/cv_unet_video-colorization'
        cache_dir = 'datasets'
        val_set = MsDataset.load('ntire23_video_colorization', namespace='damo', subset_name='val_frames', split='validation', cache_dir=cache_dir)
        assert val_set is not None, 'val set should be downloaded first'
        self.dataset_dir = 'datasets/damo/ntire23_video_colorization/master/data_files/extracted'

    def evaluate(self):

        frame_paths = glob.glob(f'{self.dataset_dir}/*/val/*/*.png')
        frame_paths.sort()

        colorizer = pipeline(task=self.task, model=self.model_id)
        output_dir = 'results'

        for img_path in tqdm.tqdm(frame_paths):
            result = colorizer(img_path)
            video_name = img_path.split('/')[-2]
            frame_name = img_path.split('/')[-1]

            if result is not None:
                out_video_path = os.path.join(output_dir, video_name)
                os.makedirs(out_video_path, exist_ok=True)
                result_path = os.path.join(out_video_path, frame_name)
                cv2.imwrite(result_path, result[OutputKeys.OUTPUT_IMG])
                # print(f'Output written to {result_path}.')

        t1 = time.time()
        print(f'Calculating FID...')
        fid = calculate_fid(output_dir)
        t2 = time.time()
        print('Calculating CDC...')
        cdc = calculate_cdc(output_dir)
        t3 = time.time()
        print('FID evaluation time:', t2-t1)
        print('CDC evaluation time:', t3-t2)
        print('Total evaluation time:', t3-t1)

        return fid, cdc

if __name__ == '__main__':
    data_root_dir = '/path/to/trainingset' # 下载的数据集路径
    model_id = 'damo/cv_unet_video-colorization'
    cache_path = os.path.expanduser('~/.cache/modelscope/hub/damo/cv_unet_video-colorization/')
    cfg_file = os.path.join(cache_path ,'configuration.json')

    # copy config
    os.makedirs(cache_path, exist_ok=True)
    shutil.copy2('ntire23_scripts/configuration.json', cfg_file)

    kwargs = dict(
        cfg_file=cfg_file,
        model=model_id,
        train_image_dir=data_root_dir,
        val_image_dir=data_root_dir,
    )

    trainer = build_trainer(name=Trainers.video_colorization, default_args=kwargs)
    trainer.evaluate()
