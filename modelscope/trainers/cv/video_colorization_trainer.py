# Copyright (c) Alibaba, Inc. and its affiliates.
import os

from modelscope.metainfo import Trainers
from modelscope.trainers.builder import TRAINERS
from modelscope.trainers.trainer import EpochBasedTrainer
from modelscope.trainers.base import BaseTrainer

from ntire23_scripts.baseline_evaluation import Evaluater


@TRAINERS.register_module(module_name=Trainers.video_colorization)
class VideoColorizationTrainer(BaseTrainer):

    def __init__(self,
                 model: str = None,
                 cfg_file: str = None,
                 cache_path: str = None,
                 *args,
                 **kwargs):
        """ High-level finetune api for Video Colorization.
        Args:
            model: Model id of modelscope models.
            cfg_file: Path to configuration file.
            cache_path: cache path of model files.
        """
        if model is not None:
            self.cache_path = self.get_or_download_model_dir(model)
            if cfg_file is None:
                self.cfg_file = os.path.join(self.cache_path,
                                             ModelFile.CONFIGURATION)
        else:
            assert cfg_file is not None and cache_path is not None, \
                'cfg_file and cache_path is needed, if model is not provided'

        if cfg_file is not None:
            self.cfg_file = cfg_file
            if cache_path is not None:
                self.cache_path = cache_path
        super().__init__(self.cfg_file)

    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)

    def evaluate(self, checkpoint_path=None, saving_fn=None, **kwargs):
        if checkpoint_path is not None:
            self.cfg.test.checkpoint_path = checkpoint_path
        evaluater = Evaluater()
        fid, cdc = evaluater.evaluate()
        print(f'FID: {fid}, CDC: {cdc}')
