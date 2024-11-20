from typing import Optional

import torch
from mmengine.hooks.hook import DATA_BATCH
from mmengine.registry import HOOKS
from mmengine.hooks import Hook

@HOOKS.register_module()
class CheckProcessTimePerBlock(Hook):
    """
    Check process time per block (Backbone, Neck, Head).
    Add to the custom hooks as follows:
    custom_hooks=[
        dict(type='CheckProcessTimePerBlock', interval=50),
    ]

    Args:
        interval (int): Checking interval (every k iterations).
            Defaults to 50.
    """
    def __init__(self, interval=50):
        self.interval = 1
        self.total_img_backbone_time = 0
        self.total_txt_backbone_time = 0
        self.total_neck_time = 0
        self.total_head_time = 0
        self.total_time = 0

        self.nms_time = 0
    def after_test_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        """
        All subclasses should override this method, if they need any
        operations after each training iteration.
        Args:
            runner (Runner): The runner of the training process.
            batch_idx (int): The index of the current batch in the train loop.
            data_batch (dict or tuple or list, optional): Data from dataloader.
            outputs (dict, optional): Outputs from model.
        """
        if self.every_n_inner_iters(batch_idx, self.interval):
            # process time(s) for one batch per gpu at test time
            img_backbone_time = runner.model.module.backbone.img_backbone_time
            text_backbone_time = runner.model.module.backbone.txt_backbone_time
            neck_time = runner.model.module.neck_time
            head_time = runner.model.module.head_time

            nms_time = runner.model.module.bbox_head.nms_time

            runner.logger.info(f'Image Backbone time: {img_backbone_time:.4f}s')
            runner.logger.info(f'Text Backbone time: {text_backbone_time:.4f}s')
            runner.logger.info(f'Neck Time: {neck_time:.4f}s')
            runner.logger.info(f'Head Time: {head_time:.4f}s')
            runner.logger.info(f'NMS Time: {nms_time:.4f}s')


            self.total_img_backbone_time += img_backbone_time
            self.total_txt_backbone_time += text_backbone_time
            self.total_neck_time += neck_time
            self.total_head_time += head_time
            self.total_time += img_backbone_time + text_backbone_time + neck_time + head_time

            self.nms_time += nms_time

    def after_test(self, runner):
        avg_img_backbone_time = self.total_img_backbone_time / len(runner.test_dataloader)
        avg_txt_backbone_time = self.total_txt_backbone_time / len(runner.test_dataloader)
        avg_neck_time = self.total_neck_time / len(runner.test_dataloader)
        avg_head_time = self.total_head_time / len(runner.test_dataloader)
        avg_time = self.total_time / len(runner.test_dataloader)

        avg_nms_time = self.nms_time / len(runner.test_dataloader)

        runner.logger.info(f'Average TOTAL Time: {avg_time:.4f}s')
        runner.logger.info(f'Average Image Backbone time: {avg_img_backbone_time:.4f}s')
        runner.logger.info(f'Average Text Backbone time: {avg_txt_backbone_time:.4f}s')
        runner.logger.info(f'Average Neck Time: {avg_neck_time:.4f}s')
        runner.logger.info(f'Average Head Time: {avg_head_time:.4f}s')
        runner.logger.info(f'Average NMS Time: {avg_nms_time:.4f}s')

        fps = 1 / (avg_time - avg_txt_backbone_time - avg_nms_time)
        runner.logger.info(f'Average FPS: {fps:.4f}')


