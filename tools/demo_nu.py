import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import argparse
try:
    import open3d
    from visual_utils import open3d_vis_utils as V
    OPEN3D_FLAG = True
except:
    import mayavi.mlab as mlab
    from visual_utils import visualize_utils as V
    OPEN3D_FLAG = False

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate, build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils



def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    # parser.add_argument('--cfg_file', type=str, default='cfgs/nuscenes_models/cbgs_pp_multihead.yaml',
    parser.add_argument('--cfg_file', type=str, default='cfgs/nuscenes_models/cbgs_dyn_pp_centerpoint.yaml',
    # parser.add_argument('--cfg_file', type=str, default='cfgs/nuscenes_models/cbgs_pillar0075_res2d_centerpoint.yaml',
                        help='specify the config for demo')
    # parser.add_argument('--ckpt', type=str, default="any_pth/pp_multihead_nds5823_updated.pth", help='specify the pretrained model')
    parser.add_argument('--ckpt', type=str, default="any_pth/cbgs_pp_centerpoint_nds6070.pth", help='specify the pretrained model')
    # parser.add_argument('--ckpt', type=str, default="/my/notebook_work/lcy/code_folder/myOpenPCDet_v2/output/nuscenes_models/cbgs_pillar0075_res2d_centerpoint_0513/default/ckpt/checkpoint_epoch_18.pth", help='specify the pretrained model')

    args = parser.parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    train_set, train_loader, train_sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=None, workers=2,
        logger=logger,
        training=False,
        merge_all_iters_to_one_epoch=False,
        total_epochs=None
    )

    logger.info(f'Total number of samples: \t{len(train_set)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    with torch.no_grad():
        for idx, data_dict in enumerate(train_set):
            logger.info(f'Visualized sample index: \t{idx}')
            data_dict = train_set.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)
            gt_boxes = data_dict['gt_boxes']
            gt_boxes = gt_boxes.squeeze(dim=0)

            # if idx % 50 == 0 and idx >= 5050:
            # if idx == 5050 or idx == 5200 or idx == 5350 or idx == 5400 or idx == 5700 or idx == 5850:
            if idx == 5050 or  idx == 5350 or idx == 5400:
            # if idx == 500:
                V.draw_scenes(
                    points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
                    ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels'],
                    gt_boxes=gt_boxes
                )

                if not OPEN3D_FLAG:
                    mlab.show(stop=True)

    logger.info('Demo done.')


if __name__ == '__main__':
    main()