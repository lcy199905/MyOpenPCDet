import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import argparse
import glob
from pathlib import Path

try:
    import open3d
    from visual_utils import open3d_vis_utils as V
    OPEN3D_FLAG = True
except:
    import mayavi.mlab as mlab
    from visual_utils import visualize_utils as V
    OPEN3D_FLAG = False

# import open3d
# from visual_utils import open3d_vis_utils as V
#
# # import mayavi.mlab as mlab
# # from visual_utils import visualize_utils as V
#
# OPEN3D_FLAG = False


import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
import pickle



class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/voxel_rdiou_3cat.yaml',
    # parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/pillarnet_rdiou_3cat.yaml',
    # parser.add_argument('--cfg_file', type=str, default='/my/notebook_work/lcy/code_folder/myOpenPCDet_v2/tools/cfgs/kitti_models/pillarnet-origin.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='/my/notebook_work/lcy/code_folder/myOpenPCDet_v2/data/kitti/training/velodyne/007462.bin',
    # parser.add_argument('--data_path', type=str, default='/my/notebook_work/lcy/code_folder/myOpenPCDet_v2/data/kitti/training/velodyne/000019.bin',
                        help='specify the point cloud data file or directory')
    # parser.add_argument('--idx', type=str, default='007462', help='bin file idx')
    parser.add_argument('--idx', type=str, default='007462', help='bin file idx')
    parser.add_argument('--ckpt', type=str, default='/my/notebook_work/lcy/code_folder/myOpenPCDet_v2/output/kitti_models/voxel_rdiou_3cat_0504/default/ckpt/checkpoint_epoch_100.pth', help='specify the pretrained model')

    # parser.add_argument('--ckpt', type=str, default='/my/notebook_work/lcy/code_folder/myOpenPCDet_v2/output/kitti_models/pillarnet_rdiou_3cat_best2/default/ckpt/checkpoint_epoch_100.pth', help='specify the pretrained model')
    # parser.add_argument('--ckpt', type=str, default='/my/notebook_work/lcy/code_folder/myOpenPCDet_v2/output/kitti_models/pillarnet-origin/default/ckpt/checkpoint_epoch_100.pth', help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')
    parser.add_argument('--pkl', type=str, default='/my/notebook_work/lcy/code_folder/myOpenPCDet_v2/data/kitti/kitti_infos_val.pkl', help='train/val pkl')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def get_gt_box(info_path='', idx='0'):
    # 格式化为6位
    # idx = "%06d" % idx
    with open(info_path, 'rb') as f:
        infos = pickle.load(f)

    print('find: ', idx)
    for k in range(len(infos)):
        info = infos[k]
        sample_idx = info['point_cloud']['lidar_idx']
        if sample_idx == idx:
            print('gt_database sample: %d/%d' % (k + 1, len(infos)))
            annos = info['annos']
            gt_boxes = annos['gt_boxes_lidar']
            return gt_boxes
    with open('/my/notebook_work/lcy/code_folder/myOpenPCDet_v2/data/kitti/kitti_infos_train.pkl', 'rb') as f:
        infos = pickle.load(f)
    print('find: ', idx)
    for k in range(len(infos)):
        info = infos[k]
        sample_idx = info['point_cloud']['lidar_idx']
        if sample_idx == idx:
            print('gt_database sample: %d/%d' % (k + 1, len(infos)))
            annos = info['annos']
            gt_boxes = annos['gt_boxes_lidar']
            return gt_boxes

def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)

            # 读取gt_box
            gt_box = get_gt_box(info_path=args.pkl, idx=args.idx)

            V.draw_scenes(
                points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
                ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels'],
                gt_boxes=gt_box
            )

            if not OPEN3D_FLAG:
                mlab.show(stop=True)

    logger.info('Demo done.')


if __name__ == '__main__':
    main()
