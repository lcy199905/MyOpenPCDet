from .anchor_head_multi import AnchorHeadMulti
from .anchor_head_single import AnchorHeadSingle
from .anchor_head_template import AnchorHeadTemplate
from .point_head_box import PointHeadBox
from .point_head_simple import PointHeadSimple
from .point_intra_part_head import PointIntraPartOffsetHead
from .center_head import CenterHead
from .voxelnext_head import VoxelNeXtHead
from .transfusion_head import TransFusionHead
from .anchor_head_rdiou_3cat import AnchorHeadRDIoU_3CAT

__all__ = {
    'AnchorHeadTemplate': AnchorHeadTemplate,
    'AnchorHeadSingle': AnchorHeadSingle,
    'PointIntraPartOffsetHead': PointIntraPartOffsetHead,
    'PointHeadSimple': PointHeadSimple,
    'PointHeadBox': PointHeadBox,
    'AnchorHeadMulti': AnchorHeadMulti,
    'CenterHead': CenterHead,
    # 'CenterHeadSingle': CenterHeadSingle,
    'VoxelNeXtHead': VoxelNeXtHead,
    'TransFusionHead': TransFusionHead,
    'AnchorHeadRDIoU_3CAT': AnchorHeadRDIoU_3CAT
}