"""
Model definitions and architectures for the IoU Improvement Pipeline.
"""

from .backbone import (
    load_dinov2_backbone,
    get_backbone_info,
    compute_patch_grid_size,
    align_image_size_to_patches,
    extract_patch_tokens,
    list_available_backbones,
    DINOV2_VARIANTS
)

from .segmentation_head import (
    SegmentationHeadConvNeXt,
    SegmentationHeadConvNeXtDeep,
    SegmentationHeadWithDeepSupervision,
    create_segmentation_head
)

__all__ = [
    # Backbone utilities
    'load_dinov2_backbone',
    'get_backbone_info',
    'compute_patch_grid_size',
    'align_image_size_to_patches',
    'extract_patch_tokens',
    'list_available_backbones',
    'DINOV2_VARIANTS',
    # Segmentation heads
    'SegmentationHeadConvNeXt',
    'SegmentationHeadConvNeXtDeep',
    'SegmentationHeadWithDeepSupervision',
    'create_segmentation_head'
]
