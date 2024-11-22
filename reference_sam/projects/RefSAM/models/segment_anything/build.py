from .build_sam import (
    build_hq_sam_vit_h,
    build_hq_sam_vit_l,
    build_hq_sam_vit_b,
    build_hq_sam_vit_t
)

from .build_sam_baseline import (
    build_sam_vit_h,
    build_sam_vit_l,
    build_sam_vit_b,
    build_sam_vit_t
)

sam_registry = {
    'sam_h': build_sam_vit_h,
    'sam_l': build_sam_vit_l,
    'sam_b': build_sam_vit_b,
    'sam_t': build_sam_vit_t,
    'hq_sam_h': build_hq_sam_vit_h,
    'hq_sam_l': build_hq_sam_vit_l,
    'hq_sam_b': build_hq_sam_vit_b,
    'hq_sam_t': build_hq_sam_vit_t
}