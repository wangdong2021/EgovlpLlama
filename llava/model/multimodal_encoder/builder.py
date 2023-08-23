from .clip_encoder import CLIPVisionTower
from .video_encoder.model import VideoEncoder

def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    if vision_tower.startswith("openai") or vision_tower.startswith("laion"):
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')


def build_video_tower(vidoe_tower_cfg, **kwargs):
    return VideoEncoder(vidoe_tower_cfg, **kwargs)
