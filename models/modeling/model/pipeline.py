from .model import build_model

_GZSL_META_ARCHITECTURES = {
    "model":build_model,
}

def build_gzsl_pipeline(cfg):
    meta_arch = _GZSL_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
    return meta_arch(cfg)