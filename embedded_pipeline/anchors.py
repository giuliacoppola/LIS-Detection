# anchors.py
# Generatore di anchors in stile MediaPipe/SSD per BlazePalm @256
# NOTA: preset "blazepalm_256" produce 2944 anchor (pattern per-layer [2,3,2] su feature map 32x32,16x16,8x8)
# Se i box risultano sballati, prova preset_alternative=True per [2,2,2,2] su stride [8,16,32,64] (2720 anc.)
# oppure regola min/max_scale.

from dataclasses import dataclass
import numpy as np

@dataclass
class AnchorOptions:
    input_size_width: int = 256
    input_size_height: int = 256
    strides: tuple = (8, 16, 32)
    min_scale: float = 0.1484375  # ~38/256, vicino ai preset MediaPipe
    max_scale: float = 0.75
    anchor_offset_x: float = 0.5
    anchor_offset_y: float = 0.5
    interpolated_scale_aspect_ratio: float = 1.0  # abilita l'anchor "extra" per layer
    fixed_anchor_size: bool = True


def _scale_for_layer(min_scale, max_scale, layer_id, num_layers):
    if num_layers == 1:
        return (min_scale + max_scale) * 0.5
    return min_scale + (max_scale - min_scale) * layer_id / (num_layers - 1)


def generate_anchors(opts: AnchorOptions, per_layer_extra=(0,1,0)):
    """
    Genera anchors SSD:
    - per_layer_extra controlla quante anchor "interpolate" aggiungere per layer (0 o 1):
      impostato a (0,1,0) => pattern [2,3,2] con aspect ratio 1.0 e l'interpolated_scale.
    Ritorna: ndarray [N,4] con (x_center, y_center, w, h) normalizzati [0,1].
    """
    num_layers = len(opts.strides)
    anchors = []
    for layer_id, stride in enumerate(opts.strides):
        scale = _scale_for_layer(opts.min_scale, opts.max_scale, layer_id, num_layers)
        next_scale = _scale_for_layer(opts.min_scale, opts.max_scale, layer_id + 1, num_layers) if layer_id != num_layers - 1 else 1.0
        fm_w = int(np.ceil(opts.input_size_width / stride))
        fm_h = int(np.ceil(opts.input_size_height / stride))

        # Sempre 1 aspect ratio (1.0). Due scale base (scale e next_scale) => 2 anchors/location.
        scales = [scale, next_scale]
        # Eventuale anchor extra interpolata
        if per_layer_extra[layer_id] == 1 and opts.interpolated_scale_aspect_ratio > 0:
            scales.append(np.sqrt(scale * next_scale))

        for y in range(fm_h):
            for x in range(fm_w):
                x_center = (x + opts.anchor_offset_x) / fm_w
                y_center = (y + opts.anchor_offset_y) / fm_h
                for sc in scales:
                    if opts.fixed_anchor_size:
                        w = h = sc
                    else:
                        w = sc
                        h = sc
                    anchors.append([x_center, y_center, w, h])
    return np.array(anchors, dtype=np.float32)


def blazepalm_256(preset_alternative=False):
    # Preset principale: 3 layer [8,16,32] con extra su layer centrale -> 2944 anchors
    if not preset_alternative:
        opts = AnchorOptions(strides=(8,16,32))
        anchors = generate_anchors(opts, per_layer_extra=(0,1,0))
    else:
        # Alternativa: 4 layer uniformi (8,16,32,64) senza extra (2720 anchors). Manteniamo per debug.
        opts = AnchorOptions(strides=(8,16,32,64))
        anchors = generate_anchors(opts, per_layer_extra=(0,0,0,0))
    return anchors


if __name__ == "__main__":
    a = blazepalm_256()
    print("Anchors:", a.shape)