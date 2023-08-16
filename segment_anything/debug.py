from segment_anything.modeling import PromptEncoder,MaskDecoder,Sam
from segment_anything.modeling.transformer import TwoWayTransformer,TwoWayAttentionBlock
from segment_anything import SamPredictor
import torch
import numpy as np

if __name__ == '__main__':
    from segment_anything import sam_model_registry

    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    input_boxes = torch.tensor([
        [75, 275, 1725, 850],
        [425, 600, 700, 875],
        [1375, 550, 1650, 800],
        [1240, 675, 1400, 750],
    ], device="cuda")
    input_point = np.array([[500, 375]])
    input_label = np.array([1])

    sam = sam_model_registry[model_type](checkpoint="../" + sam_checkpoint)
    sam.to(device="cuda")

    predictor = SamPredictor(sam)

    transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, (1024, 1024))
    # H W C
    predictor.set_image(np.zeros((1280, 1920, 3), dtype=np.uint8))

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        box=None,
        multimask_output=True,
    )
    print("MASK: ----------")
    print(masks)
    print("SCORES: ----------")
    print(scores)
    print("LOGITS: ----------")
    print(logits)
