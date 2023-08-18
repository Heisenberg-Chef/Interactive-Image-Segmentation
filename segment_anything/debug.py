from segment_anything.modeling import PromptEncoder,MaskDecoder,Sam
from segment_anything.modeling.transformer import TwoWayTransformer,TwoWayAttentionBlock
from segment_anything import SamPredictor
import torch
import numpy as np

if __name__ == '__main__':
    from segment_anything import sam_model_registry

    sam_checkpoint = "sam_vit_b_01ec64.pth"
    model_type = "vit_b"
    input_boxes = torch.tensor([
        [75, 275, 1725, 850],
        # [425, 600, 700, 875],
        # [1375, 550, 1650, 800],
        # [1240, 675, 1400, 750],
    ], device="cuda")


    sam = sam_model_registry[model_type](checkpoint="../" + sam_checkpoint)
    sam.to(device="cuda")

    predictor = SamPredictor(sam)

    transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, (1024, 1024))
    # H W C
    predictor.set_image(np.zeros((1280, 1920, 3), dtype=np.uint8))
    # --- 如果按照如下的传入prompt，我们会得到5 + 4 = 9 个 token
    input_point = np.array([[500, 375],[100,100]]) # 如果不输入box 那么就会在最后一位填上0
    input_label = np.array([1,0])
    input_box = np.array([75, 275, 1725, 850])
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        box=input_box,
        multimask_output=True,
    )
    print("MASK: ----------")
    print(masks)
    print("SCORES: ----------")
    print(scores)
    print("LOGITS: ----------")
    print(logits)
