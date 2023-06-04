import torchvision.models as models
import torch
from nets_640_tiny.yolo_yuan_evc import YoloBody
from ptflops import get_model_complexity_info

with torch.cuda.device(0):
    # anchors_path = 'model_data/yolo_anchors.txt'
    # anchors_mask = [[3, 4, 5], [1, 2, 3]]
    model = YoloBody(15, 'tiny')

    # anchors_path = 'model_data/yolo_anchors_v5.txt'
    # anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    # model = YoloBody(anchors_mask, 15)
    # model = YoloBody(anchors_mask, 15, "s", "cspdarknet")

    flops, params = get_model_complexity_info(model, (3, 1024, 1024), as_strings=True,
                                              print_per_layer_stat=True)  # 不用写batch_size大小，默认batch_size=1
    print('Flops:  ' + flops)
    print('Params: ' + params)
