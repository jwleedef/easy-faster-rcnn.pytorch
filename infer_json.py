import argparse
import os
import random
import torch

from PIL import ImageDraw
from torchvision.transforms import transforms
from dataset.base import Base as DatasetBase
from backbone.base import Base as BackboneBase
from bbox import BBox
from model import Model
from roi.pooler import Pooler
from config.eval_config import EvalConfig as Config

import json
from collections import OrderedDict
from datetime import datetime

def _infer(path_to_input_image: str, path_to_output_dir: str, path_to_checkpoint: str, dataset_name: str, backbone_name: str, prob_thresh: float):
    dataset_class = DatasetBase.from_name(dataset_name)
    backbone = BackboneBase.from_name(backbone_name)(pretrained=False)
    model = Model(backbone, dataset_class.num_classes(), pooler_mode=Config.POOLER_MODE,
                  anchor_ratios=Config.ANCHOR_RATIOS, anchor_sizes=Config.ANCHOR_SIZES,
                  rpn_pre_nms_top_n=Config.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=Config.RPN_POST_NMS_TOP_N).cuda()
    model.load(path_to_checkpoint)

    with torch.no_grad():
        image = transforms.Image.open(path_to_input_image)
        image_tensor, scale = dataset_class.preprocess(image, Config.IMAGE_MIN_SIDE, Config.IMAGE_MAX_SIDE)

        detection_bboxes, detection_classes, detection_probs, _ = \
            model.eval().forward(image_tensor.unsqueeze(dim=0).cuda())
        detection_bboxes /= scale

        kept_indices = detection_probs > prob_thresh
        detection_bboxes = detection_bboxes[kept_indices]
        detection_classes = detection_classes[kept_indices]
        detection_probs = detection_probs[kept_indices]

        jsonData = OrderedDict()
        resultData = OrderedDict()
        detectionResultDataList = []

        image_name = path_to_input_image.split('/')[-1]
        frame_num = image_name.split('.')[0].split('_')[-1]

        jsonData["image_path"] = image_name 
        jsonData["modules"] = "Faster_R-CNN_ResNet101"

        jsonData["cam_id"] = "0"
        jsonData["frame_num"] = frame_num

        for bbox, cls, prob in zip(detection_bboxes.tolist(), detection_classes.tolist(), detection_probs.tolist()):
            bbox = BBox(left=bbox[0], top=bbox[1], right=bbox[2], bottom=bbox[3])
            category = dataset_class.LABEL_TO_CATEGORY_DICT[cls]

            detectionResultData = OrderedDict()
            detectionResultData["label"] = [{'description':category, 'score':prob }]
            detectionResultData["position"] = {'x':bbox.left, 'y':bbox.top, 'w':(bbox.right-bbox.left), 'h':(bbox.bottom-bbox.top)}
            detectionResultDataList.append(detectionResultData)

        resultData["module_name"] = "Faster_R-CNN_ResNet101"
        resultData["detection_result"] = detectionResultDataList
        jsonData["results"] = [resultData]

        output_file_path = path_to_output_dir + "/" + datetime.now().strftime("%Y-%m-%d__%H:%M:%S__") + image_name.split('.')[0] + ".json"
        with open('{}'.format(output_file_path), 'w', encoding="utf-8") as make_file:
            json.dump(jsonData, make_file, ensure_ascii=False, indent="\t")


if __name__ == '__main__':
    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument('-s', '--dataset', type=str, choices=DatasetBase.OPTIONS, required=True, help='name of dataset')
        parser.add_argument('-b', '--backbone', type=str, choices=BackboneBase.OPTIONS, required=True, help='name of backbone model')
        parser.add_argument('-c', '--checkpoint', type=str, required=True, help='path to checkpoint')
        parser.add_argument('-p', '--probability_threshold', type=float, default=0.6, help='threshold of detection probability')
        parser.add_argument('--image_min_side', type=float, help='default: {:g}'.format(Config.IMAGE_MIN_SIDE))
        parser.add_argument('--image_max_side', type=float, help='default: {:g}'.format(Config.IMAGE_MAX_SIDE))
        parser.add_argument('--anchor_ratios', type=str, help='default: "{!s}"'.format(Config.ANCHOR_RATIOS))
        parser.add_argument('--anchor_sizes', type=str, help='default: "{!s}"'.format(Config.ANCHOR_SIZES))
        parser.add_argument('--pooler_mode', type=str, choices=Pooler.OPTIONS, help='default: {.value:s}'.format(Config.POOLER_MODE))
        parser.add_argument('--rpn_pre_nms_top_n', type=int, help='default: {:d}'.format(Config.RPN_PRE_NMS_TOP_N))
        parser.add_argument('--rpn_post_nms_top_n', type=int, help='default: {:d}'.format(Config.RPN_POST_NMS_TOP_N))
        parser.add_argument('input', type=str, help='path to input image')
        parser.add_argument('output_directory', type=str, help='directory path to output result json')
        args = parser.parse_args()

        path_to_input_image = args.input
        path_to_output_dir = args.output_directory
        dataset_name = args.dataset
        backbone_name = args.backbone
        path_to_checkpoint = args.checkpoint
        prob_thresh = args.probability_threshold

        os.makedirs(os.path.join(os.path.curdir, os.path.dirname(path_to_output_dir)), exist_ok=True)

        Config.setup(image_min_side=args.image_min_side, image_max_side=args.image_max_side,
                     anchor_ratios=args.anchor_ratios, anchor_sizes=args.anchor_sizes, pooler_mode=args.pooler_mode,
                     rpn_pre_nms_top_n=args.rpn_pre_nms_top_n, rpn_post_nms_top_n=args.rpn_post_nms_top_n)

        print('Arguments:')
        for k, v in vars(args).items():
            print(f'\t{k} = {v}')
        print(Config.describe())

        _infer(path_to_input_image, path_to_output_dir, path_to_checkpoint, dataset_name, backbone_name, prob_thresh)

    main()
