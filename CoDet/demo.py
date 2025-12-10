# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import glob
import multiprocessing as mp
import numpy as np
import json
import os
import tempfile
import time
import warnings
import cv2
import tqdm
import sys
import mss
import torch
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
import itertools
sys.path.insert(0, 'third_party/CenterNet2/')
from centernet.config import add_centernet_config
from codet.config import add_codet_config
from detectron2.evaluation.coco_evaluation import instances_to_coco_json
from codet.predictor import VisualizationDemo_VLDet

# Fake a video capture object OpenCV style - half width, half height of first screen using MSS
class ScreenGrab:
    def __init__(self):
        self.sct = mss.mss()
        m0 = self.sct.monitors[0]
        self.monitor = {'top': 0, 'left': 0, 'width': m0['width'] / 2, 'height': m0['height'] / 2}

    def read(self):
        img =  np.array(self.sct.grab(self.monitor))
        nf = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return (True, nf)

    def isOpened(self):
        return True
    def release(self):
        return True


# constants
WINDOW_NAME = "Detic"

def setup_cfg(args):
    cfg = get_cfg()
    cfg.set_new_allowed(True) # 允许添加自定义参数
    if args.cpu:
        cfg.MODEL.DEVICE="cpu"
    add_centernet_config(cfg)
    add_codet_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    # cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand' # load later
    cfg.SAVE_FEATURES = args.save_features
    if not args.pred_all_class:
        cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", help="Take inputs from webcam.")
    parser.add_argument("--cpu", action='store_true', help="Use CPU only.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--vocabulary",
        default="lvis",
        choices=['lvis', 'openimages', 'objects365', 'coco', 'ovcoco', 'custom', 'visualize'],
        help="different type for ovd evaluation",
    )
    parser.add_argument(
        "--visualize",
        type=bool,
        default=False,
        help="whether save features",
    )
    parser.add_argument(
        "--save-features",
        default='no',
        help="whether save features",
    )
    parser.add_argument(
        "--custom_vocabulary",
        default="",
        help="",
    )
    parser.add_argument("--pred_all_class", action='store_true')
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.3,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--json_output",
        help="A file or directory to save json prediction output",
    )
    return parser



def instances_to_dict(instances):
    """
    将 Detectron2 Instances 对象转换为可序列化的字典。
    """
    instances_dict = {
    }

    for field_name, field_data in instances.get_fields().items():
        # 检查 field_data 是否为张量
        if isinstance(field_data, torch.Tensor):
            instances_dict[field_name] = field_data.cpu().tolist()
        elif hasattr(field_data, 'tensor'):  # 假设 Boxes 或其他对象有一个返回张量的属性或方法
            instances_dict[field_name] = field_data.tensor.cpu().tolist()
        else:
            # 对于其他类型，直接转换成字符串
            instances_dict[field_name] = str(field_data)
    return instances_dict



if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo_VLDet(cfg, args)
    
    predictions_data = []
        # path
        # predictions
    

    if args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input, disable=not args.output):
            # 循环遍历每一个input中的图片路径
            img = read_image(path, format="BGR")
            start_time = time.time()
            predictions, visualized_output = demo.run_on_image(img) #  type(predictions):dict
            # print(predictions["instances"])
            
            
            current_image = {}
            current_image['img_path'] = path
            current_image["image_id"] = int(path.split('/')[-1].replace('.jpg', ''))
            if "instances" in predictions:
                instances = predictions["instances"].to(torch.device("cpu"))
                current_image["instances"] = instances_to_coco_json(instances, current_image["image_id"])
            
            # current_image['predictions'] = instances_to_dict(predictions["instances"])
            predictions_data.append(current_image)
            
            
            logger.info(
                "{}: {} in {:.2f}s".format(
                    path,
                    "detected {} instances".format(len(predictions["instances"]))
                    if "instances" in predictions
                    else "finished",
                    time.time() - start_time,
                )
            )

            if args.output:
                # 如果要保存预测结果
                if os.path.isdir(args.output):
                    # 如果output路径存在，则生成output文件名
                    assert os.path.isdir(args.output), args.output
                    out_filename = os.path.join(args.output, os.path.basename(path))
                else:
                    assert len(args.input) == 1, "Please specify a directory with args.output"
                    out_filename = args.output
                visualized_output.save(out_filename)
            else:
                # 否则则弹出窗口展示预测结果
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                if cv2.waitKey(0) == 27:
                    break  # esc to quit
                
        if args.json_output:
            # 如果要保存预测结果
            # print("predictions_data: \n", predictions_data)
            coco_results = list(itertools.chain(*[x["instances"] for x in predictions_data]))
            # coco_results = list(itertools.chain(*predictions_data))

            with open(args.json_output, 'w') as file:
                json.dump(coco_results, file)
            print("已保存json格式的预测结果")
    
    

