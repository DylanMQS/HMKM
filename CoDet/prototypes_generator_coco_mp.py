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
from tqdm import tqdm
import sys
import mss
import torch
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
import itertools

import sys
sys.path.insert(0, 'third_party/CenterNet2/')
# sys.path.insert(0, 'CenterNet2/projects/CenterNet2/')
# from centernet.config import add_centernet_config
# from vldet.config import add_vldet_config
# from vldet.evaluation.coco_evaluation import instances_to_coco_json

from codet.predictor import VisualizationDemo_VLDet
from centernet.config import add_centernet_config

from codet.config import add_codet_config

sys.path.insert(0, 'third_party/Deformable-DETR')

from detectron2.engine import default_setup
import torch.multiprocessing as mp


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
    
    default_setup(cfg, args)
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
        choices=['lvis', 'openimages', 'objects365', 'coco', 'ovcoco', 'custom'],
        help="different type for ovd evaluation",
    )
    parser.add_argument(
        "--save-features",
        default="lvis",
        help="whether save features",
    )
    parser.add_argument(
        "--save-perembeddings-path",
        default=None,
        help="whether save feature of per image",
    )
    parser.add_argument(
        "--workers-per-gpu",
        default= 1,
        help="workers for per gpu",
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
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser



# 初始化字典，用于存储不同category_id的tensor features
def append_feature_to_category(category_features, feature_tensor, category_id):
    """
    将新的feature_tensor根据category_id拼接到相应的现有tensor features上。
    
    :param feature_tensor: 新到的tensor feature, shape为[1, 2048, 7, 7]
    :param category_id: 这个tensor feature对应的category_id
    """
    # 检查category_id是否已存在
    if category_id in category_features:
        # 如果存在，拼接新的tensor feature到现有的tensor features上
        category_features[category_id] = torch.cat((category_features[category_id], feature_tensor), dim=0)
    else:
        # 如果这是第一次遇到这个category_id，直接添加
        category_features[category_id] = feature_tensor
        
    return category_features

# def accumulate_and_average_features(feature_tensor, category_id):
#     """
#     累积特征和到相应类别的总和中，并逐步计算该类别的特征平均值。
#     这种方法避免了直接存储所有特征tensor，以节省内存使用。
    
#     :param feature_tensor: 新到的tensor feature，shape为[1, 2048, 7, 7]
#     :param category_id: 这个tensor feature对应的category_id
#     """
#     if category_id in category_features:
#         # 累积该类别的特征和和计数
#         category_features[category_id] += feature_tensor.squeeze(0)
#         category_feature_counts[category_id] += 1
#     else:
#         # 对于新的类别，初始化特征和计数
#         category_features[category_id] = feature_tensor.squeeze(0)
#         category_feature_counts[category_id] = 1
        

def update_local_features(feature_tensor, category_id, category_features, category_feature_counts):
    """
    累积特征列表中的特征到相应类别的总和中，并逐步计算该类别每个特征的平均值。
    
    :param category_features: 局部的类别特征字典。
    :param feature_counts: 局部的类别特征计数字典。
    :param category_id: 当前图像的类别ID。
    :param features: 包含若干特征Tensor的列表。
    """
    if category_id in category_features:
        # 如果这个category_id已存在，累加每个特征向量并更新计数
        category_features[category_id] += feature_tensor.squeeze(0)
        category_feature_counts[category_id] += 1
    else:
        # 对于新的类别，初始化特征和计数
        category_features[category_id] = feature_tensor.squeeze(0)
        category_feature_counts[category_id] = 1 
        
        
def calculate_all_category_feature_averages(category_features, category_feature_counts):
    """
    为所有类别计算特征平均值，并返回一个新的字典，其中包含每个类别的平均特征。
    """
    category_feature_averages = {}
    for category_id in category_features:
        if category_id in category_feature_counts:
            average_feature = category_features[category_id] / category_feature_counts[category_id]
            category_feature_averages[category_id] = average_feature
    return category_feature_averages



def stack_category_feature_averages(category_feature_averages):
    """
    将category_feature_averages字典中的值按键的顺序堆叠成一个新的tensor。
    
    :param category_feature_averages: 存储不同category_id的平均特征tensor的字典, 其中键是从“1”到“65”的连续整数
    :return: 一个新的tensor, shape为[65, 2048, 7, 7]
    """
    # 确保所有键都存在，并且是连续的
    sorted_keys = sorted(category_feature_averages.keys(), key=int)
    
    # 按照键的顺序，将值添加到列表中
    features_list = [category_feature_averages[key] for key in sorted_keys]
    
    # 使用torch.stack将这些tensor堆叠成一个新的tensor
    stacked_features = torch.stack(features_list, dim=0)
    
    return stacked_features

# 然后在worker函数中使用这个函数
def worker(worker_id, gpu_id, image_paths, args, cfg, return_dict):
    torch.cuda.set_device(gpu_id)
    
    local_category_features = {}
    local_category_feature_counts = {}
    
    demo = VisualizationDemo_VLDet(cfg, args)

    for path in tqdm(image_paths, disable=not args.output):
        img = read_image(path, format="BGR")
        features = demo.run_on_image_save_features(img).cpu()
        
        
        category_id = path.split('/')[-1].split('_')[0]
        # 使用 os.path.basename 获取文件的基本名称
        basename = os.path.basename(path)
        # 使用 os.path.splitext 去除文件扩展名
        filename_without_extension = os.path.splitext(basename)[0]
                
        if args.save_perembeddings_path:
            # print("len(features)", len(features))
            # print("features[2].shape: ", features[2].shape)
            embeddings_save_folder_path = f"{args.save_perembeddings_path}/{category_id}/"
            if not os.path.exists(embeddings_save_folder_path):
                os.makedirs(embeddings_save_folder_path)
            embeddings_save_path = f"{embeddings_save_folder_path}/{filename_without_extension}.npy"
            np.save(embeddings_save_path,features.numpy())        
               
        
               
        # 更新特征向量和计数
        update_local_features(features, category_id, local_category_features, local_category_feature_counts)
        
            
        
    
    return_dict[worker_id] = (local_category_features, local_category_feature_counts)

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    cfg = setup_cfg(args)

    num_gpus = torch.cuda.device_count()
    workers_per_gpu = int(args.workers_per_gpu)


    if len(args.input) == 1:
        args.input = glob.glob(os.path.expanduser(args.input[0]))
        assert args.input, "The input path(s) was not found"
        
    total_workers = num_gpus * workers_per_gpu
    image_path_subsets = [args.input[i::total_workers] for i in range(total_workers)]

    manager = mp.Manager()
    return_dict = manager.dict()
    
    processes = []
    for gpu_id in range(num_gpus):
        for worker_num in range(workers_per_gpu):
            worker_id = gpu_id * workers_per_gpu + worker_num
            subset = image_path_subsets[worker_id]
            p = mp.Process(target=worker, args=(worker_id, gpu_id, subset, args, cfg, return_dict))
            p.start()
            processes.append(p)

    for p in processes:
        p.join()
        
        
    # 汇总所有worker的结果
    global_category_features = {}
    global_category_feature_counts = {}

    for worker_id, (local_category_features, local_category_feature_counts) in return_dict.items():
        for category_id, feature in local_category_features.items():
            if category_id not in global_category_features:
                # 初始化全局特征存储，复制第一个worker的特征值
                global_category_features[category_id] = feature.clone()
                global_category_feature_counts[category_id] = local_category_feature_counts[category_id]
            else:
                # 累加来自其他workers的特征值
                global_category_features[category_id] += feature
                global_category_feature_counts[category_id] += local_category_feature_counts[category_id]
    
    category_feature_averages = calculate_all_category_feature_averages(global_category_features, global_category_feature_counts)
    stacked_features = stack_category_feature_averages(category_feature_averages)
        # compressed_features = stacked_features.mean(dim=[2, 3])
    print(f"stacked_features.shape: {stacked_features.shape}")
        # print(f"compressed_features.shape: {compressed_features.shape}")
        
    if args.output:
        torch.save(stacked_features, args.output)
        print(f"已保存prototype到{args.output}")

    print("处理完成")


    
  