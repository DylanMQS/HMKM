# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from detectron2.config import configurable
from detectron2.layers import Linear, ShapeSpec
import json

class ZeroShotClassifier(nn.Module):
    @configurable
    def __init__(
            self,
            input_shape: ShapeSpec,
            *,
            num_classes: int,
            zs_weight_path: str,
            detection_weight_path: str,
            zs_weight_dim: int = 512,
            use_bias: float = 0.0,
            norm_weight: bool = True,
            norm_temperature: float = 50.0,
            prototype_weight_path: str, # prototype权重路径，规定prototype指的就是视觉原型
            attribute_visual_weight_path: str, # attribute visual权重路径
            concept_panda_combine_weight: list, # 同时使用prototype和attribute特征的加权系数
            x_norm_temperature: float = 0.0, # 进行region-region的计算时，对特征x的正则温度
            categories_info_path: str = '', # novel 和 base类的索引划分文件
    ):
        super().__init__()
        if isinstance(input_shape, int):  # some backward compatibility
            input_shape = ShapeSpec(channels=input_shape)
        input_size = input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)
        self.norm_weight = norm_weight
        self.norm_temperature = norm_temperature

        self.use_bias = use_bias < 0
        if self.use_bias:
            self.cls_bias = nn.Parameter(torch.ones(1) * use_bias)

        self.linear = nn.Linear(input_size, zs_weight_dim)

        if zs_weight_path == 'rand':
            zs_weight = torch.randn((zs_weight_dim, num_classes))
            nn.init.normal_(zs_weight, std=0.01)
        else:
            zs_weight = torch.tensor(np.load(zs_weight_path),dtype=torch.float32).permute(1, 0).contiguous()  # D x C
            # zs_weight_path_test = "datasets/metadata/lvis_clip_hrchy_l1_llm.npy"
            zs_weight_path_test_path = "datasets/metadata/coco_unseen_clip_hrchy_l1_llm.npy"
            zs_weight_test = torch.tensor(np.load(zs_weight_path_test_path), dtype=torch.float32).permute(1, 0).contiguous() # D x C
        zs_weight = torch.cat([zs_weight, zs_weight.new_zeros((zs_weight_dim, 1))],dim=1)  # D x (C + 1)
        zs_weight_test = torch.cat([zs_weight_test, zs_weight_test.new_zeros((zs_weight_dim, 1))], dim=1) # D x (C + 1)

        if detection_weight_path == 'rand':
            detection_weight = torch.randn((zs_weight_dim, num_classes))
            nn.init.normal_(zs_weight, std=0.01)
        else:
            detection_weight = torch.tensor(
                np.load(detection_weight_path),
                dtype=torch.float32).permute(1, 0).contiguous()  # D x C

        detection_weight = torch.cat(
            [detection_weight, detection_weight.new_zeros((zs_weight_dim, 1))],
            dim=1)  # D x (C + 1)
        
            # 直接在2048维度进行计算 65 * 2048
        prototype_weight = torch.tensor(torch.load(prototype_weight_path), dtype=torch.float32).permute(1, 0).contiguous() # D x C
        # 这里的维度可能需要改，用语言时使用share_proj_l_dim，用视觉时使用share_proj_v_dim，不对，直接就是原始维度的第一项就行
        # coco_in21k_avg_background = torch.load("/raid/mqsen/CKDet/exemplar_prototype/coco/coco_in21k_avg_background.pth").unsqueeze(1)
        # prototype_weight = torch.cat([prototype_weight, coco_in21k_avg_background], dim=1) # D x (C + 1)
        prototype_weight = torch.cat([prototype_weight, prototype_weight.new_zeros((prototype_weight.shape[0], 1))], dim=1) # D x (C + 1)
        attribute_visual_weight = torch.tensor(torch.load(attribute_visual_weight_path), dtype=torch.float32).permute(2, 1, 0).contiguous() # D x A x C
        print("attribute_visual_weight.shape: ", attribute_visual_weight.shape)    # 1024 x 9 x 1203
        D, A, C = attribute_visual_weight.shape
        attribute_visual_weight = torch.cat([attribute_visual_weight, attribute_visual_weight.new_zeros((D, A, 1))], dim=2)
        print("with zero attribute_visual_weight.shape: ", attribute_visual_weight.shape)    # 1024 x 9 x 1203


        with open(categories_info_path, 'r') as f:
            categories = json.load(f)
        self.unseen_cls = np.array(categories['novel'])
        self.seen_cls = np.array(categories['base'])
        
        
        if self.norm_weight:
            zs_weight = F.normalize(zs_weight, p=2, dim=0)
            zs_weight_test = F.normalize(zs_weight_test, p=2, dim=0)
            detection_weight = F.normalize(detection_weight, p=2, dim=0)
            prototype_weight = F.normalize(prototype_weight, p=2, dim=0)
            attribute_visual_weight = F.normalize(attribute_visual_weight, p=2, dim=0)    

        if zs_weight_path == 'rand':
            self.zs_weight = nn.Parameter(zs_weight)
            self.detection_weight = nn.Parameter(detection_weight)
        else:
            self.register_buffer('zs_weight', zs_weight)
            self.register_buffer('zs_weight_test', zs_weight_test)
            self.register_buffer('detection_weight', detection_weight)
            # 设置加载的检测权重不通过反向传播更新
            # 设置加载的知识权重不通过反向传播更新
            self.register_buffer('prototype_weight', prototype_weight)                     
            self.register_buffer('attribute_visual_weight', attribute_visual_weight) 

        # assert self.detection_weight.shape[1] == num_classes + 1, self.detection_weight.shape
        self.concept_panda_combine_weight = concept_panda_combine_weight
        self.prototype_weight_path = prototype_weight_path
        self.x_norm_temperature = x_norm_temperature
        
    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            'input_shape': input_shape,
            'num_classes': cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            'zs_weight_path': cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH,
            'detection_weight_path': cfg.MODEL.ROI_BOX_HEAD.DETECTION_WEIGHT_PATH,
            'zs_weight_dim': cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_DIM,
            'use_bias': cfg.MODEL.ROI_BOX_HEAD.USE_BIAS,
            'norm_weight': cfg.MODEL.ROI_BOX_HEAD.NORM_WEIGHT,
            'norm_temperature': cfg.MODEL.ROI_BOX_HEAD.NORM_TEMP,
            'prototype_weight_path': cfg.MODEL.ROI_BOX_HEAD.PROTOTYPE_WEIGHT_PATH, # prototype权重的路径
            'attribute_visual_weight_path': cfg.MODEL.ROI_BOX_HEAD.ATTRIBUTE_VISUAL_WEIGHT_PATH, # attribute权重路径
            'concept_panda_combine_weight': cfg.MODEL.ROI_BOX_HEAD.CONCEPT_PANDA_COMBINE_WEIGHT, # 同时使用prototype和attribute特征的加权系数
            'x_norm_temperature': cfg.MODEL.ROI_BOX_HEAD.X_NORM_TEMPERATURE, # 进行region-region的计算时，对特征x的正则温度
            'categories_info_path': cfg.MODEL.ROI_BOX_HEAD.CATEGORIES_INFO_PATH
        }

    def forward(self, input_x, ann_type='box', classifier=None):
        '''
        Inputs:
            x: B x D'
            classifier_info: (C', C' x D)
        '''
        proj_x = self.linear(input_x)
        if classifier is not None:
            zs_weight = classifier.permute(1, 0).contiguous()  # D x C'
            zs_weight = F.normalize(zs_weight, p=2, dim=0) \
                if self.norm_weight else zs_weight
        else:
            if self.training and ann_type != 'box':
                zs_weight = self.zs_weight
            else:
                #             zs_weight = self.zs_weight_test
                # print("zs_weight = self.zs_weight_test")
                zs_weight = self.detection_weight
                # zs_weight = self.zs_weight_test

                
        concept_weight = {"prototype": self.prototype_weight, "attribute_visual": self.attribute_visual_weight}
        norm_input_x = self.x_norm_temperature * F.normalize(input_x, p=2, dim=1)
        x_concept_prototype = None
        # x_concept_prototype = torch.mm(input_x, concept_weight["prototype"])
        
        x_concept_prototype = torch.mm(norm_input_x, concept_weight["prototype"])
        # mmovod特供，文本层面相似度计算
        # x_concept_prototype = torch.mm(proj_x, concept_weight["prototype"])

        
        x_concept_attribute = None
        use_visual_attribute = True
        if use_visual_attribute:
            # 所有类别的属性知识库的检索
            total_attribute_visual = concept_weight["attribute_visual"].reshape(-1, concept_weight["attribute_visual"].shape[0])
            
            # 步骤 1: 计算余弦相似度
            similarity_matrix = torch.matmul(F.normalize(input_x, p=2, dim=1), total_attribute_visual.T)
            # 选择Top-k相似项
            top_n = 2 # 可以根据需要调整k的值
            values, indices = torch.topk(similarity_matrix, k=top_n, dim=1)
            # 获取对应的加权特征
            top_features = total_attribute_visual[indices]
            # 计算加权特征，使用广播扩展values的形状
            weighted_features = top_features * values.unsqueeze(-1)
            # 计算加权平均特征
            weighted_sums = torch.sum(weighted_features, dim=1)
            sum_similarities = torch.sum(values, dim=1, keepdim=True)
            # 避免分母为0
            epsilon = 1e-8  # 小常数
            # norm_concept_attribute = self.x_norm_temperature * weighted_sums / (sum_similarities + epsilon)
            norm_concept_attribute = self.x_norm_temperature * F.normalize(weighted_sums, p=2, dim=1)
            x_concept_attribute = torch.mm(norm_concept_attribute, concept_weight["prototype"]) # 这个是开放语料库的                 
            x_concept_attribute[-1] = 0
        x_concept = [x_concept_prototype, x_concept_attribute]    
        
        if self.norm_weight:
            proj_x = self.norm_temperature * F.normalize(proj_x, p=2, dim=1)
            
        x = torch.mm(proj_x, zs_weight)
        
        # 只加全局对象原型知识
        # x = x + float(self.concept_panda_combine_weight[0]) * x_concept[0] 

        # 只对unseen加
        ##### Aggregated Object-Classification -- Visual Similarity Estimation
        # x[:, self.unseen_cls] =  self.norm_temperature * (x[:, self.unseen_cls]  + self.alpha * (scores2[:, self.unseen_cls]))
        # x[:, self.seen_cls] = self.norm_temperature * x[:, self.seen_cls]
        # print("self.concept_panda_combine_weight: ", self.concept_panda_combine_weight)
        x[:, self.unseen_cls] =  x[:, self.unseen_cls]  + float(self.concept_panda_combine_weight[0]) * x_concept[0][:, self.unseen_cls]
        x[:, self.unseen_cls] =  x[:, self.unseen_cls]  + float(self.concept_panda_combine_weight[1]) * x_concept[1][:, self.unseen_cls]
        # x = x + float(self.concept_panda_combine_weight[1]) * x_concept[1]   
        ##### 
        # x = x + float(self.concept_panda_combine_weight[0]) * x_concept[0] \
        #     + float(self.concept_panda_combine_weight[1]) * x_concept[1]    
        
        
        # lambda_total = 0.225
        # # Step 1: 相似度提取
        # sim_obj = x_concept[0][:, self.unseen_cls]    # [B, #unseen]
        # sim_attr = x_concept[1][:, self.unseen_cls]   # [B, #unseen]
        # # Step 2: 获取 softmax 权重，仅用于生成“属性增强权重”
        # # softmax(sim_obj) 表示对象 prototype 对各类别的置信度
        # obj_conf = torch.softmax(sim_obj, dim=1)      # [B, #unseen]
        # # Step 3: 属性增强部分（由对象置信度控制）
        # attr_boost = obj_conf * sim_attr              # [B, #unseen]
        # # Step 4: 融合：主干是 sim_obj，属性是附加项
        # # concept_boost = sim_obj + attr_boost
        # concept_boost = sim_obj
        # # Step 5: 融入原始文本相似度
        # x[:, self.unseen_cls] += lambda_total * concept_boost
            
        if self.use_bias:
            x = x + self.cls_bias
        return x








    def forward_without_linear_mapping(self, x, classifier=None):
        if classifier is not None:
            zs_weight = classifier.permute(1, 0).contiguous()  # D x C'
            zs_weight = F.normalize(zs_weight, p=2, dim=0) \
                if self.norm_weight else zs_weight
        else:
            zs_weight = self.zs_weight
        if self.norm_weight:
            x = self.norm_temperature * F.normalize(x, p=2, dim=1)
        x = torch.mm(x, zs_weight)
        if self.use_bias:
            x = x + self.cls_bias
        return x