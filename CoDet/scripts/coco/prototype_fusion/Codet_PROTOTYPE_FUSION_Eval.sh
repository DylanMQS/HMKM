clear
# tmux clear-history
cd ../../../

 # 'bbox_prototype/ovcoco_prototype_paper_266x160_1x1.pth' \
    #  MODEL.ROI_BOX_HEAD.PROTOTYPE_WEIGHT_PATH 'bbox_prototype/coco/ovcoco_prototype_paper_train2017_score0.95_minarea256.0.pth' \

# exemplar_prototype/coco/coco_maxIns10.pth
# models/vg_features/vg_features_2048.pth
    # MODEL.WEIGHTS  models/coco_vldet.pth \

    # MODEL.ROI_BOX_HEAD.ATTRIBUTE_TEXT_WEIGHT_PATH 'datasets/metadata/coco_attributes_sentences_avg_concepts.pth' \

# coco_ins10_.pth
# coco_ins100_attribute_kmeans15_.pth
# coco_ins100_attribute_comparision_kmeans
# coco_ins10_randomsplit_5
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python train_net.py \
    --num-gpus 8 \
    --config-file configs/CoDet_OVCOCO_R50_1x.yaml \
    --eval-only \
    MODEL.WEIGHTS models/CoDet_OVCOCO_R50_1x.pth \
    MODEL.ROI_BOX_HEAD.DETECTION_WEIGHT_PATH 'datasets/metadata/coco_clip_a+cname_mask.npy' \
    MODEL.ROI_BOX_HEAD.PROTOTYPE_WEIGHT_PATH 'exemplar_prototype/coco/coco_ins10_.pth' \
    MODEL.ROI_BOX_HEAD.ATTRIBUTE_VISUAL_WEIGHT_PATH 'exemplar_prototype/coco/coco_ins50_attribute_kmeans15_scale23.pth' \
    MODEL.ROI_BOX_HEAD.CONCEPT_PANDA_COMBINE_WEIGHT "[0.25, 0.0]" \
    MODEL.ROI_BOX_HEAD.X_NORM_TEMPERATURE 15.0

tmux capture-pane -p -S - -t 0 > /raid/mqsen/CKDet/results_analysis/exemplars_fusion/lvis_prototype_retrain2256_maxInssnums10_vl_mixed_.log