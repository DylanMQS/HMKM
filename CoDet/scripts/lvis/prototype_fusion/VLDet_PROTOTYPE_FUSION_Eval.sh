clear
tmux clear-history
cd ../../../

 # 'bbox_prototype/ovcoco_prototype_paper_266x160_1x1.pth' \
    # MODEL.ROI_BOX_HEAD.PROTOTYPE_WEIGHT_PATH './exemplar_prototype/lvis/lvis_prototype_paper_mmovod_maxInstancesnums100_context0.4_squareFalse_p5_weighted_.pth' \
    # MODEL.ROI_BOX_HEAD.CONCEPT_FUSION_MODE 'laterfusion' \
# 'exemplar_prototype/lvis/lvis_prototype_paper_mmovod_maxInstancesnums100_weighted_perPoint_.pth' \
    # MODEL.ROI_BOX_HEAD.ATTRIBUTE_TEXT_WEIGHT_PATH 'datasets/metadata/attributes_gpt3_sentences_10_avg_concepts.pth' \

    # MODEL.WEIGHTS  backup/lvis_model_origin_best_2265.pth \
    # MODEL.ROI_BOX_HEAD.OPEN_CORPUS_WEIGHT_PATH 'exemplar_prototype/in21k_lvis/in21k_lvis_combined_lvis_all.pth' \


CUDA_VISIBLE_DEVICES=4,5,6,7 \
python ./train_net.py \
    --num-gpus 4 \
    --eval-only \
    --config-file ./configs/VLDet_LbaseCCcap_CLIP_PrototypeFusion.yaml \
    MODEL.WEIGHTS models/lvis_vldet.pth \
    MODEL.ROI_BOX_HEAD.PROTOTYPE_WEIGHT_PATH './exemplar_prototype/lvis/lvis_prototype_paper_mmovod_maxInstancesnums100_context0.4_squareFalse_p5_weighted_.pth' \
    MODEL.ROI_BOX_HEAD.ATTRIBUTE_VISUAL_WEIGHT_PATH './exemplar_prototype/lvis/in21k_lvis_combined_lvis_all_attribute.pth' \
    MODEL.ROI_BOX_HEAD.ATTRIBUTE_VISUAL_FREQUENCY_WEIGHT_PATH 'datasets/metadata/attribute_visual_features_frequency_weight.pth' \
    MODEL.ROI_BOX_HEAD.OPEN_CORPUS_WEIGHT_PATH 'exemplar_prototype/lvis/lvis_maxIns100_arribute_scale23_new_occ_with_prototype.pth' \
    MODEL.ROI_BOX_HEAD.CONCEPT_SOURCE 'panda' \
    MODEL.ROI_BOX_HEAD.CONCEPT_FUSION_MODE 'laterfusion' \
    MODEL.ROI_BOX_HEAD.CONCEPT_COMBINE_WEIGHT 0.275 \
    MODEL.ROI_BOX_HEAD.CONCEPT_PANDA_COMBINE_WEIGHT "[0.225, 0.05]" \
    MODEL.ROI_BOX_HEAD.X_NORM_TEMPERATURE 15.0
tmux capture-pane -p -S - -t 0 > /raid/mqsen/CKDet/results_analysis/exemplars_fusion/lvis_paper_retrain2245_prototype025_maxIns10_tta_origin.log

# tmux capture-pane -p -S - -t 0 > /raid/mqsen/CKDet/results_analysis/exemplars_fusion/vldet_mmovod_attributes_gpt4_sentences_3_with_prototype_temp15_weight0_025__language_rclip.log
