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

    # MODEL.ROI_BOX_HEAD.OPEN_CORPUS_WEIGHT_PATH 'exemplar_prototype/lvis/swin_lvis_ins100_attribute_kmeans20_scale23_new_occ_with_prototype.pth' \

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 \
python ./train_net.py \
    --num-gpus 6 \
    --eval-only \
    --config-file ./configs/CoDet_OVLVIS_SwinB_4x_ft4x.yaml \
    MODEL.WEIGHTS models/CoDet_OVLVIS_SwinB_4x_ft4x.pth \
    MODEL.ROI_BOX_HEAD.PROTOTYPE_WEIGHT_PATH './exemplar_prototype/lvis/mmovod_maxInstancesnums100_.pth' \
    MODEL.ROI_BOX_HEAD.ATTRIBUTE_VISUAL_WEIGHT_PATH './exemplar_prototype/lvis/lvis_ins100_attribute_kmeans15_delete.pth' \
    MODEL.ROI_BOX_HEAD.CONCEPT_SOURCE 'panda' \
    MODEL.ROI_BOX_HEAD.CONCEPT_FUSION_MODE 'laterfusion' \
    MODEL.ROI_BOX_HEAD.CONCEPT_PANDA_COMBINE_WEIGHT "[0.0, 0.0]" \
    MODEL.ROI_BOX_HEAD.X_NORM_TEMPERATURE 15.0
tmux capture-pane -p -S - -t 0 > /raid/mqsen/CKDet/results_analysis/exemplars_fusion/lvis_paper_retrain2245_prototype025_maxIns10_tta_origin.log

# tmux capture-pane -p -S - -t 0 > /raid/mqsen/CKDet/results_analysis/exemplars_fusion/vldet_mmovod_attributes_gpt4_sentences_3_with_prototype_temp15_weight0_025__language_rclip.log
