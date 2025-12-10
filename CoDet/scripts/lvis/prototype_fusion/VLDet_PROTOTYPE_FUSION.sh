clear
tmux clear-history
cd ../../../

 # 'bbox_prototype/ovcoco_prototype_paper_266x160_1x1.pth' \
     # MODEL.ROI_BOX_HEAD.CONCEPT_FUSION_MODE 'laterfusion' \

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python ./train_net.py \
    --num-gpus 4 \
    --config-file ./configs/VLDet_LbaseCCcap_CLIP_PrototypeFusion.yaml \
    MODEL.WEIGHTS  backup/lvis_model_origin_best_2265.pth \
    MODEL.ROI_BOX_HEAD.PROTOTYPE_WEIGHT_PATH 'exemplar_prototype/lvis/lvis_prototype_paper_mmovod_maxInstancesnums100_context0.4_squareFalse_p5_weighted_.pth' \
    MODEL.ROI_BOX_HEAD.ATTRIBUTE_TEXT_WEIGHT_PATH 'datasets/metadata/attributes_gpt3_sentences_10_avg_concepts.pth' \
    MODEL.ROI_BOX_HEAD.ATTRIBUTE_VISUAL_WEIGHT_PATH '/raid/mqsen/GroundingDINO/temp_data/attibutes_mp_test/full_context04.pth' \
    MODEL.ROI_BOX_HEAD.ATTRIBUTE_VISUAL_FREQUENCY_WEIGHT_PATH 'datasets/metadata/attribute_visual_features_frequency_weight.pth' \
    MODEL.ROI_BOX_HEAD.CONCEPT_SOURCE 'panda' \
    MODEL.ROI_BOX_HEAD.CONCEPT_FUSION_MODE 'laterfusion' \
    MODEL.ROI_BOX_HEAD.CONCEPT_COMBINE_WEIGHT 0.25 \
    MODEL.ROI_BOX_HEAD.CONCEPT_PANDA_COMBINE_WEIGHT "[0.1, 0.00]" \
    MODEL.ROI_BOX_HEAD.X_NORM_TEMPERATURE 15.0 \
    SOLVER.MAX_ITER 90000 \
    SOLVER.CHECKPOINT_PERIOD 1000 \
    TEST.EVAL_PERIOD 1000 \
    SOLVER.BASE_LR 0.00001 \
    MODEL.ROI_BOX_HEAD.USE_CAPTION False \
    MODEL.ROI_BOX_HEAD.OT_LOSS_WEIGHT 0.00
tmux capture-pane -p -S - -t 0 > /raid/mqsen/CKDet/results_analysis/prototype_train/vldet_lr1e-5_period1e3_batch1632_2265_protype01.log