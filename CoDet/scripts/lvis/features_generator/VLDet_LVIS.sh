cd ../../../
# 注意cuda设备号从0开始，且修改的cuda设备数量要和num-gpu对应
    # --input  ./bbox_prototype/crop_images/*/* \
    # --input  "./bbox_prototype/lvis/crop_images_score35_minarea256/*/*" \
    # --output ./bbox_prototype/lvis/lvis_prototype_paper_train2017_score35_minarea256_mixed.pth \

    # --input  "./inference_demo/debug/*/*" \
    # --output ./bbox_prototype/lvis/test.pth \
    # --input  "./exemplar_prototype/lvis/mmovod_maxInstancesnums10_tta/regions/*" \

CUDA_VISIBLE_DEVICES=4 \
python ./prototypes_generator_lvis_mp.py \
    --workers-per-gpu 1\
    --config-file ./configs/VLDet_LbaseCCcap_CLIP_PrototypeFusion.yaml \
    --vocabulary lvis \
    --input  "./temp/tta_temp/*/*" \
    --input  "./exemplar_prototype/lvis/mmovod_maxInstancesnums10_tta/regions/*" \
    --save-features 'lvis' \
    --save-perembeddings-path "./exemplar_prototype/lvis/mmovod_maxInstancesnums10_tta//embeddings/" \
    --opts MODEL.WEIGHTS models/lvis_vldet.pth MODEL.ROI_BOX_HEAD.CONCEPT_FUSION_MODE 'no'
    
    # INPUT.TEST_SIZE 160 INPUT.TRAIN_SIZE 160
    
# INPUT.MIN_SIZE_TEST 160 INPUT.MAX_SIZE_TEST 266

tmux capture-pane -p -S - -t 0 > /raid/mqsen/CKDet/results_analysis/exemplars_fusion/lvis_prototype_retrain2256_maxInssnums10_vl_mixed_.log