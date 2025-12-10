cd ../../../
# 注意cuda设备号从0开始，且修改的cuda设备数量要和num-gpu对应
    # --input  ./bbox_prototype/crop_images/*/* \
    # --input  "./bbox_prototype/lvis/crop_images_score35_minarea256/*/*" \
    # --output ./bbox_prototype/lvis/lvis_prototype_paper_train2017_score35_minarea256_mixed.pth \

    # --input  "./inference_demo/debug_mp/regions/*/*" \
    # --output ./bbox_prototype/lvis/test.pth \

    # --input  "./exemplar_prototype/lvis/mmovod_maxInstancesnums10_tta/regions/*" \

CUDA_VISIBLE_DEVICES=2,3 \
python ./prototypes_generator_lvis_mp.py \
    --workers-per-gpu 3\
    --config-file ./configs/CoDet_OVLVIS_R5021k_4x_ft4x.yaml \
    --vocabulary lvis \
    --input  "/raid/mqsen/CKDet/exemplar_prototype/lvis/mmovod_maxInstancesnums100_context0.4_squareFalse/regions/*/*" \
    --output exemplar_prototype/lvis/mmovod_maxIns100.pth \
    --save-features 'lvis' \
    --save-perembeddings-path "exemplar_prototype/lvis/mmovod_maxInstancesnums100_context0.4_squareFalse/embeddings/" \
    --opts MODEL.WEIGHTS models/CoDet_OVLVIS_R5021k_4x_ft4x.pth
    
    # INPUT.TEST_SIZE 160 INPUT.TRAIN_SIZE 160
    
# INPUT.MIN_SIZE_TEST 160 INPUT.MAX_SIZE_TEST 266

tmux capture-pane -p -S - -t 0 > /raid/mqsen/CKDet/results_analysis/exemplars_fusion/lvis_prototype_retrain2256_maxInssnums10_vl_mixed_.log