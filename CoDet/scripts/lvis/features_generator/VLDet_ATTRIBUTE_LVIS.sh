cd ../../../
# 注意cuda设备号从0开始，且修改的cuda设备数量要和num-gpu对应
    # --input  ./bbox_prototype/crop_images/*/* \
    # --input  "./bbox_prototype/lvis/crop_images_score35_minarea256/*/*" \
    # --output ./bbox_prototype/lvis/lvis_prototype_paper_train2017_score35_minarea256_mixed.pth \

    # --input  "./inference_demo/debug/*/*" \
    # --output ./bbox_prototype/lvis/test.pth \

CUDA_VISIBLE_DEVICES=4,5,6 \
python ./attributes_generator_lvis_mp.py \
    --workers-per-gpu 2\
    --config-file ./configs/VLDet_LbaseCCcap_CLIP_PrototypeFusion.yaml \
    --vocabulary lvis \
    --input  "/raid/mqsen/GroundingDINO/test_imgs/output/attributes_context0.4/*" \
    --output "/raid/mqsen/GroundingDINO/temp_data/attibutes_mp_test/full_context04.pth" \
    --save-features 'lvis' \
    --save-perembeddings-path "" \
    --opts MODEL.WEIGHTS models/lvis_vldet.pth  MODEL.ROI_BOX_HEAD.CONCEPT_FUSION_MODE 'no'

    
    # INPUT.TEST_SIZE 160 INPUT.TRAIN_SIZE 160
    
# INPUT.MIN_SIZE_TEST 160 INPUT.MAX_SIZE_TEST 266

tmux capture-pane -p -S - -t 0 > /raid/mqsen/CKDet/results_analysis/visual_attribute/lvis_arribute_visual_features_.log