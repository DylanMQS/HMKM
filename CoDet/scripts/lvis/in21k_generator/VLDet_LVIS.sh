cd ../../../
# 注意cuda设备号从0开始，且修改的cuda设备数量要和num-gpu对应
    # --input  ./bbox_prototype/crop_images/*/* \

    # --input  "./bbox_prototype/coco/crop_images_train2017_score0.95_minarea256.0/*/*" \
    # --output ./bbox_prototype/coco/ovcoco_prototype_paper_train2017_score0.95_minarea256.0_1333x800.pth \

    # --input  "./inference_demo/debug_mp/*/*" \
    # --output ./bbox_prototype/coco/test_mp_9.pth \
CUDA_VISIBLE_DEVICES=4,5,6,7 \
python ./prototypes_generator_lvis_in21k.py \
    --workers-per-gpu 2 \
    --config-file ./configs/VLDet_LbaseCCcap_CLIP_PrototypeFusion.yaml \
    --vocabulary lvis \
    --input  "/raid/mqsen/mm-ovod/datasets/imagenet/imagenet21k_P/train/*/" \
    --save-features 'lvis' \
    --save-perembeddings-path "./exemplar_prototype/in21k_lvis/" \
    --opts MODEL.WEIGHTS models/lvis_vldet.pth MODEL.ROI_BOX_HEAD.CONCEPT_FUSION_MODE 'no'  MODEL.ROI_BOX_HEAD.OPEN_CORPUS_WEIGHT_PATH 'exemplar_prototype/in21k_coco/in21k_coco_combined.pth' \

    # INPUT.MAX_SIZE_TEST 160


tmux capture-pane -p -S - -t 0 > /raid/mqsen/CKDet/results_analysis/prototype_fusion/vldet_debug.log