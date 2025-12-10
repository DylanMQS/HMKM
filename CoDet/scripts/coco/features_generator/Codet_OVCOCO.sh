cd ../../../
# 注意cuda设备号从0开始，且修改的cuda设备数量要和num-gpu对应
    # --input  ./bbox_prototype/crop_images/*/* \

    # --input  "./bbox_prototype/coco/crop_images_train2017_score0.95_minarea256.0/*/*" \
    # --output ./bbox_prototype/coco/ovcoco_prototype_paper_train2017_score0.95_minarea256.0_1333x800.pth \

    # --input  "./inference_demo/debug_mp/*/*" \
    # --output ./bbox_prototype/coco/test_mp_9.pth \

    # --opts MODEL.WEIGHTS models/coco_vldet.pth MODEL.ROI_BOX_HEAD.CONCEPT_FUSION_MODE 'no'

    # --input  "debug_mp/regions/*/*" \
    # --output debug_mp/debug_mp.pth \
    # --save-features 'coco' \
    # --save-perembeddings-path "debug_mp/embeddings/" \


    # --input  "./exemplar_prototype/coco/ins10_context04_squareFalse/regions/*/*" \
    # --output ./exemplar_prototype/coco/coco_ins10_.pth \
    # --save-features 'coco' \
    # --save-perembeddings-path "./exemplar_prototype/coco/ins10_context04_squareFalse/embeddings/" \
CUDA_VISIBLE_DEVICES=7 \
python ./prototypes_generator_coco_mp.py \
    --workers-per-gpu 1 \
    --config-file configs/CoDet_OVCOCO_R50_1x.yaml \
    --vocabulary ovcoco \
    --input  "../CKDet/exemplar_prototype/coco/ins100_attribute_scale12/regions/*/*" \
    --output ./exemplar_prototype/coco/coco_temp.pth \
    --save-features 'coco' \
    --opts MODEL.WEIGHTS models/CoDet_OVCOCO_R50_1x.pth MODEL.ROI_BOX_HEAD.DETECTION_WEIGHT_PATH 'datasets/metadata/coco_clip_a+cname_mask.npy'
    # --save-perembeddings-path "./exemplar_prototype/coco/ins10_randomsplit_5/embeddings/" \

    # MODEL.ROI_BOX_HEAD.PROTOTYPE_WEIGHT_PATH /raid/mqsen/CKDet/backup/coco_maxIns10.pth MODEL.ROI_BOX_HEAD.CONCEPT_SOURCE 'panda' MODEL.ROI_BOX_HEAD.CONCEPT_FUSION_MODE 'laterfusion' MODEL.ROI_BOX_HEAD.CONCEPT_PANDA_COMBINE_WEIGHT "[0.275, 0.00, 0.00]" MODEL.ROI_BOX_HEAD.X_NORM_TEMPERATURE 15.0
    # INPUT.MAX_SIZE_TEST 160


# tmux capture-pane -p -S - -t 0 > /mnt/diskb/mqsen_workspace/Detic/results_analysis/prototype_fusion/vldet_debug.log