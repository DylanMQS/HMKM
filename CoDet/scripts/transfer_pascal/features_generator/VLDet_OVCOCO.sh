cd ../../../
# 注意cuda设备号从0开始，且修改的cuda设备数量要和num-gpu对应
    # --input  ./bbox_prototype/crop_images/*/* \

    # --input  "./bbox_prototype/coco/crop_images_train2017_score0.95_minarea256.0/*/*" \
    # --output ./bbox_prototype/coco/ovcoco_prototype_paper_train2017_score0.95_minarea256.0_1333x800.pth \

    # --input  "./inference_demo/debug_mp/*/*" \
    # --output ./bbox_prototype/coco/test_mp_9.pth \

    # --opts MODEL.WEIGHTS models/coco_vldet.pth MODEL.ROI_BOX_HEAD.CONCEPT_FUSION_MODE 'no'

CUDA_VISIBLE_DEVICES=4,5,6,7 \
python ./prototypes_generator_coco_mp.py \
    --workers-per-gpu 2 \
    --config-file ./configs/CoDet_OVCOCO_TRANSFER_PASCAL.yaml \
    --vocabulary coco \
    --input  "../CKDet/exemplar_prototype/pascal_voc/mmovod_maxInstancesnums100/regions/*/*" \
    --output ./exemplar_prototype/pascal_voc/transfer_pascal_ins100_.pth \
    --save-features 'coco' \
    --save-perembeddings-path "./exemplar_prototype/pascal_voc/transfer_pascal_ins100_/embeddings/" \
    --opts MODEL.WEIGHTS models/CoDet_OVCOCO_R50_1x.pth     
    # INPUT.MAX_SIZE_TEST 160

# CUDA_VISIBLE_DEVICES=4 \
# python ./prototypes_generator_lvis_mp.py \
#     --workers-per-gpu 1\
#     --config-file ./configs/VLDet_LbaseCCcap_CLIP_PrototypeFusion.yaml \
#     --vocabulary lvis \
#     --input  "./exemplar_prototype/lvis/mmovod_maxInstancesnums10_tta/regions/*" \
#     --save-features 'lvis' \
#     --save-perembeddings-path "./exemplar_prototype/lvis/mmovod_maxInstancesnums10_tta//embeddings/" \
#     --opts MODEL.WEIGHTS models/lvis_vldet.pth MODEL.ROI_BOX_HEAD.CONCEPT_FUSION_MODE 'no'


tmux capture-pane -p -S - -t 0 > /raid/mqsen/CKDet/results_analysis/prototype_fusion/vldet_debug.log