cd ../../../
# 注意cuda设备号从0开始，且修改的cuda设备数量要和num-gpu对应
    # --input  ./datasets/coco/ov-inference/* \
CUDA_VISIBLE_DEVICES=6 \
python ./demo.py \
    --config-file ./configs/CoDet_OVCOCO_R50_1x.yaml \
    --vocabulary ovcoco \
    --input  '../CKDet/ovcoco_inference/test/pic/*' \
    --output ../CKDet/ovcoco_inference/test/output/ \
    --json_output ../CKDet/ovcoco_inference/test/codet_test.json \
    --opts MODEL.WEIGHTS models/CoDet_OVCOCO_R50_1x.pth MODEL.ROI_BOX_HEAD.CONCEPT_PANDA_COMBINE_WEIGHT "[0.0, 0.3]"


# CUDA_VISIBLE_DEVICES=6 \
# python ./demo.py \
#     --config-file ./configs/CoDet_OVCOCO_R50_1x.yaml \
#     --vocabulary ovcoco \
#     --input  '../CKDet/ovcoco_inference/pictures/*' \
#     --output ../CKDet/ovcoco_inference/codet_default/ \
#     --json_output ../CKDet/ovcoco_inference/codet_default.json \
#     --opts MODEL.WEIGHTS models/CoDet_OVCOCO_R50_1x.pth MODEL.ROI_BOX_HEAD.CONCEPT_PANDA_COMBINE_WEIGHT "[0.0, 0.0]"