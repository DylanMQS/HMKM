cd ../../../
CUDA_VISIBLE_DEVICES=7,2,0,1 \
python ./train_net.py \
    --num-gpus 4 \
    --eval-only \
    --config-file ./configs/VLDet_LbaseCCcap_CLIP_PrototypeFusion.yaml  \
    MODEL.WEIGHTS ./models/lvis_vldet.pth
# tmux capture-pane -p -S - -t 0 > ./train_test_0102.log