
VID=$1; 
EXP_NAME=$2

CUDA_VISIBLE_DEVICES=0 python train.py \
  --vid $VID \
  --exp_name rel/$VID/$EXP_NAME \
  --train_ratio 1 --num_epochs 50 --suppress_person
