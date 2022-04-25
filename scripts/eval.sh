
CKP=ckpts/rel
VID=$1
EXP_NAME=$2
EXP=rel
OUT="masks"
MASKS_N_SAMPLES=0
SUMMARY_N_SAMPLES=0

EPOCH=49

CUDA_VISIBLE_DEVICES=0 python evaluate.py \
  --path $CKP\/$VID\/$EXP_NAME\/epoch\=$EPOCH\.ckpt \
  --vid $VID --exp $EXP --exp_name $EXP_NAME\
  --is_eval_script \
  --outputs $OUT \
  --masks_n_samples $MASKS_N_SAMPLES \
  --summary_n_samples $SUMMARY_N_SAMPLES #--suppress_person
