BEAM_SIZE=5
MAX_LEN_B=100
MIN_LEN=0
LEN_PEN=1.5

DATA_PATH=$DATA/binarized_data/jointgen_data
MODEL_PATH=$MODEL/generation_models/jointgen/model.pt
USER_DIR=../fairseq_models/jointgen
RESULT_PATH=$1

fairseq-generate $DATA_PATH \
    --path $MODEL_PATH --results-path $RESULT_PATH \
    --task joint_generation --qt type \
    --beam $BEAM_SIZE --max-len-b $MAX_LEN_B --min-len $MIN_LEN --lenpen $LEN_PEN \
    --batch-size 64 \
    --truncate-source \
    --user-dir $USER_DIR;