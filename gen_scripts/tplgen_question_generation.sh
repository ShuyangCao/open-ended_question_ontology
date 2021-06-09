BEAM_SIZE=5
MAX_LEN_B=100
MIN_LEN=1
LEN_PEN=1.5

DATA_PATH=$DATA/binarized_data/jointgen_data
MODEL_PATH=$MODEL/generation_models/tplgen_question_generation/model.pt
FOCUS_DATA_PATH=$DATA/binarized_data/focus_data/tplgen
TEMPLATE_DATA_PATH=$DATA/binarized_data/template_data/tplgen
USER_DIR=../fairseq_models/tplgen_qg
RESULT_PATH=$1

fairseq-generate $DATA_PATH --focus-data $FOCUS_DATA_PATH --template-data $TEMPLATE_DATA_PATH \
    --path $MODEL_PATH --results-path $RESULT_PATH \
    --task template_question_generation --qt type \
    --beam $BEAM_SIZE --max-len-b $MAX_LEN_B --min-len $MIN_LEN --lenpen $LEN_PEN \
    --batch-size 64 \
    --truncate-source \
    --user-dir $USER_DIR;