BEAM_SIZE=5
MAX_LEN_B=100
MIN_LEN=1
LEN_PEN=1.5

for i in 1 2 3 4 5 6 7 8 9
do
  DATA_PATH=$DATA/binarized_data/jointgen_data
  MODEL_PATH=$MODEL/generation_models/tplgen_question_generation/model.pt
  FOCUS_DATA_PATH=$DATA/binarized_data/focus_data/control_focus/type${i}
  TEMPLATE_DATA_PATH=$DATA/binarized_data/template_data/control_template/type${i}
  USER_DIR=../fairseq_models/tplgen_qg
  RESULT_PATH=$1/type${i}

  fairseq-generate $DATA_PATH --focus-data $FOCUS_DATA_PATH --template-data $TEMPLATE_DATA_PATH \
      --path $MODEL_PATH --results-path $RESULT_PATH \
      --task template_question_generation --qt control_type${i} \
      --beam $BEAM_SIZE --max-len-b $MAX_LEN_B --min-len $MIN_LEN --lenpen $LEN_PEN \
      --batch-size 64 \
      --truncate-source \
      --user-dir $USER_DIR;
done