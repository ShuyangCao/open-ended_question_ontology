BEAM_SIZE=5
MAX_LEN_B=100
MIN_LEN=0
LEN_PEN=1.5

for i in 1 2 3 4 5 6 7 8 9
do
  DATA_PATH=$DATA/binarized_data/tplgen_data
  MODEL_PATH=$MODEL/generation_models/tplgen_template_generation/model.pt
  EXEMPLAR_DATA_PATH=$DATA/binarized_data/exemplar_data/control_exemplar/type${i}
  USER_DIR=../fairseq_models/jointgen
  RESULT_PATH=$1/type${i}

  fairseq-generate $DATA_PATH --exemplar-data $EXEMPLAR_DATA_PATH \
      --path $MODEL_PATH --results-path $RESULT_PATH \
      --task joint_generation --qt control_type${i} \
      --beam $BEAM_SIZE --max-len-b $MAX_LEN_B --min-len $MIN_LEN --lenpen $LEN_PEN \
      --batch-size 64 \
      --truncate-source \
      --user-dir $USER_DIR;
done