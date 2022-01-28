TOTAL_NUM_UPDATES=12000
WARMUP_UPDATES=720
LR=3e-05
MAX_TOKENS=4096
UPDATE_FREQ=4

BART_PATH=$1

DATA_PATH=$DATA/binarized_data/jointgen_data
TEMPLATE_DATA_PATH=$DATA/binarized_data/template_data/oracle
FOCUS_DATA_PATH=$DATA/binarized_data/focus_data/tplgen
USER_DIR=../fairseq_models/tplgen_qg

SAVE_PATH=$2

fairseq-train $DATA_PATH \
    --save-dir $SAVE_PATH \
    --restore-file $BART_PATH \
    --max-tokens $MAX_TOKENS \
    --focus-data $FOCUS_DATA_PATH --template-data $TEMPLATE_DATA_PATH \
    --task template_question_generation --qt oracle_type \
    --source-lang source --target-lang target \
    --truncate-source \
    --layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --arch focus_bart \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
    --clip-norm 0.1 --max-epoch 4 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --fp16 --update-freq $UPDATE_FREQ \
    --no-epoch-checkpoints --no-save-optimizer-state \
    --skip-invalid-size-inputs-valid-test \
    --find-unused-parameters \
    --user-dir $USER_DIR;