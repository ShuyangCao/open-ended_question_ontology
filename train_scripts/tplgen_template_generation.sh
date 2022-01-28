TOTAL_NUM_UPDATES=12000
WARMUP_UPDATES=720
LR=3e-05
MAX_TOKENS=4096
UPDATE_FREQ=4

BART_PATH=$1

DATA_PATH=$DATA/binarized_data/tplgen_data
EXEMPLAR_DATA_PATH=$DATA/binarized_data/exemplar_data/oracle
USER_DIR=../fairseq_models/jointgen

SAVE_PATH=$2

fairseq-train $DATA_PATH \
    --save-dir $SAVE_PATH \
    --restore-file $BART_PATH --exemplar-data $EXEMPLAR_DATA_PATH \
    --max-tokens $MAX_TOKENS \
    --task joint_generation --qt oracle_type \
    --source-lang source --target-lang target \
    --truncate-source \
    --layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --arch joint_bart \
    --criterion joint_generation_loss --sentence-avg \
    --label-smoothing 0.1 \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
    --clip-norm 0.1 --max-epoch 4 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --update-freq $UPDATE_FREQ \
    --no-epoch-checkpoints --no-save-optimizer-state \
    --best-checkpoint-metric ppl \
    --skip-invalid-size-inputs-valid-test \
    --find-unused-parameters \
    --train-subset train \
    --user-dir $USER_DIR;