# Controllable Open-ended Question Generation with A New Question Type Ontology

Code for ACL 2021 paper "Controllable Open-ended Question Generation with A New Question Type Ontology".

### Raw data

[Question type annotation](https://drive.google.com/drive/folders/1eipP3o5d_8aGqks3EaOTJxxgYe_oD3vQ?usp=sharing)

[Reddit](https://drive.google.com/drive/folders/1ofpx55ClsXQUaFPaaxdM2S84RgaObmsH?usp=sharing)

Our Yahoo dataset is based on the Yahoo Answer L6 dataset. 
After obtaining the license for the L6 dataset,
please email Shuyang (caoshuy@umich.edu) with the proof of license attached
to obtain the Yahoo dataset.

### Environment

Our experiments are based on `PyTorch 1.7.0` and [Fairseq](https://github.com/pytorch/fairseq) at commit `0db28cd` with a simple edit. Newer versions of Fairseq might also work. For graph neural networks, we use `PyTorch-Geometric 1.7.2`.

```shell
# virtual environment
conda create -n open_ended_qg python=3.7
conda activate open_ended_qg

# install pytorch
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=10.2 -c pytorch

# install fairseq, note that you need to follow the instructions in fairseq/README.md 
# to install other dependencies (e.g., apex for training)
cd lib/fairseq
pip install -e .
# fix hydra error
pip install hydra-core==1.0.7

# install torch-geometric
pip install torch-scatter==2.0.7 -f https://data.pyg.org/whl/torch-1.7.0+cu102.html
pip install torch-sparse==0.6.9 -f https://data.pyg.org/whl/torch-1.7.0+cu102.html
pip install torch-geometric==1.7.2
```

**Note:** we use AllenNLP during data processing, which requires a different PyTorch version.
Please use a **different virtual environment for AllenNLP**. 

### Data Preprocess

Preprocessed binarized Reddit data can be downloaded from [here](https://drive.google.com/drive/folders/1ceFL4mgm3CYBEvS8y4ScpcrvonoVXOhO?usp=sharing).

For data preprocessing, please refer to the README in [data_preprocess](data_preprocess).

------

### Run our models

Please download the generation models from [here](https://drive.google.com/drive/folders/1EpfamTiOosKvy_s9Bm_tW1NdtAzKpkY7?usp=sharing) 
and put them under `$MODEL/generation_models`. The binarized dataset should be under `$DATA/binarized_data`.

To convert the fairseq generation output to text, use `convert_output.py`:

```shell
python convert_output.py --generate-dir <result_dir>
```

##### JointGen

```shell
cd gen_scripts
./jointgen.sh $DATA/output/jointgen
```

##### ExplGen

```shell
cd gen_scripts
./explgen.sh $DATA/output/explgen
```

##### TplGen

```shell
cd gen_scripts
./tplgen_question_generation.sh $DATA/output/tplgen_question
```

##### ExplGen: conditioned on top 9 types

```shell
cd gen_scripts
./explgen_9types.sh $DATA/output/explgen_9types
```

##### TplGen: conditioned on top 9 types

```shell
cd gen_scripts
./tplgen_question_generation_9types.sh $DATA/output/tplgen_question_9types
```

------

### Train our models

Please set `BART_PATH` as the path to the `bart.large` model, which can be downloaded [here](https://github.com/pytorch/fairseq/tree/master/examples/bart).

```shell
export BART_PATH=<path_to_bart_large_dir>/model.pt
```

##### JointGen

```shell
cd train_scripts
CUDA_VISIBLE_DEVICES=0,1 ./jointgen.sh $BART_PATH $MODEL/jointgen
```

##### ExplGen

```shell
cd train_scripts
CUDA_VISIBLE_DEVICES=0,1 ./explgen.sh $BART_PATH $MODEL/explgen
```

##### TplGen: template generation

```shell
cd train_scripts
CUDA_VISIBLE_DEVICES=0,1 ./tplgen_template_generation.sh $BART_PATH $MODEL/tplgen_template_generation
```

##### TplGen: question generation

```shell
cd train_scripts
CUDA_VISIBLE_DEVICES=0,1 ./tplgen_question_generation.sh $BART_PATH $MODEL/tplgen_question_generation
```
