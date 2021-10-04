## Data Preprocess

Instructions for creating the hybrid semantic graphs and template data.

First, download the raw Reddit data from [here](https://drive.google.com/drive/folders/1ofpx55ClsXQUaFPaaxdM2S84RgaObmsH?usp=sharing).
Please set the environment variable `DATA` to the directory for storing data.
The following instruction assumes the raw Reddit data is stored in `$DATA/raw`.

### Parsing and semantic role labeling

We use `Stanford CoreNLP 4.1.0` for constituency parsing, dependency parsing, and coreference resolution.
Download CoreNLP from [here](https://stanfordnlp.github.io/CoreNLP/history.html).

For semantic role labeling, we use `AllenNLP 1.1.0`.

##### Convert to single files for parsing

```shell
for SPLIT in train valid test
do
  python convert_to_single_files.py $DATA/raw/${SPLIT}.jsonl \
    $DATA/stanford_parsing_raw/${SPLIT}_input $DATA/stanford_parsing_raw/${SPLIT}_path.txt
done
```

##### Run Stanford CoreNLP

Go to the directory where CoreNLP is located and then run the following command:

```shell
for SPLIT in train valid test
do
  ./corenlp.sh -fileList $DATA/stanford_parsing_raw/${SPLIT}_path.txt \
    -outputDirectory $DATA/stanford_parsing_raw/${SPLIT}_output -outputFormat json \
    -annotators tokenize,ssplit,pos,lemma,ner,depparse,parse,coref
done
```

Note that you would need to modify the `corenlp.sh` file to increase the memory limit.
We use `-mx32g`. Finally, merge the parsing outputs:

```shell
mkdir $DATA/stanford_parsing_output
for SPLIT in train valid test
do
  python merge_corenlp_output.py $DATA/stanford_parsing_raw/${SPLIT}_output \
    $DATA/stanford_parsing_raw/${SPLIT}_path.txt \
    $DATA/stanford_parsing_output/${SPLIT}_answer.jsonl \
    $DATA/stanford_parsing_output/${SPLIT}_question.jsonl
done
```

##### Run AllenNLP semantic role tagger

The semantic role tagger is run on each sentence. First obtain sentences from the CoreNLP outputs:

```shell
for SPLIT in train valid test
do
  python split_sentence_from_corenlp.py $DATA/stanford_parsing_output/${SPLIT}_answer.jsonl \
    $DATA/stanford_parsing_output/${SPLIT}_answer_sents.jsonl
done
```

Then run the AllenNLP semantic role tagger:

```shell
mkdir $DATA/allennlp_srl_output
for SPLIT in train valid test
do 
  python allennlp_srl.py $DATA/stanford_parsing_output/${SPLIT}_answer_sents.jsonl \
    $DATA/allennlp_srl_output/${SPLIT}_answer.jsonl
done
```

### Build hybrid graphs, question focuses, question template, and create binarized data

##### Build hybrid graphs

```shell
mkdir $DATA/hybrid_graph
for SPLIT in train valid test
do 
  python build_hybrid_graph.py \
    $DATA/stanford_parsing_output/${SPLIT}_answer.jsonl \
    $DATA/allennlp_srl_output/${SPLIT}_answer.jsonl \
    $DATA/raw/${SPLIT}.jsonl \
    $DATA/hybrid_graph/${SPLIT}.jsonl
done
```

##### Build question focuses

First convert the CoreNLP parsing output:

```shell
for SPLIT in train valid test
do 
  python convert_corenlp_output.py $DATA/stanford_parsing_output/${SPLIT}_answer.jsonl \
    $DATA/stanford_parsing_output/${SPLIT}_answer_converted.jsonl
  python convert_corenlp_output.py $DATA/stanford_parsing_output/${SPLIT}_question.jsonl \
    $DATA/stanford_parsing_output/${SPLIT}_question_converted.jsonl
done
```

Build oracle question focus:

```shell
mkdir -p $DATA/question_focus/oracle
for SPLIT in train valid test
do 
  python build_question_focus.py $DATA/stanford_parsing_output/${SPLIT}_question_converted.jsonl \
    $DATA/stanford_parsing_output/${SPLIT}_answer_converted.jsonl $DATA/question_focus/oracle/${SPLIT}.jsonl
done
```

Create graph focus prediction data:

```shell
mkdir -p $DATA/focus_prediction_data/oracle_raw
for SPLIT in train valid test
do 
  python create_graph_focus_prediction_data.py $DATA/question_focus/oracle/${SPLIT}.jsonl \
    $DATA/hybrid_graph/${SPLIT}.jsonl $DATA/focus_prediction_data/oracle_raw/${SPLIT}
done
```

##### Build question templates

First obtain words from the CoreNLP parsing output:

```shell
for SPLIT in train valid test
do 
  python tokenized_word_from_corenlp.py $DATA/stanford_parsing_output/${SPLIT}_answer.jsonl \
    $DATA/stanford_parsing_output/${SPLIT}_answer_words.jsonl
done
```

Build oracle templates:

```shell
wget -N 'https://conceptnet.s3.amazonaws.com/downloads/2019/numberbatch/numberbatch-en-19.08.txt.gz'
gzip -d numberbatch-en-19.08.txt.gz

mkdir -p $DATA/question_template/oracle
for SPLIT in train valid test
do 
  mkdir $DATA/tmp
  python build_question_template.py $DATA/stanford_parsing_output/${SPLIT}_question_converted.jsonl \
    $DATA/stanford_parsing_output/${SPLIT}_answer_words.jsonl $DATA/tmp $DATA/question_template/oracle/${SPLIT}.jsonl
  rm -rf $DATA/tmp
done
```

Predict question types. Please download the type prediction models from [here]()
and put them under `$MODEL/prediction_models/type_classifiers`.

```shell
mkdir $DATA/predicted_types
mkdir $DATA/raw_text
for SPLIT in train valid test
do
  python convert_to_text.py $DATA/raw/${SPLIT}.jsonl $DATA/raw_text/${SPLIT}
  python ../prediction_scripts/type_prediction.py $MODEL/prediction_models/type_classifiers/question_input \
    $DATA/raw_text/${SPLIT}_question.txt $DATA/predicted_types/${SPLIT}.oracle_type
done

python ../prediction_scripts/type_prediction.py $MODEL/prediction_models/type_classifiers/answer_input/reddit \
  $DATA/raw_text/test_answer.txt $DATA/predicted_types/test.type
```

Create template data:

```shell
mkdir -p $DATA/template_generation_data/oracle_raw
for SPLIT in train valid test
do 
  python create_template_data.py $DATA/question_template/oracle/${SPLIT}.jsonl \
    $DATA/predicted_types/${SPLIT}.oracle_type $DATA/template_generation_data/oracle_raw/${SPLIT}
done
```

Get oracle exemplar and create data:

```shell
mkdir -p $DATA/question_exemplar/oracle
mkdir -p $DATA/exemplar_data/oracle
for SPLIT in train valid test
do
  python get_exemplar.py $DATA/template_generation_data/oracle_raw/${SPLIT}.source \
    $DATA/predicted_types/${SPLIT}.oracle_type exemplars/exemplars.txt \
    $DATA/question_exemplar/oracle/${SPLIT}.txt
  python create_bpe_exemplar_data.py $DATA/question_exemplar/oracle/${SPLIT}.txt \
    $DATA/predicted_types/${SPLIT}.oracle_type exemplars/exemplar_bpe.txt \
    $DATA/exemplar_data/oracle/${SPLIT}.bpe.source
done
```

Predict question exemplars. Please download the exemplar prediction models from [here](https://drive.google.com/drive/folders/17rRXei80hhhM3KiITKKLc0Vns7ZxXQVC?usp=sharing)
and put them under `$MODEL/prediction_models/exemplar_classifiers`.

```shell
mkdir -p $DATA/question_exemplar/prediction
mkdir -p $DATA/exemplar_data/prediction
python ../prediction_scripts/exemplar_prediction.py $MODEL/prediction_models/exemplar_classifiers/reddit \
  $DATA/focus_prediction_data/oracle_raw/test.bpe.source $DATA/predicted_types/test.type \
   $DATA/question_exemplar/prediction/test.txt
python create_bpe_exemplar_data.py $DATA/question_exemplar/prediction/test.txt \
  $DATA/predicted_types/test.type exemplars/exemplar_bpe.txt \
  $DATA/exemplar_data/prediction/test.bpe.source
```


##### Create binarized data

Data for JointGen.

```shell
mkdir -p $DATA/binarized_data/jointgen_data

for SPLIT in train valid test
do 
  cp $DATA/focus_prediction_data/oracle_raw/${SPLIT}.* $DATA/binarized_data/jointgen_data
  cp $DATA/template_generation_data/oracle_raw/${SPLIT}.bpe.target $DATA/binarized_data/jointgen_data
done

cp $DATA/predicted_types/* $DATA/binarized_data/jointgen_data

wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt'

fairseq-preprocess --source-lang source --target-lang target \
  --trainpref $DATA/binarized_data/jointgen_data/train.bpe \
  --validpref $DATA/binarized_data/jointgen_data/valid.bpe \
  --testpref $DATA/binarized_data/jointgen_data/test.bpe \
  --destdir $DATA/binarized_data/jointgen_data \
  --workers 60 --srcdict dict.txt --tgtdict dict.txt
```

Exemplar data for ExplGen.

```shell
mkdir -p $DATA/binarized_data/exemplar_data/oracle
mkdir -p $DATA/binarized_data/exemplar_data/prediction
for SPLIT in train valid test
do 
  cp $DATA/exemplar_data/oracle/* $DATA/binarized_data/exemplar_data/oracle
  cp $DATA/template_generation_data/oracle_raw/${SPLIT}.bpe.target $DATA/binarized_data/exemplar_data/oracle
done

cp $DATA/exemplar_data/prediction/* $DATA/binarized_data/exemplar_data/prediction
cp $DATA/template_generation_data/oracle_raw/test.bpe.target $DATA/binarized_data/exemplar_data/prediction

fairseq-preprocess --source-lang source --target-lang target \
  --trainpref $DATA/binarized_data/exemplar_data/oracle/train.bpe \
  --validpref $DATA/binarized_data/exemplar_data/oracle/valid.bpe \
  --destdir $DATA/binarized_data/exemplar_data/oracle \
  --workers 60 --srcdict dict.txt --tgtdict dict.txt
  
fairseq-preprocess --source-lang source --target-lang target \
  --testpref $DATA/binarized_data/exemplar_data/prediction/test.bpe \
  --destdir $DATA/binarized_data/exemplar_data/prediction \
  --workers 60 --srcdict dict.txt --tgtdict dict.txt
```

Template generation data for TplGen.

```shell
mkdir $DATA/binarized_data/tplgen_data

for SPLIT in train valid test
do 
  cp $DATA/focus_prediction_data/oracle_raw/${SPLIT}.* $DATA/binarized_data/tplgen_data
  cp $DATA/template_generation_data/oracle_raw/${SPLIT}.bpe.source $DATA/binarized_data/tplgen_data/${SPLIT}.bpe.target
done

cp $DATA/predicted_types/* $DATA/binarized_data/tplgen_data

fairseq-preprocess --source-lang source --target-lang target \
  --trainpref $DATA/binarized_data/tplgen_data/train.bpe \
  --validpref $DATA/binarized_data/tplgen_data/valid.bpe \
  --testpref $DATA/binarized_data/tplgen_data/test.bpe \
  --destdir $DATA/binarized_data/tplgen_data \
  --workers 60 --srcdict dict.txt --tgtdict dict.txt
```

##### template->question data

First obtain predicted focus from trained template generation model. 
Download the template generation model from [here](https://drive.google.com/drive/folders/1cppLYeVWpVYrrpMysJnnUhuKa-Q9_Tol?usp=sharing) and put it under `$MODEL/generation_models/tplgen_template_generation`.

```shell
mkdir $DATA/predicted_focus
for SPLIT in train valid
do 
  python ../prediction_scripts/joint_focus_node_prediction.py $DATA/binarized_data/tplgen_data \
    --path $MODEL/generation_models/tplgen_template_generation/model.pt --user-dir ../fairseq_models/jointgen \
    --task joint_generation --qt oracle_type --gen-subset ${SPLIT} \
    --exemplar-data $DATA/binarized_data/exemplar_data/oracle --results-path $DATA/predicted_focus/${SPLIT}.focus_prob
done

python ../prediction_scripts/joint_focus_node_prediction.py $DATA/binarized_data/tplgen_data \
--path $MODEL/generation_models/tplgen_template_generation/model.pt --user-dir ../fairseq_models/jointgen \
--task joint_generation --qt type --gen-subset test \
--exemplar-data $DATA/binarized_data/exemplar_data/prediction --results-path $DATA/predicted_focus/test.focus_prob
```

Create data and binarize.

```shell
for SPLIT in train valid test
do 
  python create_predicted_focus_data.py $DATA/predicted_focus/${SPLIT}.focus_prob \
    $DATA/binarized_data/tplgen_data/${SPLIT} $DATA/question_focus/oracle/${SPLIT}.jsonl \
    $DATA/stanford_parsing_output/${SPLIT}_answer_converted.jsonl $DATA/predicted_focus/${SPLIT}
  cp $DATA/template_generation_data/oracle_raw/${SPLIT}.bpe.target $DATA/predicted_focus
done

fairseq-preprocess --source-lang source --target-lang target \
  --trainpref $DATA/predicted_focus/train.bpe \
  --validpref $DATA/predicted_focus/valid.bpe \
  --testpref $DATA/predicted_focus/test.bpe \
  --destdir $DATA/binarized_data/focus_data/tplgen \
  --workers 60 --srcdict dict.txt --tgtdict dict.txt
```

Template generation.

```shell
cd ../gen_scripts
./tplgen_template_generation.sh $DATA/output/tplgen_template
cd ..
python convert_output.py --generate-dir $DATA/output/tplgen_template

mkdir -p $DATA/binarized_data/template_data/tplgen
cp $DATA/output/tplgen_template/bpe-test.txt $DATA/binarized_data/template_data/tplgen/test.bpe.source
cp $DATA/template_generation_data/oracle_raw/test.bpe.target $DATA/binarized_data/template_data/tplgen

fairseq-preprocess --source-lang source --target-lang target \
  --testpref $DATA/binarized_data/template_data/tplgen/test.bpe \
  --destdir $DATA/binarized_data/template_data/tplgen \
  --workers 60 --srcdict dict.txt --tgtdict dict.txt
```

##### Exemplar for 9 types

Predict the top 9 types and corresponding exemplars.

```shell
python ../prediction_scripts/type_prediction_topk.py $MODEL/prediction_models/type_classifiers/answer_input/reddit \
  $DATA/raw_text/test_answer.txt $DATA/control_types
  
mkdir -p $DATA/question_exemplar/control
mkdir -p $DATA/exemplar_data/control
mkdir -p $DATA/binarized_data/exemplar_data/control_exemplar
for i in 1 2 3 4 5 6 7 8 9
do
  cp $DATA/control_types/control_type${i} $DATA/binarized_data/jointgen_data/test.control_type{i}
  cp $DATA/control_types/control_type${i} $DATA/binarized_data/tplgen_data/test.control_type{i}
  python ../prediction_scripts/exemplar_prediction.py $MODEL/prediction_models/exemplar_classifiers/reddit \
    $DATA/focus_prediction_data/oracle_raw/test.bpe.source $DATA/control_types/control_type${i} \
    $DATA/question_exemplar/control/control_type${i}
  mkdir -p $DATA/exemplar_data/control/type${i}
  python create_bpe_exemplar_data.py $DATA/question_exemplar/control/control_type${i} \
    $DATA/control_types/control_type${i} exemplars/exemplar_bpe.txt \
    $DATA/exemplar_data/control/type${i}/test.bpe.source
  mkdir -p $DATA/binarized_data/exemplar_data/control_exemplar/type${i}
  cp $DATA/exemplar_data/control/type${i}/* $DATA/binarized_data/exemplar_data/control_exemplar/type${i}
  cp $DATA/template_generation_data/oracle_raw/test.bpe.target $DATA/binarized_data/exemplar_data/control_exemplar/type${i}
  fairseq-preprocess --source-lang source --target-lang target \
    --testpref $DATA/binarized_data/exemplar_data/control_exemplar/type${i}/test.bpe \
    --destdir $DATA/binarized_data/exemplar_data/control_exemplar/type${i} \
    --workers 60 --srcdict dict.txt --tgtdict dict.txt
done
```

Predict focus for 9 types.

```shell
for i in 1 2 3 4 5 6 7 8 9
do
  mkdir -p $DATA/control_focus/type${i}
  python ../prediction_scripts/joint_focus_node_prediction.py $DATA/binarized_data/tplgen_data \
    --path $MODEL/generation_models/tplgen_template_generation/model.pt --user-dir ../fairseq_models/jointgen \
    --task joint_generation --qt control_type${i} --gen-subset test \
    --exemplar-data $DATA/binarized_data/exemplar_data/control_exemplar/type${i} --results-path $DATA/control_focus/type${i}/test.focus_prob
  python create_predicted_focus_data.py $DATA/control_focus/type${i}/test.focus_prob \
    $DATA/binarized_data/tplgen_data/test $DATA/question_focus/oracle/test.jsonl \
    $DATA/stanford_parsing_output/test_answer_converted.jsonl $DATA/control_focus/type${i}/test
  cp $DATA/template_generation_data/oracle_raw/${SPLIT}.bpe.target $DATA/control_focus/type${i}
  fairseq-preprocess --source-lang source --target-lang target \
    --testpref $DATA/control_focus/type${i}/test.bpe \
    --destdir $DATA/binarized_data/focus_data/control_focus/type${i} \
    --workers 60 --srcdict dict.txt --tgtdict dict.txt
done
```

Template generation for 9 types.
```shell
cd ../gen_scripts
./tplgen_template_generation_9types.sh $DATA/output/control_template
cd ..
python convert_output.py --generate-dir $DATA/output/tplgen_template/type*

for i in 1 2 3 4 5 6 7 8 9
do
  mkdir -p $DATA/binarized_data/template_data/control_template/type${i}
  cp $DATA/output/tplgen_template/type${i}/bpe-test.txt $DATA/binarized_data/template_data/control_template/type${i}/test.bpe.source
  cp $DATA/template_generation_data/oracle_raw/test.bpe.target $DATA/binarized_data/template_data/control_template/type${i}
  fairseq-preprocess --source-lang source --target-lang target \
    --testpref $DATA/binarized_data/template_data/control_template/type${i}/test.bpe \
    --destdir $DATA/binarized_data/template_data/control_template/type${i} \
    --workers 60 --srcdict dict.txt --tgtdict dict.txt
done
```

