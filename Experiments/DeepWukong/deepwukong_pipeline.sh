#!/bin/bash

project_name="all"
SLURM_TMPDIR="data/${project_name}"
# cd $SLURM_TMPDIR

#source_code extraction
mv "../../Dataset/all_source_code/*" "${SLURM_TMPDIR}/source_code/*" # TODO: 다른 SW 테스트 할때는 tar -xf "Dataset/${project_name}_source_code.tar.gz" -C $SLURM_TMPDIR

#sGeneration of PDG
/data/ReVeal/code-slicer/joern/joern-parse "${SLURM_TMPDIR}/source_code" # TODO: joern-parse 경로 수정 필요

mv parsed "${SLURM_TMPDIR}/csv"
tar -I 'pigz -p $(nproc) -6' -cf "../../Dataset/${project_name}_csv.tar.gz" csv # tar -zcvf "Dataset/${project_name}_csv.tar.gz" csv

#Generation of XFG
python "data_generator.py" -c "./config/config.yaml"
tar -I 'pigz -p $(nproc) -6' -cf "../../Dataset/XFG_${project_name}.tar.gz" XFG # 원래 이건데 왼쪽으로 해봄 tar -zcf "/data/dataset/XFG_${project_name}.tar.gz" XFG

#Symbolize and Split Dataset
python "preprocess/dataset_generator.py" -c "/config/config.yaml"
cd $SLURM_TMPDIR
tar -I 'pigz -p $(nproc) -6' -cf "../../Dataset/XFG_${project_name}__processed.tar.gz" XFG # tar -zcf "/data/dataset/XFG_${project_name}__processed.tar.gz" XFG

#Word Embedding Pretraining
python "preprocess/word_embedding.py" -c "./config/config.yaml"

#Training and Testing
python "run.py" -c "./config/config.yaml"

