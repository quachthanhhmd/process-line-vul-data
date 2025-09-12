#!/bin/bash

project_name="<project_name>"
SLURM_TMPDIR = "<job_path>"
cd $SLURM_TMPDIR

#source_code extraction
tar -xf "/data/dataset/${project_name}_source_code.tar.gz" -C $SLURM_TMPDIR

#sGeneration of PDG

./joern-parse $SLURM_TMPDIR/source_code
tar -zcvf "/data/dataset/${project_name}_csv.tar.gz" csv

#Generation of XFG
python "/code/models/DeepWukong/data_generator.py" -c "/config/config.yaml"
cd $SLURM_TMPDIR
tar -zcf "/data/dataset/XFG_${project_name}.tar.gz" XFG

#Symbolize and Split Dataset
python "/code/models/DeepWukong/preprocess/dataset_generator.py" -c "/config/config.yaml"
cd $SLURM_TMPDIR
tar -zcf "/data/dataset/XFG_${project_name}__processed.tar.gz" XFG

#Word Embedding Pretraining
python "/code/models/DeepWukong/preprocess/word_embedding.py" -c "/config/config.yaml"

#Training and Testing
python "/code/models/DeepWukong/run.py" -c "/config/config.yaml"

