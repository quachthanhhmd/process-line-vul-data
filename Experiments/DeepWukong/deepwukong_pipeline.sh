#!/bin/bash
unset RAY_USE_MULTIPROCESSING_CPU_COUNT

enable_archive=false
csv_BYPASS=false
xfg_BYPASS=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --enable-archive)
            enable_archive=true
            shift
            ;;
        --bypass-csv)
            csv_BYPASS=true
            shift
            ;;
        --bypass-xfg)
            xfg_BYPASS=true
            shift
            ;;
        *)
            ARGUMENT=${1}
            shift
            ;;
    esac
done

DS_NAME=$ARGUMENT

SLURM_TMPDIR="data/${DS_NAME}"
if [ -d "$SLURM_TMPDIR" ]; then
    rm -rf "$SLURM_TMPDIR"/*
fi
mkdir -p "$SLURM_TMPDIR" || { echo "Failed to create $SLURM_TMPDIR" >&2; exit 1; }
cd $SLURM_TMPDIR

# Source code preparation (prepare.sh에서 다운로드한 파일을 압축 해제)
# docker-compose.yml에서 /data/dataset으로 마운트됨
# 전체 데이터 (이미 압축 해제되어 있음)
if [ ! -d "/data/dataset/$DS_NAME/all_source_code" ] && [ -f "/data/dataset/$DS_NAME/all_source_code.tar.xz" ]; then
    echo "Extracting all_source_code..."
    tar -xf "/data/dataset/$DS_NAME/all_source_code.tar.xz" -C "/data/dataset/$DS_NAME/"
    mv "/data/dataset/$DS_NAME/source_code" "/data/dataset/$DS_NAME/all_source_code"
fi
ln -s "/data/dataset/$DS_NAME/all_source_code" "source_code"

#sGeneration of PDG
echo $PWD
echo "Generating PDG for project: $ARGUMENT"
if [ "$csv_BYPASS" == true ] && [ -f "/data/dataset/${DS_NAME}/${ARGUMENT}_csv.tar.gz" ]; then
    echo "Bypassing joern, extracting precomputed CSVs..."
    tar --use-compress-program=pigz -xvf "/data/dataset/${DS_NAME}/${ARGUMENT}_csv.tar.gz" -C .
else
    /tools/ReVeal/code-slicer/joern/joern-parse "./source_code" # TODO: joern-parse 경로 수정 필요
    mkdir csv && find parsed/source_code/ -mindepth 1 -maxdepth 1 -type d | xargs -I{} mv {} csv/ # mv source_code/ ../csv && cd .. # root@22995bd65f6d:/code/models/DeepWukong/data/all# mv parsed csv
    if [ "$enable_archive" = true ]; then
        tar -I 'pigz -p $(nproc) -6' -cf "/data/dataset/${DS_NAME}/${ARGUMENT}_csv.tar.gz" csv # tar -zcvf "Dataset/${project_name}_csv.tar.gz" csv
        # 압축 해제 시, tar --use-compress-program=pigz -xvf archive.tar.gz -C /path/to/dest
    fi
fi
cd -

#Generation of XFG
echo $PWD
echo "Generating XFG for project: $ARGUMENT"
if [ "$xfg_BYPASS" == true ] && [ -f "/data/dataset/${DS_NAME}/XFG_${ARGUMENT}.tar.gz" ]; then
    pushd $SLURM_TMPDIR
    tar --use-compress-program=pigz -xvf "/data/dataset/${DS_NAME}/XFG_${ARGUMENT}.tar.gz" -C .
    echo $PWD
    popd
    echo "Bypassing XFG generation, extracting precomputed XFGs..."
    echo $PWD
else
    PROJECT_NAME="all" SLURM_TMPDIR="." python3 "data_generator.py" -c "./config/config.yaml"
    if [ "$enable_archive" = true ]; then
        cd $SLURM_TMPDIR
        tar -I 'pigz -p $(nproc) -6' -cf "/data/dataset/${DS_NAME}/XFG_${ARGUMENT}.tar.gz" XFG # 원래 이건데 왼쪽으로 해봄 tar -zcf "/data/dataset/XFG_${project_name}.tar.gz" XFG
        cd -
    fi
fi

#Symbolize and Split Dataset
python3 "preprocess/dataset_generator.py" -c "./config/config.yaml"

if [ "$enable_archive" = true ]; then
    cd $SLURM_TMPDIR
    tar -I 'pigz -p $(nproc) -6' -cf "/data/dataset/${DS_NAME}/XFG_${ARGUMENT}__processed.tar.gz" XFG # tar -zcf "/data/dataset/XFG_${project_name}__processed.tar.gz" XFG
    cd -
fi

#Word Embedding Pretraining
python3 "preprocess/word_embedding.py" -c "./config/config.yaml"

# #Training and Testing
# SLURM_TMPDIR="." python3 "run.py" -c "./config/config.yaml"