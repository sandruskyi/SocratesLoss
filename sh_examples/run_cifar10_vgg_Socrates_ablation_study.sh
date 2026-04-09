MANUAL_SEED=1
GPU_ID=5

ARCH=vgg16_bn
DATASET=cifar10

LOSS=Socrates
MOM=1
PRETRAIN=0
# FOCAL
GAMMA_FOCAL_LOSS=2
ALPHA_FOCAL_LOSS=1
VERSION=1 
OLD=0


#####################################
###############################################################
###############################################################
VERSION_SH=4
SAVE_DIR='./log/PAPER_'${DATASET}_${ARCH}_${LOSS}_${VERSION_SH}_ES${PRETRAIN}_MOM${MOM}_GAMMA${GAMMA_FOCAL_LOSS}_OLD${OLD}_ab_study$

mkdir -p ./log

### train
CUDA_VISIBLE_DEVICES=${GPU_ID} python -u main.py --arch ${ARCH} --gpu-id ${GPU_ID} --pretrain ${PRETRAIN} --sat-momentum ${MOM} \
       --gamma-focal-loss ${GAMMA_FOCAL_LOSS} --alpha-focal-loss ${ALPHA_FOCAL_LOSS}\
       --cluster \
       --dynamic --version ${VERSION} \
       --old ${OLD}  \
       --version_SAT_original --version_FOCALinGT --version_FOCALinSAT  \
       --manualSeed ${MANUAL_SEED} \
       --loss ${LOSS} \
       --dataset ${DATASET} --save ${SAVE_DIR}  \
       2>&1 | tee -a ${SAVE_DIR}.log
        # --idkforpredator
        # --manualSeed ${MANUAL_SEED}

### eval
CUDA_VISIBLE_DEVICES=${GPU_ID} python -u main.py --arch ${ARCH} --gpu-id ${GPU_ID} --sat-momentum ${MOM} \
       --gamma-focal-loss ${GAMMA_FOCAL_LOSS} --alpha-focal-loss ${ALPHA_FOCAL_LOSS}\
       --cluster \
       --dynamic --version ${VERSION} \
       --old ${OLD}  \
       --version_SAT_original --version_FOCALinGT --version_FOCALinSAT  \
       --manualSeed ${MANUAL_SEED} \
       --loss ${LOSS} --dataset ${DATASET} \
       --save ${SAVE_DIR} --evaluate \
       2>&1 | tee -a ${SAVE_DIR}.log
       #--idkforpredator \
      # --manualSeed ${MANUAL_SEED}

###############################################################
###############################################################
###############################################################


VERSION_SH=5
SAVE_DIR='./log/PAPER_'${DATASET}_${ARCH}_${LOSS}_${VERSION_SH}_ES${PRETRAIN}_MOM${MOM}_GAMMA${GAMMA_FOCAL_LOSS}_OLD${OLD}_ab_study$

mkdir -p ./log

### train
CUDA_VISIBLE_DEVICES=${GPU_ID} python -u main.py --arch ${ARCH} --gpu-id ${GPU_ID} --pretrain ${PRETRAIN} --sat-momentum ${MOM} \
       --gamma-focal-loss ${GAMMA_FOCAL_LOSS} --alpha-focal-loss ${ALPHA_FOCAL_LOSS}\
       --dynamic --version ${VERSION} \
       --cluster \
       --old ${OLD} \
       --version_SAT_original --version_FOCALinGT  \
       --manualSeed ${MANUAL_SEED} \
       --loss ${LOSS} \
       --dataset ${DATASET} --save ${SAVE_DIR}  \
       2>&1 | tee -a ${SAVE_DIR}.log
        # --idkforpredator
        # --manualSeed ${MANUAL_SEED}

### eval
CUDA_VISIBLE_DEVICES=${GPU_ID} python -u main.py --arch ${ARCH} --gpu-id ${GPU_ID} --sat-momentum ${MOM} \
       --gamma-focal-loss ${GAMMA_FOCAL_LOSS} --alpha-focal-loss ${ALPHA_FOCAL_LOSS}\
       --dynamic --version ${VERSION} \
       --cluster \
       --old ${OLD} \
       --version_SAT_original --version_FOCALinGT  \
       --manualSeed ${MANUAL_SEED} \
       --loss ${LOSS} --dataset ${DATASET} \
       --save ${SAVE_DIR} --evaluate \
       2>&1 | tee -a ${SAVE_DIR}.log
       #--idkforpredator \
      # --manualSeed ${MANUAL_SEED}


###############################################################
###############################################################
###############################################################


VERSION_SH=13
SAVE_DIR='./log/PAPER_'${DATASET}_${ARCH}_${LOSS}_${VERSION_SH}_ES${PRETRAIN}_MOM${MOM}_GAMMA${GAMMA_FOCAL_LOSS}_OLD${OLD}_ab_study$
mkdir -p ./log

### train
CUDA_VISIBLE_DEVICES=${GPU_ID} python -u main.py --arch ${ARCH} --gpu-id ${GPU_ID} --pretrain ${PRETRAIN} --sat-momentum ${MOM} \
       --gamma-focal-loss ${GAMMA_FOCAL_LOSS} --alpha-focal-loss ${ALPHA_FOCAL_LOSS}\
       --dynamic --version ${VERSION} \
       --version_SAT_original --version_changingWithIdk --version_FOCALinGT \
       --cluster \
       --old ${OLD} \
       --manualSeed ${MANUAL_SEED} \
       --loss ${LOSS} \
       --dataset ${DATASET} --save ${SAVE_DIR}  \
       2>&1 | tee -a ${SAVE_DIR}.log
        # --idkforpredator
        # --manualSeed ${MANUAL_SEED}

### eval
CUDA_VISIBLE_DEVICES=${GPU_ID} python -u main.py --arch ${ARCH} --gpu-id ${GPU_ID} --sat-momentum ${MOM} \
       --gamma-focal-loss ${GAMMA_FOCAL_LOSS} --alpha-focal-loss ${ALPHA_FOCAL_LOSS}\
       --dynamic --version ${VERSION} \
       --version_SAT_original --version_changingWithIdk --version_FOCALinGT \
       --cluster \
       --old ${OLD}  \
       --manualSeed ${MANUAL_SEED} \
       --loss ${LOSS} --dataset ${DATASET} \
       --save ${SAVE_DIR} --evaluate \
       2>&1 | tee -a ${SAVE_DIR}.log

###############################################################
###############################################################
###############################################################

# Ablation Study version 8

VERSION_SH=14
SAVE_DIR='./log/PAPER_'${DATASET}_${ARCH}_${LOSS}_${VERSION_SH}_ES${PRETRAIN}_MOM${MOM}_GAMMA${GAMMA_FOCAL_LOSS}_OLD${OLD}_ab_study$

mkdir -p ./log

### train
CUDA_VISIBLE_DEVICES=${GPU_ID} python -u main.py --arch ${ARCH} --gpu-id ${GPU_ID} --pretrain ${PRETRAIN} --sat-momentum ${MOM} \
       --gamma-focal-loss ${GAMMA_FOCAL_LOSS} --alpha-focal-loss ${ALPHA_FOCAL_LOSS}\
       --dynamic --version ${VERSION} \
       --cluster \
       --old ${OLD} \
       --version_SAT_original --version_FOCALinSAT  \
       --manualSeed ${MANUAL_SEED} \
       --loss ${LOSS} \
       --dataset ${DATASET} --save ${SAVE_DIR}  \
       2>&1 | tee -a ${SAVE_DIR}.log
        # --idkforpredator
        # --manualSeed ${MANUAL_SEED}

### eval
CUDA_VISIBLE_DEVICES=${GPU_ID} python -u main.py --arch ${ARCH} --gpu-id ${GPU_ID} --sat-momentum ${MOM} \
       --gamma-focal-loss ${GAMMA_FOCAL_LOSS} --alpha-focal-loss ${ALPHA_FOCAL_LOSS}\
       --dynamic --version ${VERSION} \
       --cluster \
       --old ${OLD}  \
       --version_SAT_original --version_FOCALinSAT  \
       --manualSeed ${MANUAL_SEED} \
       --loss ${LOSS} --dataset ${DATASET} \
       --save ${SAVE_DIR} --evaluate \
       2>&1 | tee -a ${SAVE_DIR}.log
       #--idkforpredator \
      # --manualSeed ${MANUAL_SEED}

###############################################################
###############################################################
###############################################################

# Ablation Study version 5

VERSION_SH=15
SAVE_DIR='./log/PAPER_'${DATASET}_${ARCH}_${LOSS}_${VERSION_SH}_ES${PRETRAIN}_MOM${MOM}_GAMMA${GAMMA_FOCAL_LOSS}_OLD${OLD}_ab_study$
mkdir -p ./log

### train
CUDA_VISIBLE_DEVICES=${GPU_ID} python -u main.py --arch ${ARCH} --gpu-id ${GPU_ID} --pretrain ${PRETRAIN} --sat-momentum ${MOM} \
       --gamma-focal-loss ${GAMMA_FOCAL_LOSS} --alpha-focal-loss ${ALPHA_FOCAL_LOSS}\
       --dynamic --version ${VERSION} \
       --version_SAT_original --version_changingWithIdk --version_FOCALinSAT \
       --cluster \
       --old ${OLD} \
       --manualSeed ${MANUAL_SEED} \
       --loss ${LOSS} \
       --dataset ${DATASET} --save ${SAVE_DIR}  \
       2>&1 | tee -a ${SAVE_DIR}.log
        # --idkforpredator
        # --manualSeed ${MANUAL_SEED}

### eval
CUDA_VISIBLE_DEVICES=${GPU_ID} python -u main.py --arch ${ARCH} --gpu-id ${GPU_ID} --sat-momentum ${MOM} \
       --gamma-focal-loss ${GAMMA_FOCAL_LOSS} --alpha-focal-loss ${ALPHA_FOCAL_LOSS}\
       --dynamic --version ${VERSION} \
       --version_SAT_original --version_changingWithIdk --version_FOCALinSAT \
       --cluster \
       --old ${OLD} \
       --manualSeed ${MANUAL_SEED} \
       --loss ${LOSS} --dataset ${DATASET} \
       --save ${SAVE_DIR} --evaluate \
       2>&1 | tee -a ${SAVE_DIR}.log

###############################################################
###############################################################
###############################################################

# Ablation Study version 3

VERSION_SH=16
SAVE_DIR='./log/PAPER_'${DATASET}_${ARCH}_${LOSS}_${VERSION_SH}_ES${PRETRAIN}_MOM${MOM}_GAMMA${GAMMA_FOCAL_LOSS}_OLD${OLD}_ab_study$
mkdir -p ./log

### train
CUDA_VISIBLE_DEVICES=${GPU_ID} python -u main.py --arch ${ARCH} --gpu-id ${GPU_ID} --pretrain ${PRETRAIN} --sat-momentum ${MOM} \
       --gamma-focal-loss ${GAMMA_FOCAL_LOSS} --alpha-focal-loss ${ALPHA_FOCAL_LOSS}\
       --dynamic --version ${VERSION} \
       --version_SAT_original --version_changingWithIdk \
       --cluster \
       --old ${OLD} \
       --manualSeed ${MANUAL_SEED} \
       --loss ${LOSS} \
       --dataset ${DATASET} --save ${SAVE_DIR}  \
       2>&1 | tee -a ${SAVE_DIR}.log
        # --idkforpredator
        # --manualSeed ${MANUAL_SEED}

### eval
CUDA_VISIBLE_DEVICES=${GPU_ID} python -u main.py --arch ${ARCH} --gpu-id ${GPU_ID} --sat-momentum ${MOM} \
       --gamma-focal-loss ${GAMMA_FOCAL_LOSS} --alpha-focal-loss ${ALPHA_FOCAL_LOSS}\
       --dynamic --version ${VERSION} \
       --version_SAT_original --version_changingWithIdk \
       --cluster \
       --old ${OLD}  \
       --manualSeed ${MANUAL_SEED} \
       --loss ${LOSS} --dataset ${DATASET} \
       --save ${SAVE_DIR} --evaluate \
       2>&1 | tee -a ${SAVE_DIR}.log




###############################################################
###############################################################
###############################################################

# Alternative Dynamic Uncertainty Penalties version 2

VERSION_SH=2
VERSION=2

SAVE_DIR='./log/PAPER_'${DATASET}_${ARCH}_${LOSS}_${VERSION_SH}_ES${PRETRAIN}_MOM${MOM}_GAMMA${GAMMA_FOCAL_LOSS}_OLD${OLD}_ab_study$
mkdir -p ./log


### train
CUDA_VISIBLE_DEVICES=${GPU_ID} python -u main.py --arch ${ARCH} --gpu-id ${GPU_ID} --pretrain ${PRETRAIN} --sat-momentum ${MOM} \
       --gamma-focal-loss ${GAMMA_FOCAL_LOSS} --alpha-focal-loss ${ALPHA_FOCAL_LOSS}\
       --cluster \
       --old ${OLD}  \
       --dynamic --version ${VERSION} \
       --version_SAT_original --version_FOCALinGT --version_FOCALinSAT --version_changingWithIdk \
       --manualSeed ${MANUAL_SEED} \
       --loss ${LOSS} \
       --dataset ${DATASET} --save ${SAVE_DIR}  \
       2>&1 | tee -a ${SAVE_DIR}.log
        # --idkforpredator
        # --manualSeed ${MANUAL_SEED}

### eval
CUDA_VISIBLE_DEVICES=${GPU_ID} python -u main.py --arch ${ARCH} --gpu-id ${GPU_ID} --sat-momentum ${MOM} \
       --gamma-focal-loss ${GAMMA_FOCAL_LOSS} --alpha-focal-loss ${ALPHA_FOCAL_LOSS}\
       --cluster \
       --old ${OLD} \
       --dynamic --version ${VERSION} \
       --version_SAT_original --version_FOCALinGT --version_FOCALinSAT --version_changingWithIdk \
       --manualSeed ${MANUAL_SEED} \
       --loss ${LOSS} --dataset ${DATASET} \
       --save ${SAVE_DIR} --evaluate \
       2>&1 | tee -a ${SAVE_DIR}.log
       #--idkforpredator \
      # --manualSeed ${MANUAL_SEED}


###############################################################
###############################################################
###############################################################

# Alternative Dynamic Uncertainty Penalties version 3

VERSION_SH=3
VERSION=3

SAVE_DIR='./log/PAPER_'${DATASET}_${ARCH}_${LOSS}_${VERSION_SH}_ES${PRETRAIN}_MOM${MOM}_GAMMA${GAMMA_FOCAL_LOSS}_OLD${OLD}_ab_study$
mkdir -p ./log


### train
CUDA_VISIBLE_DEVICES=${GPU_ID} python -u main.py --arch ${ARCH} --gpu-id ${GPU_ID} --pretrain ${PRETRAIN} --sat-momentum ${MOM} \
       --gamma-focal-loss ${GAMMA_FOCAL_LOSS} --alpha-focal-loss ${ALPHA_FOCAL_LOSS}\
       --cluster \
       --old ${OLD}  \
       --dynamic --version ${VERSION} \
       --version_SAT_original --version_FOCALinGT --version_FOCALinSAT --version_changingWithIdk \
       --manualSeed ${MANUAL_SEED} \
       --loss ${LOSS} \
       --dataset ${DATASET} --save ${SAVE_DIR}  \
       2>&1 | tee -a ${SAVE_DIR}.log
        # --idkforpredator
        # --manualSeed ${MANUAL_SEED}

### eval
CUDA_VISIBLE_DEVICES=${GPU_ID} python -u main.py --arch ${ARCH} --gpu-id ${GPU_ID} --sat-momentum ${MOM} \
       --gamma-focal-loss ${GAMMA_FOCAL_LOSS} --alpha-focal-loss ${ALPHA_FOCAL_LOSS}\
       --cluster \
       --old ${OLD}  \
       --dynamic --version ${VERSION} \
       --version_SAT_original --version_FOCALinGT --version_FOCALinSAT --version_changingWithIdk \
       --manualSeed ${MANUAL_SEED} \
       --loss ${LOSS} --dataset ${DATASET} \
       --save ${SAVE_DIR} --evaluate \
       2>&1 | tee -a ${SAVE_DIR}.log
       #--idkforpredator \
      # --manualSeed ${MANUAL_SEED}


