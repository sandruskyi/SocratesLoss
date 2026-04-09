MANUAL_SEED=1
GPU_ID=2

ARCH=vgg16_bn
DATASET=cifar10



#############################
##############################
#############################
##############################
#############################
##############################
LOSS=Socrates
MOM=0.8
PRETRAIN=0

OLD=0


# FOCAL
GAMMA_FOCAL_LOSS=2
ALPHA_FOCAL_LOSS=1

###############################################################
###############################################################
###############################################################
VERSION_SH=1
SAVE_DIR='./log/'${DATASET}_${ARCH}_${LOSS}_${VERSION_SH}_MOM${MOM}_GAMMA${GAMMA_FOCAL_LOSS}_SPLIT_old${OLD}$
VERSION=1
mkdir -p ./log

### train
CUDA_VISIBLE_DEVICES=${GPU_ID} python -u main.py --arch ${ARCH} --gpu-id ${GPU_ID} --pretrain ${PRETRAIN} --sat-momentum ${MOM} \
       --gamma-focal-loss ${GAMMA_FOCAL_LOSS} --alpha-focal-loss ${ALPHA_FOCAL_LOSS}\
       --cluster --extract-values-paper \
       --dynamic --version ${VERSION} \
       --old ${OLD} \
       --version_SAT_original --version_FOCALinSAT --version_FOCALinGT --version_changingWithIdk \
       --manualSeed ${MANUAL_SEED} \
       --loss ${LOSS} \
       --dataset ${DATASET} --save ${SAVE_DIR}  \
       2>&1 | tee -a ${SAVE_DIR}.log
        # --idkforpredator
        # --manualSeed ${MANUAL_SEED}

GPU_ID=0

### eval
CUDA_VISIBLE_DEVICES=${GPU_ID} python -u main.py --arch ${ARCH} --gpu-id ${GPU_ID} --sat-momentum ${MOM} \
       --gamma-focal-loss ${GAMMA_FOCAL_LOSS} --alpha-focal-loss ${ALPHA_FOCAL_LOSS}\
       --cluster --extract-values-paper \
       --dynamic --version ${VERSION} \
       --old ${OLD} \
       --version_SAT_original --version_FOCALinSAT --version_FOCALinGT --version_changingWithIdk \
       --manualSeed ${MANUAL_SEED} \
       --loss ${LOSS} --dataset ${DATASET} \
       --save ${SAVE_DIR} --evaluate \
       2>&1 | tee -a ${SAVE_DIR}.log
       #--idkforpredator \
      # --manualSeed ${MANUAL_SEED}


