GPU_ID=0 

ARCH=vgg16_bn
DATASET=cifar10


MANUAL_SEED=1
############################### 
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
SAVE_DIR='./log/'${DATASET}_${ARCH}_${LOSS}_${VERSION_SH}_MOM${MOM}_GAMMA${GAMMA_FOCAL_LOSS}_SPLIT_old${OLD}_mat_scaling$
VERSION=1
mkdir -p ./log


### eval
python -u main.py --arch ${ARCH} --gpu-id ${GPU_ID} --sat-momentum ${MOM} \
       --gamma-focal-loss ${GAMMA_FOCAL_LOSS} --alpha-focal-loss ${ALPHA_FOCAL_LOSS}\
       --matrixScaling \
       --cluster --clusterLess \
       --dynamic --version ${VERSION} \
       --old ${OLD} \
       --version_SAT_original --version_FOCALinSAT --version_FOCALinGT --version_changingWithIdk \
       --manualSeed ${MANUAL_SEED} \
       --loss ${LOSS} --dataset ${DATASET} \
       --save ${SAVE_DIR} --evaluate \
       2>&1 | tee -a ${SAVE_DIR}.log
       #--idkforpredator \
      # --manualSeed ${MANUAL_SEED}

#############################
##############################
#############################
##############################
#############################
##############################
LOSS=ce
PRETRAIN=0
MOM=0.9
SAVE_DIR='./log/'${DATASET}_${ARCH}_${LOSS}_ES${PRETRAIN}_SPLIT_mat_scaling$

###########

# If IDKPREDATOR=False remove the label --idkforpredator

mkdir -p ./log

### eval
python -u main.py --arch ${ARCH} --gpu-id ${GPU_ID} \
       --manualSeed ${MANUAL_SEED} \
       --matrixScaling \
       --cluster --clusterLess \
       --loss ${LOSS} --dataset ${DATASET} \
       --save ${SAVE_DIR} --evaluate \
       2>&1 | tee -a ${SAVE_DIR}.log
       #--idkforpredator \



MANUAL_SEED=2
###############################
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
SAVE_DIR='./log/'${DATASET}_${ARCH}_${LOSS}_${VERSION_SH}_MOM${MOM}_GAMMA${GAMMA_FOCAL_LOSS}_SPLIT_old${OLD}_mat_scaling$
VERSION=1
mkdir -p ./log


### eval
python -u main.py --arch ${ARCH} --gpu-id ${GPU_ID} --sat-momentum ${MOM} \
       --gamma-focal-loss ${GAMMA_FOCAL_LOSS} --alpha-focal-loss ${ALPHA_FOCAL_LOSS}\
       --matrixScaling \
       --cluster --clusterLess \
       --dynamic --version ${VERSION} \
       --old ${OLD} \
       --version_SAT_original --version_FOCALinSAT --version_FOCALinGT --version_changingWithIdk \
       --manualSeed ${MANUAL_SEED} \
       --loss ${LOSS} --dataset ${DATASET} \
       --save ${SAVE_DIR} --evaluate \
       2>&1 | tee -a ${SAVE_DIR}.log
       #--idkforpredator \
      # --manualSeed ${MANUAL_SEED}

#############################
##############################
#############################
##############################
#############################
##############################
LOSS=ce
PRETRAIN=0
MOM=0.9
SAVE_DIR='./log/'${DATASET}_${ARCH}_${LOSS}_ES${PRETRAIN}_SPLIT_mat_scaling$

###########

# If IDKPREDATOR=False remove the label --idkforpredator

mkdir -p ./log

### eval
python -u main.py --arch ${ARCH} --gpu-id ${GPU_ID} \
       --manualSeed ${MANUAL_SEED} \
       --matrixScaling \
       --cluster --clusterLess \
       --loss ${LOSS} --dataset ${DATASET} \
       --save ${SAVE_DIR} --evaluate \
       2>&1 | tee -a ${SAVE_DIR}.log
       #--idkforpredator \
MANUAL_SEED=3
###############################
###############################
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
SAVE_DIR='./log/'${DATASET}_${ARCH}_${LOSS}_${VERSION_SH}_MOM${MOM}_GAMMA${GAMMA_FOCAL_LOSS}_SPLIT_old${OLD}_mat_scaling$
VERSION=1
mkdir -p ./log


### eval
python -u main.py --arch ${ARCH} --gpu-id ${GPU_ID} --sat-momentum ${MOM} \
       --gamma-focal-loss ${GAMMA_FOCAL_LOSS} --alpha-focal-loss ${ALPHA_FOCAL_LOSS}\
       --matrixScaling \
       --cluster --clusterLess \
       --dynamic --version ${VERSION} \
       --old ${OLD} \
       --version_SAT_original --version_FOCALinSAT --version_FOCALinGT --version_changingWithIdk \
       --manualSeed ${MANUAL_SEED} \
       --loss ${LOSS} --dataset ${DATASET} \
       --save ${SAVE_DIR} --evaluate \
       2>&1 | tee -a ${SAVE_DIR}.log
       #--idkforpredator \
      # --manualSeed ${MANUAL_SEED}

#############################
##############################
#############################
##############################
#############################
##############################
LOSS=ce
PRETRAIN=0
MOM=0.9
SAVE_DIR='./log/'${DATASET}_${ARCH}_${LOSS}_ES${PRETRAIN}_SPLIT_mat_scaling$


###########

# If IDKPREDATOR=False remove the label --idkforpredator

mkdir -p ./log

### eval
python -u main.py --arch ${ARCH} --gpu-id ${GPU_ID} \
       --manualSeed ${MANUAL_SEED} \
       --matrixScaling \
       --cluster --clusterLess \
       --loss ${LOSS} --dataset ${DATASET} \
       --save ${SAVE_DIR} --evaluate \
       2>&1 | tee -a ${SAVE_DIR}.log
       #--idkforpredator \
MANUAL_SEED=4
###############################
###############################
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
SAVE_DIR='./log/'${DATASET}_${ARCH}_${LOSS}_${VERSION_SH}_MOM${MOM}_GAMMA${GAMMA_FOCAL_LOSS}_SPLIT_old${OLD}_mat_scaling$
VERSION=1
mkdir -p ./log


### eval
python -u main.py --arch ${ARCH} --gpu-id ${GPU_ID} --sat-momentum ${MOM} \
       --gamma-focal-loss ${GAMMA_FOCAL_LOSS} --alpha-focal-loss ${ALPHA_FOCAL_LOSS}\
       --matrixScaling \
       --cluster --clusterLess \
       --dynamic --version ${VERSION} \
       --old ${OLD} \
       --version_SAT_original --version_FOCALinSAT --version_FOCALinGT --version_changingWithIdk \
       --manualSeed ${MANUAL_SEED} \
       --loss ${LOSS} --dataset ${DATASET} \
       --save ${SAVE_DIR} --evaluate \
       2>&1 | tee -a ${SAVE_DIR}.log
       #--idkforpredator \
      # --manualSeed ${MANUAL_SEED}

#############################
##############################
#############################
##############################
#############################
##############################
LOSS=ce
PRETRAIN=0
MOM=0.9
SAVE_DIR='./log/'${DATASET}_${ARCH}_${LOSS}_ES${PRETRAIN}_SPLIT_mat_scaling$

###########

# If IDKPREDATOR=False remove the label --idkforpredator

mkdir -p ./log

### eval
python -u main.py --arch ${ARCH} --gpu-id ${GPU_ID} \
       --manualSeed ${MANUAL_SEED} \
       --matrixScaling \
       --cluster --clusterLess \
       --loss ${LOSS} --dataset ${DATASET} \
       --save ${SAVE_DIR} --evaluate \
       2>&1 | tee -a ${SAVE_DIR}.log
       #--idkforpredator \
MANUAL_SEED=5
###############################
###############################
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
SAVE_DIR='./log/'${DATASET}_${ARCH}_${LOSS}_${VERSION_SH}_MOM${MOM}_GAMMA${GAMMA_FOCAL_LOSS}_SPLIT_old${OLD}_mat_scaling$
VERSION=1
mkdir -p ./log


### eval
python -u main.py --arch ${ARCH} --gpu-id ${GPU_ID} --sat-momentum ${MOM} \
       --gamma-focal-loss ${GAMMA_FOCAL_LOSS} --alpha-focal-loss ${ALPHA_FOCAL_LOSS}\
       --matrixScaling \
       --cluster --clusterLess \
       --dynamic --version ${VERSION} \
       --old ${OLD} \
       --version_SAT_original --version_FOCALinSAT --version_FOCALinGT --version_changingWithIdk \
       --manualSeed ${MANUAL_SEED} \
       --loss ${LOSS} --dataset ${DATASET} \
       --save ${SAVE_DIR} --evaluate \
       2>&1 | tee -a ${SAVE_DIR}.log
       #--idkforpredator \
      # --manualSeed ${MANUAL_SEED}

#############################
##############################
#############################
##############################
#############################
##############################
LOSS=ce
PRETRAIN=0
MOM=0.9
SAVE_DIR='./log/'${DATASET}_${ARCH}_${LOSS}_ES${PRETRAIN}_SPLIT_mat_scaling$

###########

# If IDKPREDATOR=False remove the label --idkforpredator

mkdir -p ./log

### eval
python -u main.py --arch ${ARCH} --gpu-id ${GPU_ID} \
       --manualSeed ${MANUAL_SEED} \
       --matrixScaling \
       --cluster --clusterLess \
       --loss ${LOSS} --dataset ${DATASET} \
       --save ${SAVE_DIR} --evaluate \
       2>&1 | tee -a ${SAVE_DIR}.log
       #--idkforpredator \


MANUAL_SEED=1
###############################
###############################
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
SAVE_DIR='./log/'${DATASET}_${ARCH}_${LOSS}_${VERSION_SH}_MOM${MOM}_GAMMA${GAMMA_FOCAL_LOSS}_SPLIT_old${OLD}_vect_scaling$
VERSION=1
mkdir -p ./log


### eval
python -u main.py --arch ${ARCH} --gpu-id ${GPU_ID} --sat-momentum ${MOM} \
       --gamma-focal-loss ${GAMMA_FOCAL_LOSS} --alpha-focal-loss ${ALPHA_FOCAL_LOSS}\
       --vectorScaling \
       --cluster --clusterLess \
       --dynamic --version ${VERSION} \
       --old ${OLD} \
       --version_SAT_original --version_FOCALinSAT --version_FOCALinGT --version_changingWithIdk \
       --manualSeed ${MANUAL_SEED} \
       --loss ${LOSS} --dataset ${DATASET} \
       --save ${SAVE_DIR} --evaluate \
       2>&1 | tee -a ${SAVE_DIR}.log
       #--idkforpredator \
      # --manualSeed ${MANUAL_SEED}


#############################
##############################
#############################
##############################
#############################
##############################
LOSS=ce
PRETRAIN=0
MOM=0.9
SAVE_DIR='./log/'${DATASET}_${ARCH}_${LOSS}_ES${PRETRAIN}_SPLIT_vect_scaling$

###########

# If IDKPREDATOR=False remove the label --idkforpredator

mkdir -p ./log

### eval
python -u main.py --arch ${ARCH} --gpu-id ${GPU_ID} \
       --manualSeed ${MANUAL_SEED} \
       --vectorScaling \
       --cluster --clusterLess \
       --loss ${LOSS} --dataset ${DATASET} \
       --save ${SAVE_DIR} --evaluate \
       2>&1 | tee -a ${SAVE_DIR}.log
       #--idkforpredator \



MANUAL_SEED=2
###############################
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
SAVE_DIR='./log/'${DATASET}_${ARCH}_${LOSS}_${VERSION_SH}_MOM${MOM}_GAMMA${GAMMA_FOCAL_LOSS}_SPLIT_old${OLD}_vect_scaling$
VERSION=1
mkdir -p ./log


### eval
python -u main.py --arch ${ARCH} --gpu-id ${GPU_ID} --sat-momentum ${MOM} \
       --gamma-focal-loss ${GAMMA_FOCAL_LOSS} --alpha-focal-loss ${ALPHA_FOCAL_LOSS}\
       --vectorScaling \
       --cluster --clusterLess \
       --dynamic --version ${VERSION} \
       --old ${OLD} \
       --version_SAT_original --version_FOCALinSAT --version_FOCALinGT --version_changingWithIdk \
       --manualSeed ${MANUAL_SEED} \
       --loss ${LOSS} --dataset ${DATASET} \
       --save ${SAVE_DIR} --evaluate \
       2>&1 | tee -a ${SAVE_DIR}.log
       #--idkforpredator \
      # --manualSeed ${MANUAL_SEED}

#############################
##############################
#############################
##############################
#############################
##############################
LOSS=ce
PRETRAIN=0
MOM=0.9
SAVE_DIR='./log/'${DATASET}_${ARCH}_${LOSS}_ES${PRETRAIN}_SPLIT_vect_scaling$

###########

# If IDKPREDATOR=False remove the label --idkforpredator

mkdir -p ./log

### eval
python -u main.py --arch ${ARCH} --gpu-id ${GPU_ID} \
       --manualSeed ${MANUAL_SEED} \
       --vectorScaling \
       --cluster --clusterLess \
       --loss ${LOSS} --dataset ${DATASET} \
       --save ${SAVE_DIR} --evaluate \
       2>&1 | tee -a ${SAVE_DIR}.log
       #--idkforpredator \
MANUAL_SEED=3
###############################
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
SAVE_DIR='./log/'${DATASET}_${ARCH}_${LOSS}_${VERSION_SH}_MOM${MOM}_GAMMA${GAMMA_FOCAL_LOSS}_SPLIT_old${OLD}_vect_scaling$
VERSION=1
mkdir -p ./log


### eval
python -u main.py --arch ${ARCH} --gpu-id ${GPU_ID} --sat-momentum ${MOM} \
       --gamma-focal-loss ${GAMMA_FOCAL_LOSS} --alpha-focal-loss ${ALPHA_FOCAL_LOSS}\
       --vectorScaling \
       --cluster --clusterLess \
       --dynamic --version ${VERSION} \
       --old ${OLD} \
       --version_SAT_original --version_FOCALinSAT --version_FOCALinGT --version_changingWithIdk \
       --manualSeed ${MANUAL_SEED} \
       --loss ${LOSS} --dataset ${DATASET} \
       --save ${SAVE_DIR} --evaluate \
       2>&1 | tee -a ${SAVE_DIR}.log
       #--idkforpredator \
      # --manualSeed ${MANUAL_SEED}

#############################
##############################
#############################
##############################
#############################
##############################
LOSS=ce
PRETRAIN=0
MOM=0.9
SAVE_DIR='./log/'${DATASET}_${ARCH}_${LOSS}_ES${PRETRAIN}_SPLIT_vect_scaling$

###########

# If IDKPREDATOR=False remove the label --idkforpredator

mkdir -p ./log

### eval
python -u main.py --arch ${ARCH} --gpu-id ${GPU_ID} \
       --manualSeed ${MANUAL_SEED} \
       --vectorScaling \
       --cluster --clusterLess \
       --loss ${LOSS} --dataset ${DATASET} \
       --save ${SAVE_DIR} --evaluate \
       2>&1 | tee -a ${SAVE_DIR}.log
       #--idkforpredator \
MANUAL_SEED=4
#
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
SAVE_DIR='./log/'${DATASET}_${ARCH}_${LOSS}_${VERSION_SH}_MOM${MOM}_GAMMA${GAMMA_FOCAL_LOSS}_SPLIT_old${OLD}_vect_scaling$
VERSION=1
mkdir -p ./log


### eval
python -u main.py --arch ${ARCH} --gpu-id ${GPU_ID} --sat-momentum ${MOM} \
       --gamma-focal-loss ${GAMMA_FOCAL_LOSS} --alpha-focal-loss ${ALPHA_FOCAL_LOSS}\
       --vectorScaling \
       --cluster --clusterLess \
       --dynamic --version ${VERSION} \
       --old ${OLD} \
       --version_SAT_original --version_FOCALinSAT --version_FOCALinGT --version_changingWithIdk \
       --manualSeed ${MANUAL_SEED} \
       --loss ${LOSS} --dataset ${DATASET} \
       --save ${SAVE_DIR} --evaluate \
       2>&1 | tee -a ${SAVE_DIR}.log
       #--idkforpredator \
      # --manualSeed ${MANUAL_SEED}

#############################
##############################
#############################
##############################
#############################
##############################
LOSS=ce
PRETRAIN=0
MOM=0.9
SAVE_DIR='./log/'${DATASET}_${ARCH}_${LOSS}_ES${PRETRAIN}_SPLIT_vect_scaling$

###########

# If IDKPREDATOR=False remove the label --idkforpredator

mkdir -p ./log

### eval
python -u main.py --arch ${ARCH} --gpu-id ${GPU_ID} \
       --manualSeed ${MANUAL_SEED} \
       --vectorScaling \
       --cluster --clusterLess \
       --loss ${LOSS} --dataset ${DATASET} \
       --save ${SAVE_DIR} --evaluate \
       2>&1 | tee -a ${SAVE_DIR}.log
       #--idkforpredator \
MANUAL_SEED=5
#
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
SAVE_DIR='./log/'${DATASET}_${ARCH}_${LOSS}_${VERSION_SH}_MOM${MOM}_GAMMA${GAMMA_FOCAL_LOSS}_SPLIT_old${OLD}_vect_scaling$
VERSION=1
mkdir -p ./log


### eval
python -u main.py --arch ${ARCH} --gpu-id ${GPU_ID} --sat-momentum ${MOM} \
       --gamma-focal-loss ${GAMMA_FOCAL_LOSS} --alpha-focal-loss ${ALPHA_FOCAL_LOSS}\
       --vectorScaling \
       --cluster --clusterLess \
       --dynamic --version ${VERSION} \
       --old ${OLD} \
       --version_SAT_original --version_FOCALinSAT --version_FOCALinGT --version_changingWithIdk \
       --manualSeed ${MANUAL_SEED} \
       --loss ${LOSS} --dataset ${DATASET} \
       --save ${SAVE_DIR} --evaluate \
       2>&1 | tee -a ${SAVE_DIR}.log
       #--idkforpredator \
      # --manualSeed ${MANUAL_SEED}

#############################
##############################
#############################
##############################
#############################
##############################
LOSS=ce
PRETRAIN=0
MOM=0.9
SAVE_DIR='./log/'${DATASET}_${ARCH}_${LOSS}_ES${PRETRAIN}_SPLIT_vect_scaling$

###########

# If IDKPREDATOR=False remove the label --idkforpredator

mkdir -p ./log

### eval
python -u main.py --arch ${ARCH} --gpu-id ${GPU_ID} \
       --manualSeed ${MANUAL_SEED} \
       --vectorScaling \
       --cluster --clusterLess \
       --loss ${LOSS} --dataset ${DATASET} \
       --save ${SAVE_DIR} --evaluate \
       2>&1 | tee -a ${SAVE_DIR}.log
       #--idkforpredator \



MANUAL_SEED=1
###############################

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
SAVE_DIR='./log/'${DATASET}_${ARCH}_${LOSS}_${VERSION_SH}_MOM${MOM}_GAMMA${GAMMA_FOCAL_LOSS}_SPLIT_old${OLD}_temp_scaling$
VERSION=1
mkdir -p ./log


### eval
python -u main.py --arch ${ARCH} --gpu-id ${GPU_ID} --sat-momentum ${MOM} \
       --gamma-focal-loss ${GAMMA_FOCAL_LOSS} --alpha-focal-loss ${ALPHA_FOCAL_LOSS}\
       --temperatureScaling \
       --cluster --clusterLess \
       --dynamic --version ${VERSION} \
       --old ${OLD} \
       --version_SAT_original --version_FOCALinSAT --version_FOCALinGT --version_changingWithIdk \
       --manualSeed ${MANUAL_SEED} \
       --loss ${LOSS} --dataset ${DATASET} \
       --save ${SAVE_DIR} --evaluate \
       2>&1 | tee -a ${SAVE_DIR}.log
       #--idkforpredator \
      # --manualSeed ${MANUAL_SEED}


#############################
##############################
#############################
##############################
#############################
##############################
LOSS=ce
PRETRAIN=0
MOM=0.9
SAVE_DIR='./log/'${DATASET}_${ARCH}_${LOSS}_ES${PRETRAIN}_SPLIT_temp_scaling$


###########

# If IDKPREDATOR=False remove the label --idkforpredator

mkdir -p ./log

### eval
python -u main.py --arch ${ARCH} --gpu-id ${GPU_ID} \
       --manualSeed ${MANUAL_SEED} \
       --temperatureScaling \
       --cluster --clusterLess \
       --loss ${LOSS} --dataset ${DATASET} \
       --save ${SAVE_DIR} --evaluate \
       2>&1 | tee -a ${SAVE_DIR}.log
       #--idkforpredator \



MANUAL_SEED=2
###############################

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
SAVE_DIR='./log/'${DATASET}_${ARCH}_${LOSS}_${VERSION_SH}_MOM${MOM}_GAMMA${GAMMA_FOCAL_LOSS}_SPLIT_old${OLD}_temp_scaling$
VERSION=1
mkdir -p ./log


### eval
python -u main.py --arch ${ARCH} --gpu-id ${GPU_ID} --sat-momentum ${MOM} \
       --gamma-focal-loss ${GAMMA_FOCAL_LOSS} --alpha-focal-loss ${ALPHA_FOCAL_LOSS}\
       --temperatureScaling \
       --cluster --clusterLess \
       --dynamic --version ${VERSION} \
       --old ${OLD} \
       --version_SAT_original --version_FOCALinSAT --version_FOCALinGT --version_changingWithIdk \
       --manualSeed ${MANUAL_SEED} \
       --loss ${LOSS} --dataset ${DATASET} \
       --save ${SAVE_DIR} --evaluate \
       2>&1 | tee -a ${SAVE_DIR}.log
       #--idkforpredator \
      # --manualSeed ${MANUAL_SEED}

#############################
##############################
#############################
##############################
#############################
##############################
LOSS=ce
PRETRAIN=0
MOM=0.9
SAVE_DIR='./log/'${DATASET}_${ARCH}_${LOSS}_ES${PRETRAIN}_SPLIT_temp_scaling$


###########

# If IDKPREDATOR=False remove the label --idkforpredator

mkdir -p ./log

### eval
python -u main.py --arch ${ARCH} --gpu-id ${GPU_ID} \
       --manualSeed ${MANUAL_SEED} \
       --temperatureScaling \
       --cluster --clusterLess \
       --loss ${LOSS} --dataset ${DATASET} \
       --save ${SAVE_DIR} --evaluate \
       2>&1 | tee -a ${SAVE_DIR}.log
       #--idkforpredator \
MANUAL_SEED=3
###############################

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
SAVE_DIR='./log/'${DATASET}_${ARCH}_${LOSS}_${VERSION_SH}_MOM${MOM}_GAMMA${GAMMA_FOCAL_LOSS}_SPLIT_old${OLD}_temp_scaling$
VERSION=1
mkdir -p ./log


### eval
python -u main.py --arch ${ARCH} --gpu-id ${GPU_ID} --sat-momentum ${MOM} \
       --gamma-focal-loss ${GAMMA_FOCAL_LOSS} --alpha-focal-loss ${ALPHA_FOCAL_LOSS}\
       --temperatureScaling \
       --cluster --clusterLess \
       --dynamic --version ${VERSION} \
       --old ${OLD} \
       --version_SAT_original --version_FOCALinSAT --version_FOCALinGT --version_changingWithIdk \
       --manualSeed ${MANUAL_SEED} \
       --loss ${LOSS} --dataset ${DATASET} \
       --save ${SAVE_DIR} --evaluate \
       2>&1 | tee -a ${SAVE_DIR}.log
       #--idkforpredator \
      # --manualSeed ${MANUAL_SEED}

#############################
##############################
#############################
##############################
#############################
##############################
LOSS=ce
PRETRAIN=0
MOM=0.9
SAVE_DIR='./log/'${DATASET}_${ARCH}_${LOSS}_ES${PRETRAIN}_SPLIT_temp_scaling$


###########

# If IDKPREDATOR=False remove the label --idkforpredator

mkdir -p ./log

### eval
python -u main.py --arch ${ARCH} --gpu-id ${GPU_ID} \
       --manualSeed ${MANUAL_SEED} \
       --temperatureScaling \
       --cluster --clusterLess \
       --loss ${LOSS} --dataset ${DATASET} \
       --save ${SAVE_DIR} --evaluate \
       2>&1 | tee -a ${SAVE_DIR}.log
       #--idkforpredator \
MANUAL_SEED=4
###############################

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
SAVE_DIR='./log/'${DATASET}_${ARCH}_${LOSS}_${VERSION_SH}_MOM${MOM}_GAMMA${GAMMA_FOCAL_LOSS}_SPLIT_old${OLD}_temp_scaling$
VERSION=1
mkdir -p ./log


### eval
python -u main.py --arch ${ARCH} --gpu-id ${GPU_ID} --sat-momentum ${MOM} \
       --gamma-focal-loss ${GAMMA_FOCAL_LOSS} --alpha-focal-loss ${ALPHA_FOCAL_LOSS}\
       --temperatureScaling \
       --cluster --clusterLess \
       --dynamic --version ${VERSION} \
       --old ${OLD} \
       --version_SAT_original --version_FOCALinSAT --version_FOCALinGT --version_changingWithIdk \
       --manualSeed ${MANUAL_SEED} \
       --loss ${LOSS} --dataset ${DATASET} \
       --save ${SAVE_DIR} --evaluate \
       2>&1 | tee -a ${SAVE_DIR}.log
       #--idkforpredator \
      # --manualSeed ${MANUAL_SEED}

#############################
##############################
#############################
##############################
#############################
##############################
LOSS=ce
PRETRAIN=0
MOM=0.9
SAVE_DIR='./log/'${DATASET}_${ARCH}_${LOSS}_ES${PRETRAIN}_SPLIT_temp_scaling$


###########

# If IDKPREDATOR=False remove the label --idkforpredator

mkdir -p ./log

### eval
python -u main.py --arch ${ARCH} --gpu-id ${GPU_ID} \
       --manualSeed ${MANUAL_SEED} \
       --temperatureScaling \
       --cluster --clusterLess \
       --loss ${LOSS} --dataset ${DATASET} \
       --save ${SAVE_DIR} --evaluate \
       2>&1 | tee -a ${SAVE_DIR}.log
       #--idkforpredator \
MANUAL_SEED=5
###############################

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
SAVE_DIR='./log/'${DATASET}_${ARCH}_${LOSS}_${VERSION_SH}_MOM${MOM}_GAMMA${GAMMA_FOCAL_LOSS}_SPLIT_old${OLD}_temp_scaling$
VERSION=1
mkdir -p ./log


### eval
python -u main.py --arch ${ARCH} --gpu-id ${GPU_ID} --sat-momentum ${MOM} \
       --gamma-focal-loss ${GAMMA_FOCAL_LOSS} --alpha-focal-loss ${ALPHA_FOCAL_LOSS}\
       --temperatureScaling \
       --cluster --clusterLess \
       --dynamic --version ${VERSION} \
       --old ${OLD} \
       --version_SAT_original --version_FOCALinSAT --version_FOCALinGT --version_changingWithIdk \
       --manualSeed ${MANUAL_SEED} \
       --loss ${LOSS} --dataset ${DATASET} \
       --save ${SAVE_DIR} --evaluate \
       2>&1 | tee -a ${SAVE_DIR}.log
       #--idkforpredator \
      # --manualSeed ${MANUAL_SEED}

#############################
##############################
#############################
##############################
#############################
##############################
LOSS=ce
PRETRAIN=0
MOM=0.9
SAVE_DIR='./log/'${DATASET}_${ARCH}_${LOSS}_ES${PRETRAIN}_SPLIT_temp_scaling$


###########

# If IDKPREDATOR=False remove the label --idkforpredator

mkdir -p ./log

### eval
python -u main.py --arch ${ARCH} --gpu-id ${GPU_ID} \
       --manualSeed ${MANUAL_SEED} \
       --temperatureScaling \
       --cluster --clusterLess \
       --loss ${LOSS} --dataset ${DATASET} \
       --save ${SAVE_DIR} --evaluate \
       2>&1 | tee -a ${SAVE_DIR}.log
       #--idkforpredator \
