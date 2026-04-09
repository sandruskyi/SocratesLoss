GPU_ID_TRAIN=3
GPU_ID_EVAL=0

MANUAL_SEED=1

ARCH=vgg16_bn
DATASET=cifar10 

#############################
##############################
#############################
##############################
#############################



##########
##########

LOSS=csc
MOM=0.9
m=0.999
k=300
rewards=1.0
t=0.5
mb=64
PRETRAIN=150



SAVE_DIR='./log/'${DATASET}_${ARCH}_${LOSS}_ES${PRETRAIN}_MOM${MOM}_m${m}_k${k}_rewards${rewards}_t${t}_mb${mb}_SPLIT$


# If IDKPREDATOR=False remove the label --idkforpredator

mkdir -p ./log


### train
CUDA_VISIBLE_DEVICES=${GPU_ID_TRAIN} python -u main.py --arch ${ARCH} --gpu-id ${GPU_ID_TRAIN} --pretrain ${PRETRAIN} --sat-momentum ${MOM} \
       --moco-m ${m} --moco-k ${k}  --rewards ${rewards} --moco-t ${t} --train-batch ${mb}\
       --debug \
       --manualSeed ${MANUAL_SEED} \
       --loss ${LOSS} \
       --cluster --extract-values-paper \
       --dataset ${DATASET} --save ${SAVE_DIR}  \
       2>&1 | tee -a ${SAVE_DIR}.log


### eval
CUDA_VISIBLE_DEVICES=${GPU_ID_EVAL} python -u main.py --arch ${ARCH} --gpu-id ${GPU_ID_EVAL} --pretrain ${PRETRAIN} \
       --moco-m ${m} --moco-k ${k}  --rewards ${rewards} --moco-t ${t} --train-batch ${mb}\
       --debug \
       --manualSeed ${MANUAL_SEED} \
       --cluster --extract-values-paper \
       --loss ${LOSS} --dataset ${DATASET} \
       --save ${SAVE_DIR} --evaluate \
       2>&1 | tee -a ${SAVE_DIR}.log

##########
##########

LOSS=sat
MOM=0.9
PRETRAIN=0
SAVE_DIR='./log/'${DATASET}_${ARCH}_${LOSS}_ES${PRETRAIN}_MOM${MOM}_SPLIT$


# If IDKPREDATOR=False remove the label --idkforpredator

mkdir -p ./log


### train
CUDA_VISIBLE_DEVICES=${GPU_ID_TRAIN} python -u main.py --arch ${ARCH} --gpu-id ${GPU_ID_TRAIN} --pretrain ${PRETRAIN} --sat-momentum ${MOM} \
       --debug \
       --manualSeed ${MANUAL_SEED} \
       --loss ${LOSS} \
       --cluster --extract-values-paper \
       --dataset ${DATASET} --save ${SAVE_DIR}  \
       2>&1 | tee -a ${SAVE_DIR}.log


### eval
CUDA_VISIBLE_DEVICES=${GPU_ID_EVAL} python -u main.py --arch ${ARCH} --gpu-id ${GPU_ID_EVAL} \
       --debug \
       --manualSeed ${MANUAL_SEED} \
       --cluster --extract-values-paper \
       --loss ${LOSS} --dataset ${DATASET} \
       --save ${SAVE_DIR} --evaluate \
       2>&1 | tee -a ${SAVE_DIR}.log


###########
###########

LOSS=focal
MOM=0.9
PRETRAIN=0


# FOCAL
GAMMA_FOCAL_LOSS=1
ALPHA_FOCAL_LOSS=1
SAVE_DIR='./log/'${DATASET}_${ARCH}_${LOSS}_ES${PRETRAIN}_GAMMA${GAMMA_FOCAL_LOSS}_SPLIT$



# If IDKPREDATOR=False remove the label --idkforpredator

mkdir -p ./log

### train
CUDA_VISIBLE_DEVICES=${GPU_ID_TRAIN} python -u main.py --arch ${ARCH} --gpu-id ${GPU_ID_TRAIN} --pretrain ${PRETRAIN} --sat-momentum ${MOM} \
       --gamma-focal-loss ${GAMMA_FOCAL_LOSS} --alpha-focal-loss ${ALPHA_FOCAL_LOSS}\
       --debug \
       --cluster --extract-values-paper \
       --manualSeed ${MANUAL_SEED} \
       --loss ${LOSS} \
       --dataset ${DATASET} --save ${SAVE_DIR}  \
       2>&1 | tee -a ${SAVE_DIR}.log



### eval
CUDA_VISIBLE_DEVICES=${GPU_ID_EVAL} python -u main.py --arch ${ARCH} --gpu-id ${GPU_ID_EVAL} \
       --debug \
       --cluster --extract-values-paper \
       --gamma-focal-loss ${GAMMA_FOCAL_LOSS} --alpha-focal-loss ${ALPHA_FOCAL_LOSS}\
       --manualSeed ${MANUAL_SEED} \
       --loss ${LOSS} --dataset ${DATASET} \
       --save ${SAVE_DIR} --evaluate \
       2>&1 | tee -a ${SAVE_DIR}.log



###########
###########
LOSS=brierScore
PRETRAIN=0
MOM=0.9
SAVE_DIR='./log/'${DATASET}_${ARCH}_${LOSS}_ES${PRETRAIN}_SPLIT$


###########

# If IDKPREDATOR=False remove the label --idkforpredator

mkdir -p ./log

### train
CUDA_VISIBLE_DEVICES=${GPU_ID_TRAIN} python -u main.py --arch ${ARCH} --gpu-id ${GPU_ID_TRAIN} --pretrain ${PRETRAIN} --sat-momentum ${MOM} \
       --manualSeed ${MANUAL_SEED} \
       --cluster --extract-values-paper \
       --loss ${LOSS} \
       --dataset ${DATASET} --save ${SAVE_DIR}  \
       2>&1 | tee -a ${SAVE_DIR}.log
        # --idkforpredator

### eval
CUDA_VISIBLE_DEVICES=${GPU_ID_EVAL} python -u main.py --arch ${ARCH} --gpu-id ${GPU_ID_EVAL} \
       --manualSeed ${MANUAL_SEED} \
       --cluster --extract-values-paper \
       --loss ${LOSS} --dataset ${DATASET} \
       --save ${SAVE_DIR} --evaluate \
       2>&1 | tee -a ${SAVE_DIR}.log
       #--idkforpredator \



###########
###########
LOSS=dece
PRETRAIN=0
MOM=0.9
SAVE_DIR='./log/'${DATASET}_${ARCH}_${LOSS}_ES${PRETRAIN}_SPLIT$


###########

# If IDKPREDATOR=False remove the label --idkforpredator

mkdir -p ./log

### train
CUDA_VISIBLE_DEVICES=${GPU_ID_TRAIN} python -u main.py --arch ${ARCH} --gpu-id ${GPU_ID_TRAIN} --pretrain ${PRETRAIN} --sat-momentum ${MOM} \
       --manualSeed ${MANUAL_SEED} \
       --cluster --extract-values-paper \
       --loss ${LOSS} \
       --dataset ${DATASET} --save ${SAVE_DIR}  \
       2>&1 | tee -a ${SAVE_DIR}.log
        # --idkforpredator

### eval
CUDA_VISIBLE_DEVICES=${GPU_ID_EVAL} python -u main.py --arch ${ARCH} --gpu-id ${GPU_ID_EVAL} \
       --manualSeed ${MANUAL_SEED} \
       --cluster --extract-values-paper \
       --loss ${LOSS} --dataset ${DATASET} \
       --save ${SAVE_DIR} --evaluate \
       2>&1 | tee -a ${SAVE_DIR}.log
       #--idkforpredator \


###########
###########

LOSS=focalAdaptive
MOM=0.9

# FOCAL
GAMMA_FOCAL_LOSS=1
ALPHA_FOCAL_LOSS=1
SAVE_DIR='./log/'${DATASET}_${ARCH}_${LOSS}_ES${PRETRAIN}_GAMMA${GAMMA_FOCAL_LOSS}_SPLIT$



# If IDKPREDATOR=False remove the label --idkforpredator

mkdir -p ./log

### train
CUDA_VISIBLE_DEVICES=${GPU_ID_TRAIN} python -u main.py --arch ${ARCH} --gpu-id ${GPU_ID_TRAIN} --pretrain ${PRETRAIN} --sat-momentum ${MOM} \
       --gamma-focal-loss ${GAMMA_FOCAL_LOSS} --alpha-focal-loss ${ALPHA_FOCAL_LOSS}\
       --debug \
       --cluster --extract-values-paper \
       --manualSeed ${MANUAL_SEED} \
       --loss ${LOSS} \
       --dataset ${DATASET} --save ${SAVE_DIR}  \
       2>&1 | tee -a ${SAVE_DIR}.log


### eval
CUDA_VISIBLE_DEVICES=${GPU_ID_EVAL} python -u main.py --arch ${ARCH} --gpu-id ${GPU_ID_EVAL} \
       --debug \
       --cluster --extract-values-paper \
       --gamma-focal-loss ${GAMMA_FOCAL_LOSS} --alpha-focal-loss ${ALPHA_FOCAL_LOSS}\
       --manualSeed ${MANUAL_SEED} \
       --loss ${LOSS} --dataset ${DATASET} \
       --save ${SAVE_DIR} --evaluate \
       2>&1 | tee -a ${SAVE_DIR}.log

###########
###########
LOSS=ce
PRETRAIN=0
MOM=0.9
SAVE_DIR='./log/'${DATASET}_${ARCH}_${LOSS}_ES${PRETRAIN}_SPLIT$


###########

# If IDKPREDATOR=False remove the label --idkforpredator

mkdir -p ./log


### train
CUDA_VISIBLE_DEVICES=${GPU_ID_TRAIN} python -u main.py --arch ${ARCH} --gpu-id ${GPU_ID_TRAIN} --pretrain ${PRETRAIN} --sat-momentum ${MOM} \
       --manualSeed ${MANUAL_SEED} \
       --cluster --extract-values-paper \
       --loss ${LOSS} \
       --dataset ${DATASET} --save ${SAVE_DIR}  \
       2>&1 | tee -a ${SAVE_DIR}.log
        # --idkforpredator


### eval
CUDA_VISIBLE_DEVICES=${GPU_ID_EVAL} python -u main.py --arch ${ARCH} --gpu-id ${GPU_ID_EVAL} \
       --manualSeed ${MANUAL_SEED} \
       --cluster --extract-values-paper \
       --loss ${LOSS} --dataset ${DATASET} \
       --save ${SAVE_DIR} --evaluate \
       2>&1 | tee -a ${SAVE_DIR}.log
       #--idkforpredator \

