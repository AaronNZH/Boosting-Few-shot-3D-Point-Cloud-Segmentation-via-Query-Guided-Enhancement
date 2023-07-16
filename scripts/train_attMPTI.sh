GPU_ID=0

DATASET='s3dis'
SPLIT=0
DATA_PATH='./datasets/S3DIS/blocks_bs1_s1'
SAVE_PATH='./log_s3dis/'

NUM_POINTS=2048
PC_ATTRIBS='xyzrgbXYZ'
EDGECONV_WIDTHS='[[64,64], [64, 64], [64, 64]]'
MLP_WIDTHS='[512, 256]'
K=20
BASE_WIDTHS='[128, 64]'

PRETRAIN_CHECKPOINT='./log_s3dis/log_pretrain_s3dis_S0'
N_WAY=1
K_SHOT=1
N_QUERIES=1
N_TEST_EPISODES=200

NUM_ITERS=30000
EVAL_INTERVAL=2000
LR=0.001
DECAY_STEP=5000
DECAY_RATIO=0.5

N_SUBPROTOTYPES=100  # N_SUBPROTOTYPES=150 if 2 way 5 shot
N_TEACHER_SUBPROTOTYPES=100
K_CONNECT=200
SIM_FUNCTION='gaussian'
SIGMA=1  # SIGMA=5 if scannet

args=(--phase 'mptitrain' --dataset "${DATASET}" --cvfold $SPLIT
      --data_path  "$DATA_PATH" --save_path "$SAVE_PATH"
      --pretrain_checkpoint_path "$PRETRAIN_CHECKPOINT"
      --n_subprototypes $N_SUBPROTOTYPES --n_teacher_subprototypes $N_TEACHER_SUBPROTOTYPES
      --k_connect $K_CONNECT --dist_method "$SIM_FUNCTION" --mpti_sigma $SIGMA
      --pc_npts $NUM_POINTS --pc_attribs "$PC_ATTRIBS" --pc_augm
      --edgeconv_widths "$EDGECONV_WIDTHS" --dgcnn_k $K 
      --dgcnn_mlp_widths "$MLP_WIDTHS" --base_widths "$BASE_WIDTHS" 
      --n_iters $NUM_ITERS --eval_interval $EVAL_INTERVAL --batch_size 1
      --lr $LR  --step_size $DECAY_STEP --gamma $DECAY_RATIO
      --n_way $N_WAY --k_shot $K_SHOT --n_queries $N_QUERIES --n_episode_test $N_TEST_EPISODES
      --use_bpa --use_hr)

CUDA_VISIBLE_DEVICES=$GPU_ID python main.py "${args[@]}"
