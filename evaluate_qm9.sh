ulimit -c unlimited

[ -z "${layers}" ] && layers=12
[ -z "${hidden_size}" ] && hidden_size=768
[ -z "${ffn_size}" ] && ffn_size=768
[ -z "${num_head}" ] && num_head=32
[ -z "${batch_size}" ] && batch_size=128
# [ -z "${update_freq}" ] && update_freq=1
[ -z "${seed}" ] && seed=42
# [ -z "${clip_norm}" ] && clip_norm=5
[ -z "${data_path}" ] && data_path='./datasets/'
[ -z "${save_path}" ] && save_path='/home/zhoujie/transformer-bias-add/logs/exp-dataset-qm9H-lr-7e-5-end_lr-1e-9-tsteps-600000-wsteps-60000-L12-D768-F768-H32-SLN-false-BS128-CLIP5-dp0.0-attn_dp0.1-wd0.0-dpp0.0/SEED42-TASK6-LOSS-L1-STD-no_std-RF-cls'
[ -z "${dataset_name}" ] && dataset_name="qm9H"
[ -z "${dropout}" ] && dropout=0.0
[ -z "${act_dropout}" ] && act_dropout=0.1
[ -z "${attn_dropout}" ] && attn_dropout=0.1
[ -z "${weight_decay}" ] && weight_decay=0.0
[ -z "${sandwich_ln}" ] && sandwich_ln="false"
[ -z "${droppath_prob}" ] && droppath_prob=0.1
[ -z "${noise_scale}" ] && noise_scale=0.2
[ -z "${mode_prob}" ] && mode_prob="0.2,0.2,0.6"
[ -z "${task_idx}" ] && task_idx=6  # 必须和你训练时一样
[ -z "${add_3d}" ] && add_3d="true"
[ -z "${no_2d}" ] && no_2d="true"
[ -z "${num_3d_bias_kernel}" ] && num_3d_bias_kernel=128


# 设置环境变量，防止 import 报错
export PYTHONPATH=$PYTHONPATH:$(pwd)/Transformer-M/Transformer-M/data

echo "开始评估..."

python evaluate.py \
    --user-dir $(realpath ./Transformer-M) \
    --data-path $data_path \
    --num-workers 16 --ddp-backend=legacy_ddp \
    --dataset-name $dataset_name \
    --batch-size $batch_size --data-buffer-size 20 \
    --task graph_prediction_qm9 \
    --criterion graph_prediction_qm9 \
    --arch transformer_m \
    --load-qm9 \
    --num-classes 1 \
    --task-idx $task_idx \
    --loss-type L1 --std-type no_std --readout-type cls \
    --encoder-layers $layers --encoder-attention-heads $num_head \
    --add-3d --no-2d --num-3d-bias-kernel $num_3d_bias_kernel \
    --encoder-embed-dim $hidden_size --encoder-ffn-embed-dim $ffn_size \
    --droppath-prob $droppath_prob --dropout $dropout --attention-dropout $attn_dropout --act-dropout $act_dropout \
    --save-dir $save_path \
    --mode-prob $mode_prob \
    --split test --metric mae