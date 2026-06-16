ulimit -c unlimited

[ -z "${layers}" ] && layers=12
[ -z "${hidden_size}" ] && hidden_size=768
[ -z "${ffn_size}" ] && ffn_size=768
[ -z "${num_head}" ] && num_head=32
[ -z "${batch_size}" ] && batch_size=8
[ -z "${update_freq}" ] && update_freq=4
# [ -z "${update_freq}" ] && update_freq=2
[ -z "${seed}" ] && seed=42
[ -z "${clip_norm}" ] && clip_norm=1
[ -z "${data_path}" ] && data_path='./datasets/moleculenet/bacce'
[ -z "${save_path}" ] && save_path='/home/zhaoqc/Transformer_bias_add_moleculenet/logs/exp-dataset-molnet-bace-lr-2e-5-end_lr-1e-6-tsteps-20000-wsteps-2000-L12-D768-F768-H32-SLN-false-BS32-CLIP5-dp0.3-attn_dp0.1-wd0.01-dpp0.0/SEED42-TASKbace-LOSS-BCE-STD-std_logits-RF-cls'
[ -z "${dataset_name}" ] && dataset_name="molnet-bace"
# [ -z "${dropout}" ] && dropout=0.0
[ -z "${dropout}" ] && dropout=0.3
[ -z "${act_dropout}" ] && act_dropout=0.1
[ -z "${attn_dropout}" ] && attn_dropout=0.1
# [ -z "${weight_decay}" ] && weight_decay=0.0
[ -z "${weight_decay}" ] && weight_decay=1e-2
[ -z "${sandwich_ln}" ] && sandwich_ln="false"
[ -z "${droppath_prob}" ] && droppath_prob=0.1
[ -z "${noise_scale}" ] && noise_scale=0.2
[ -z "${mode_prob}" ] && mode_prob="0.2,0.2,0.6"
[ -z "${task_idx}" ] && task_idx=bace # 必须和你训练时一样
[ -z "${add_3d}" ] && add_3d="true"
[ -z "${no_2d}" ] && no_2d="false"
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
    --task graph_prediction \
    --criterion graph_prediction \
    --arch transformer_m \
    --num-classes 1 \
    --encoder-layers $layers --encoder-attention-heads $num_head \
    --add-3d --no-2d --num-3d-bias-kernel $num_3d_bias_kernel \
    --encoder-embed-dim $hidden_size --encoder-ffn-embed-dim $ffn_size \
    --droppath-prob $droppath_prob --dropout $dropout --attention-dropout $attn_dropout --act-dropout $act_dropout \
    --save-dir $save_path \
    --mode-prob $mode_prob \
    --loss-type BCE \
    --split test --metric auc

    # --loss-type L1 --std-type no_std --readout-type cls \