import torch
import numpy as np
import os
import logging
from pathlib import Path
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error

# 1. 强力屏蔽 AzureML 告警，防止阻塞输出
os.environ["AZUREML_DEPRECATE_WARNING"] = "False"

from fairseq import checkpoint_utils, utils, options, tasks
from fairseq.logging import progress_bar

# 确保导入 Transformer-M 相关模块
import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def eval_single_checkpoint(args, checkpoint_path):
    logger.info(f"==================================================")
    logger.info(f"开始评估 Checkpoint: {checkpoint_path}")
    
    # 使用 arg_overrides 覆盖路径
    arg_overrides = {
        "data_path": args.data_path,
        "seed": args.seed,
    }

    # 1. 加载模型 (这里如果是大模型会耗时 10-30s)
    logger.info("正在加载模型架构与权重...")
    models, cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        [checkpoint_path],
        arg_overrides=arg_overrides,
    )
    model = models[0].eval().to(torch.cuda.current_device())
    logger.info("模型加载成功！")
    
    # 2. 加载数据集 (HIV/BACE 大数据集这里最容易卡住，因为要读取巨大的 .pt 文件)
    logger.info(f"正在从磁盘加载数据集 [{args.split}] (如果是大任务请耐心等待)...")
    task.load_dataset(args.split)
    dataset = task.dataset(args.split)
    logger.info(f"数据集加载完毕！总样本数: {len(dataset)}")
    
    # 3. 获取回归统计量
    train_mean, train_std = 0.0, 1.0
    try:
        # 尝试从训练集恢复归一化参数
        task.load_dataset('train') # 推理回归任务必须加载训练集统计量
        train_ds = task.dataset('train')
        train_mean = getattr(train_ds, 'y_mean', 0.0)
        train_std = getattr(train_ds, 'y_std', 1.0)
        logger.info(f"已加载训练集统计量 -> Mean: {train_mean:.4f}, Std: {train_std:.4f}")
    except Exception as e:
        logger.warning(f"无法获取训练集统计量，将使用默认值 (0, 1)。错误原因: {e}")

    # 4. 构建迭代器
    batch_iterator = task.get_batch_iterator(
        dataset=dataset,
        max_tokens=cfg.dataset.max_tokens_valid,
        max_sentences=args.batch_size if args.batch_size else cfg.dataset.batch_size_valid,
        num_workers=args.num_workers, # 使用命令行传入的 num_workers
        seed=cfg.common.seed,
    )
    itr = batch_iterator.next_epoch_itr(shuffle=False)

    # 5. 推理循环
    logger.info("开始推理循环...")
    progress = progress_bar.progress_bar(itr, log_format="tqdm")

    y_pred, y_true = [], []
    with torch.no_grad():
        for sample in progress:
            sample = utils.move_to_cuda(sample)
            output = model(**sample["net_input"])
            
            # 获取预测值 (CLS 位置)
            logits = output[0][:, 0, :].reshape(-1)
            targets = sample["target"].reshape(-1)

            y_pred.extend(logits.cpu().tolist())
            y_true.extend(targets.cpu().tolist())

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    # 6. 指标计算
    dataset_name = cfg.task.dataset_name.lower()
    is_reg = any(name in dataset_name for name in ['esol', 'freesolv', 'lipophilicity', 'qm9'])
    
    # 你之前加的 Debug 打印
    logger.info(f"当前使用的评估指标类型: {args.metric}")
    
    if args.metric == 'auc' or not is_reg:
        # 分类任务：ROC-AUC
        scores = 1 / (1 + np.exp(-y_pred)) # Sigmoid
        result = roc_auc_score(y_true, scores)
        logger.info(f">>> [{args.split}] 最终结果 ROC-AUC: {result:.6f}")
    else:
        # 回归任务：RMSE
        # 1. 将模型输出还原回真实物理量纲
        y_real_pred = y_pred * train_std + train_mean
        
        # 2. 计算真实量纲下的 MSE
        mse_real = mean_squared_error(y_true, y_real_pred)
        rmse_real = np.sqrt(mse_real)
        mae_real = mean_absolute_error(y_true, y_real_pred)
        
        result = rmse_real
        logger.info(f">>> [{args.split}] 评估参数 -> Mean: {train_mean:.4f}, Std: {train_std:.4f}")
        logger.info(f">>> [{args.split}] 最终结果 Real RMSE: {rmse_real:.6f}")
        logger.info(f">>> [{args.split}] 最终结果 Real MAE: {mae_real:.6f}")
    
    logger.info(f"==================================================\n")
    return result

def main():
    parser = options.get_training_parser()
    parser.add_argument("--split", type=str, default="test", help="split: valid, test")
    parser.add_argument("--metric", type=str, choices=['auc', 'rmse'], default='auc', help="metric type")
    
    # 允许在命令行指定 num-workers
    args = options.parse_args_and_arch(parser)
    
    if not args.save_dir or not os.path.exists(args.save_dir):
        logger.error(f"找不到指定的 save-dir 目录: {args.save_dir}")
        return

    # 自动识别所有 .pt 文件
    checkpoints = sorted([f for f in os.listdir(args.save_dir) if f.endswith('.pt')])
    logger.info(f"在目录下共找到 {len(checkpoints)} 个权重文件进行评估。")
    
    for ckpt_name in checkpoints:
        ckpt_path = os.path.join(args.save_dir, ckpt_name)
        try:
            res = eval_single_checkpoint(args, ckpt_path)
            # 保存文本结果
            res_log = f"{ckpt_path}_{args.split}_{res:.5f}.txt"
            with open(res_log, 'w') as f:
                f.write(f"Dataset: {args.dataset_name}, Split: {args.split}, Result: {res}\n")
        except Exception as e:
            logger.error(f"评估文件 {ckpt_name} 时发生错误: {e}")
            continue

if __name__ == '__main__':
    main()