import torch
import numpy as np
from fairseq import checkpoint_utils, utils, options, tasks
from fairseq.logging import progress_bar
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
import ogb
import sys
import os
from pathlib import Path
from sklearn.metrics import roc_auc_score

import sys
from os import path

sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

import logging

def eval(args, use_pretrained, checkpoint_path=None, logger=None):
    np.random.seed(args.seed)
    utils.set_torch_seed(args.seed)

    logger.info(f"loading model and task from {checkpoint_path}...")
    
    # =======================================================
    # 【关键修改】使用 Fairseq 标准加载器
    # 它会自动读取 Checkpoint 里的 Config，而不是用命令行的 args
    # 这样能保证 Task 设置（如 dataset_name, num_atoms）和 Model 架构完全匹配
    # =======================================================
    models, cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        [checkpoint_path],
        arg_overrides={
            "data_path": args.data_path, # 强制覆盖数据路径
            "split": args.split          # 强制覆盖 split
        }
    )
    model = models[0].eval().to(torch.cuda.current_device())
    
    # 2. 加载数据集
    # 注意：这里直接用从 checkpoint 恢复出来的 task 加载数据
    # 它会自动包含正确的 dataset_name (qm9H) 和 atoms 数量
    task.load_dataset(args.split)
    
    # 3. 获取训练集统计量 (用于反归一化)
    train_mean = 0.0
    train_std = 1.0
    if hasattr(task, 'dm') and hasattr(task.dm, 'dataset_train'):
        train_ds = task.dm.dataset_train
        if hasattr(train_ds, 'mean') and hasattr(train_ds, 'std'):
            train_mean = train_ds.mean
            train_std = train_ds.std
            logger.info(f"Loaded Train Stats -> Mean: {train_mean:.4f}, Std: {train_std:.4f}")
    
    # 4. 构建 Batch Iterator
    batch_iterator = task.get_batch_iterator(
        dataset=task.dataset(args.split),
        max_tokens=cfg.dataset.max_tokens_valid,
        max_sentences=cfg.dataset.batch_size_valid,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            model.max_positions(),
        ),
        ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=cfg.dataset.required_batch_size_multiple,
        seed=cfg.common.seed,
        num_workers=cfg.dataset.num_workers,
        epoch=0,
        data_buffer_size=cfg.dataset.data_buffer_size,
        disable_iterator_cache=False,
    )
    itr = batch_iterator.next_epoch_itr(
        shuffle=False, set_dataset_epoch=False
    )
    progress = progress_bar.progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_interval=cfg.common.log_interval,
        default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple")
    )

    # 5. 推理循环
    y_pred = []
    y_true = []
    with torch.no_grad():

        for i, sample in enumerate(progress):
            sample = utils.move_to_cuda(sample)
            
            # # --- 最终 Debug：检查这次数据读对了吗 ---
            # if i == 0:
            #     logger.info("[DEBUG] Checking first batch inputs...")
            #     # 检查原子类型: 应该有 1(H), 6(C), 7(N)... 绝不全是 7
            #     atom_types = sample['net_input']['batched_data']['x'][0, :10, 0]
            #     logger.info(f"Atom Types (First 10): {atom_types.cpu().tolist()}")
            #     logger.info(f"Input Shape: {sample['net_input']['batched_data']['x'].shape}")
            # # ------------------------------------------

            # 获取模型输出
            output = model(**sample["net_input"])
            
            # 这里的 logits 已经是归一化后的预测值
            logits = output[0][:, 0, :].reshape(-1)
            targets = sample["target"].reshape(-1)

            y_pred.extend(logits.cpu())
            y_true.extend(targets.cpu())

    y_pred = torch.tensor(y_pred)
    y_true = torch.tensor(y_true)

    # 6. 计算指标
    # 反归一化
    y_pred_real = y_pred * train_std + train_mean
    y_true_real = y_true * train_std + train_mean

    mae_norm = np.mean(np.abs(y_true.numpy() - y_pred.numpy()))
    mae_real = np.mean(np.abs(y_true_real.numpy() - y_pred_real.numpy()))

    logger.info("=" * 30)
    logger.info(f"Target Task: {cfg.task.dataset_name} (Should be qm9H/qm9)")
    logger.info(f"Normalized MAE: {mae_norm:.6f}")
    logger.info(f"Real MAE (eV/D): {mae_real:.6f}")
    logger.info("=" * 30)

    return mae_real

def main():
    parser = options.get_training_parser()
    parser.add_argument(
        "--split",
        type=str,
    )
    parser.add_argument(
        "--metric",
        type=str,
    )
    # 确保 data_path 存在，否则 load_dataset 会找不到文件
    parser.add_argument("--data-path", type=str, default="./", help="Path to data") 
    
    args = options.parse_args_and_arch(parser, modify_parser=None)
    if not hasattr(args, 'seed'): 
        args.seed = 42
    logger = logging.getLogger(__name__)
    for checkpoint_fname in os.listdir(args.save_dir):
        checkpoint_path = Path(args.save_dir) / checkpoint_fname
        if str(checkpoint_path)[-3:] == '.pt':
            logger.info(f"evaluating checkpoint file {checkpoint_path}")
            result = eval(args, False, str(checkpoint_path), logger)
            open(str(checkpoint_path)[:-3] + f'_{args.split}_{result:.5f}.txt', 'w')


if __name__ == '__main__':
    main()
