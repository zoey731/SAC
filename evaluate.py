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

def eval_single_checkpoint_raw(args, checkpoint_path):
    logger.info(f"==================================================")
    logger.info(f"开始评估 Checkpoint: {checkpoint_path}")
    
    # 使用 arg_overrides 覆盖路径
    arg_overrides = {
        "data_path": args.data_path,
        "seed": 42,
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
        task.load_dataset('train')
        train_ds = task.dataset('train')
        
        # --- 策略 A: 递归穿透包装层 ---
        actual_ds = train_ds
        found = False
        while True:
            # 检查当前层是否有我们需要的值
            if hasattr(actual_ds, 'y_mean') and hasattr(actual_ds, 'y_std'):
                train_mean = actual_ds.y_mean
                train_std = actual_ds.y_std
                found = True
                break
            # 穿透 Fairseq 包装层 (.dataset)
            if hasattr(actual_ds, 'dataset'):
                actual_ds = actual_ds.dataset
            # 穿透 NestedDictionaryDataset (.datasets 字典)
            elif hasattr(actual_ds, 'datasets') and isinstance(actual_ds.datasets, dict):
                # 尝试从 net_input 下的 batched_data 提取
                if 'net_input' in actual_ds.datasets:
                    actual_ds = actual_ds.datasets['net_input']
                    if hasattr(actual_ds, 'datasets') and 'batched_data' in actual_ds.datasets:
                        actual_ds = actual_ds.datasets['batched_data']
                else:
                    break
            else:
                break
        
        # --- 策略 B: 如果 A 失败，直接从底层获取 ---
        if not found:
            # 强制从 task 的数据管理器直接拿，绕过 Dataset 包装
            if hasattr(task, 'dm') and hasattr(task.dm, 'dataset_train'):
                inner_ds = task.dm.dataset_train
                # 再次穿透可能的 InMemoryDataset 包装
                while hasattr(inner_ds, 'dataset'):
                    inner_ds = inner_ds.dataset
                train_mean = getattr(inner_ds, 'y_mean', 0.0)
                train_std = getattr(inner_ds, 'y_std', 1.0)

        logger.info(f"最终获取到的统计量 -> Mean: {train_mean:.4f}, Std: {train_std:.4f}")
        if abs(train_std - 1.0) < 1e-6:
            logger.error("❌ 警告：统计量获取仍然失败（结果为 0/1），请检查 MolNetPosDataset 内部 y_mean 是否赋值成功！")
            
    except Exception as e:
        logger.warning(f"获取统计量过程出错: {e}")

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

# def eval_single_checkpoint(args, checkpoint_path):
#     logger.info(f"==================================================")
#     logger.info(f"开始评估 Checkpoint: {checkpoint_path}")
    
#     # 使用 arg_overrides 覆盖路径
#     arg_overrides = {
#         "data_path": args.data_path,
#         "seed": 42,
#     }

#     # 1. 加载模型
#     logger.info("正在加载模型架构与权重...")
#     models, cfg, task = checkpoint_utils.load_model_ensemble_and_task(
#         [checkpoint_path],
#         arg_overrides=arg_overrides,
#     )
#     model = models[0].eval().to(torch.cuda.current_device())
#     logger.info("模型加载成功！")
    
#     # 2. 加载数据集
#     logger.info(f"正在从磁盘加载数据集 [{args.split}] (如果是大任务请耐心等待)...")
#     task.load_dataset(args.split)
#     dataset = task.dataset(args.split)
#     logger.info(f"数据集加载完毕！总样本数: {len(dataset)}")
    
#     # 3. 获取回归统计量 (因为你的预处理没有归一化 y，所以 0 和 1 是正确的)
#     train_mean, train_std = 0.0, 1.0
#     try:
#         task.load_dataset('train')
#         inner_ds = task.dataset('train')
#         while hasattr(inner_ds, 'dataset'):
#             inner_ds = inner_ds.dataset
#         train_mean = getattr(inner_ds, 'y_mean', 0.0)
#         train_std = getattr(inner_ds, 'y_std', 1.0)
#     except:
#         pass
#     logger.info(f"当前归一化参数 -> Mean: {train_mean:.4f}, Std: {train_std:.4f} (未归一化时应为 0 和 1)")

#     # 4. 构建迭代器
#     batch_iterator = task.get_batch_iterator(
#         dataset=dataset,
#         max_tokens=cfg.dataset.max_tokens_valid,
#         max_sentences=args.batch_size if args.batch_size else cfg.dataset.batch_size_valid,
#         num_workers=args.num_workers,
#         seed=cfg.common.seed,
#     )
#     itr = batch_iterator.next_epoch_itr(shuffle=False)

#     # 5. 推理循环
#     logger.info("开始推理循环...")
#     progress = progress_bar.progress_bar(itr, log_format="tqdm")

#     y_pred, y_true, is_cliff_list = [], [], []
#     with torch.no_grad():
#         for sample in progress:
#             sample = utils.move_to_cuda(sample)
#             output = model(**sample["net_input"])
            
#             # 获取预测值和真实值
#             logits = output[0][:, 0, :].reshape(-1)
#             targets = sample["target"].reshape(-1)

#             y_pred.extend(logits.cpu().tolist())
#             y_true.extend(targets.cpu().tolist())
            
#             # ==========================================
#             # 【终极暴力提取法】：全方位搜索 is_cliff
#             # ==========================================
#             batch_is_cliff = None
            
#             # 尝试路径 1：Transformer-M 标准装载位置
#             if 'batched_data' in sample.get('net_input', {}):
#                 batched_data = sample['net_input']['batched_data']
#                 if 'is_cliff' in batched_data:
#                     batch_is_cliff = batched_data['is_cliff']
            
#             # 尝试路径 2：浅层直装位置
#             if batch_is_cliff is None and 'is_cliff' in sample:
#                 batch_is_cliff = sample['is_cliff']
                
#             # --- 核心：如果真的没找到，立刻抛出详细证据 ---
#             if batch_is_cliff is not None:
#                 # 安全展平并存入列表
#                 is_cliff_list.extend(batch_is_cliff.view(-1).cpu().tolist())
#             else:
#                 # 打印整个批次字典的结构，让你一眼看出标签丢在哪了
#                 logger.error(f"严重警告：没找到 is_cliff！当前 batched_data 包含的键值有：{list(sample.get('net_input', {}).get('batched_data', {}).keys())}")

#     y_pred = np.array(y_pred)
#     y_true = np.array(y_true)

#     # 6. 指标计算
#     dataset_name = cfg.task.dataset_name.lower()
    
#     # 【修复点】：把 moleculeace 加入回归任务判定名单！
#     is_reg = any(name in dataset_name for name in ['esol', 'freesolv', 'lipophilicity', 'qm9', 'moleculeace'])
    
#     logger.info(f"当前任务被判定为: {'回归 (Regression)' if is_reg else '分类 (Classification)'}")
    
#     if args.metric == 'auc' and not is_reg:
#         scores = 1 / (1 + np.exp(-y_pred)) # Sigmoid
#         result = roc_auc_score(y_true, scores)
#         logger.info(f">>> [{args.split}] 最终结果 ROC-AUC: {result:.6f}")
#     else:
#         # 回归任务：RMSE
#         y_real_pred = y_pred * train_std + train_mean
#         y_real_true = y_true * train_std + train_mean  # 【新增这一行】
#         mse_real = mean_squared_error(y_true, y_real_pred)
#         rmse_real = np.sqrt(mse_real)
#         mae_real = mean_absolute_error(y_true, y_real_pred)
        
#         logger.info(f">>> [{args.split}] 最终结果 全量 RMSE (Total): {rmse_real:.6f}")
#         logger.info(f">>> [{args.split}] 最终结果 全量 MAE (Total): {mae_real:.6f}")
        
#         # ==========================================
#         # 【论文亮点】：计算 Cliff-RMSE
#         # ==========================================
#         print("++++++++++++++++++++=")
#         print(f"Debug: is_cliff_list 长度 = {len(is_cliff_list)}, y_true 长度 = {len(y_true)}")
#         if len(is_cliff_list) == len(y_true):
#             is_cliff_arr = np.array(is_cliff_list)
#             cliff_idx = np.where(is_cliff_arr == 1)[0]
#             non_cliff_idx = np.where(is_cliff_arr == 0)[0]
            
#             if len(cliff_idx) > 0:
#                 cliff_rmse = np.sqrt(mean_squared_error(y_true[cliff_idx], y_real_pred[cliff_idx]))
#                 normal_rmse = np.sqrt(mean_squared_error(y_true[non_cliff_idx], y_real_pred[non_cliff_idx]))
                
#                 logger.info(f">>> [{args.split}] ---------------------------------")
#                 logger.info(f">>> [{args.split}] 悬崖分子数量 (is_cliff=1): {len(cliff_idx)}")
#                 logger.info(f">>> [{args.split}] 悬崖专项误差 (Cliff RMSE): {cliff_rmse:.6f}  <--- 论文核心指标！")
#                 logger.info(f">>> [{args.split}] 普通分子误差 (Normal RMSE): {normal_rmse:.6f}")
#                 logger.info(f">>> [{args.split}] ---------------------------------")
#         else:
#             logger.warning("未能成功对齐 is_cliff 标签，跳过 Cliff-RMSE 计算。")
            
#         result = rmse_real
    
#     logger.info(f"==================================================\n")
#     return result

from fairseq import tasks

def eval_single_checkpoint_right(args, checkpoint_path):
    logger.info(f"==================================================")
    logger.info(f"开始评估 Checkpoint: {checkpoint_path}")
    
    arg_overrides = {
        "data_path": args.data_path,
        "seed": 42,
    }

    # 1. 正常加载模型架构与权重（最纯净的加载，不搞黑客操作）
    logger.info("正在加载模型架构与权重...")
    models, cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        [checkpoint_path],
        arg_overrides=arg_overrides,
    )
    model = models[0].eval().to(torch.cuda.current_device())
    logger.info("模型加载成功！")
    
    # 2. 加载数据集
    logger.info(f"正在从磁盘加载数据集 [{args.split}] (如果是大任务请耐心等待)...")
    task.load_dataset(args.split)
    dataset = task.dataset(args.split)
    logger.info(f"数据集加载完毕！总样本数: {len(dataset)}")
    
    # 3. 获取回归统计量 (直接从物理磁盘读取)
    train_mean, train_std = 0.0, 1.0
    try:
        import os
        prop_name = cfg.task.dataset_name.split('-')[-1] 
        split_path = os.path.join(cfg.task.data_path, prop_name, "split_82_cluster.pt")
        
        if os.path.exists(split_path):
            split_dict = torch.load(split_path)
            train_mean = split_dict.get('y_mean', 0.0)
            train_std = split_dict.get('y_std', 1.0)
            logger.info(f"✅ 成功从磁盘直接读取归一化参数: {split_path}")
        else:
            logger.warning(f"❌ 找不到缓存文件: {split_path}")
    except Exception as e:
        logger.error(f"读取归一化参数时发生异常: {e}")

    logger.info(f"当前使用的目标值还原参数 -> Mean: {train_mean:.4f}, Std: {train_std:.4f}")

    # 4. 构建迭代器
    batch_iterator = task.get_batch_iterator(
        dataset=dataset,
        max_tokens=cfg.dataset.max_tokens_valid,
        max_sentences=args.batch_size if args.batch_size else cfg.dataset.batch_size_valid,
        num_workers=args.num_workers,
        seed=cfg.common.seed,
    )
    itr = batch_iterator.next_epoch_itr(shuffle=False)

    # 5. 推理循环
    logger.info("开始推理循环...")
    progress = progress_bar.progress_bar(itr, log_format="tqdm")

    y_pred, y_true, is_cliff_list = [], [], []
    with torch.no_grad():
        for sample in progress:
            sample = utils.move_to_cuda(sample)
            
            # 模型前向传播
            output = model(**sample["net_input"])
            
            # ==========================================
            # 【终极核心自适应解析】：无论模型返回什么，强行提取正确预测！
            # ==========================================
            batch_size = sample["target"].size(0)
            logits = None
            raw_out = output[0] if isinstance(output, tuple) else output
            
            if raw_out.dim() <= 2 and raw_out.size(0) == batch_size:
                # 理想情况: [batch, 1]
                logits = raw_out
            elif raw_out.dim() == 3 and raw_out.size(1) == 1:
                # 嵌套情况: [batch, 1, 1]
                logits = raw_out[:, 0, :]
            elif raw_out.dim() == 3 and raw_out.size(2) == 1:
                # 序列情况: 取 CLS 节点的标量 [batch, seq_len, 1] -> [batch, 1]
                logits = raw_out[:, 0, :]
            elif raw_out.dim() == 3 and raw_out.size(2) > 1:
                # 隐层特征情况: [batch, seq_len, hidden_dim] -> 手动送入回归头！！！
                cls_features = raw_out[:, 0, :]
                if hasattr(model, 'encoder') and hasattr(model.encoder, 'proj_out'):
                    logits = model.encoder.proj_out(cls_features)
                elif hasattr(model, 'classification_heads'):
                    head_name = next(iter(model.classification_heads.keys()))
                    logits = model.classification_heads[head_name](cls_features)
                else:
                    raise ValueError("找不到回归头，无法将特征转换为预测值！")
            
            if logits is None:
                raise ValueError(f"无法解析模型输出形状: {raw_out.shape}")
                
            logits = logits.reshape(-1)
            # ==========================================
            
            targets = sample["target"].reshape(-1)

            y_pred.extend(logits.cpu().tolist())
            y_true.extend(targets.cpu().tolist())
            
            # 全方位搜索 is_cliff
            batch_is_cliff = None
            if 'batched_data' in sample.get('net_input', {}):
                batched_data = sample['net_input']['batched_data']
                if 'is_cliff' in batched_data:
                    batch_is_cliff = batched_data['is_cliff']
            if batch_is_cliff is None and 'is_cliff' in sample:
                batch_is_cliff = sample['is_cliff']
                
            if batch_is_cliff is not None:
                is_cliff_list.extend(batch_is_cliff.view(-1).cpu().tolist())

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    # 6. 指标计算
    dataset_name = cfg.task.dataset_name.lower()
    is_reg = any(name in dataset_name for name in ['esol', 'freesolv', 'lipophilicity', 'qm9', 'moleculeace'])
    logger.info(f"当前任务被判定为: {'回归 (Regression)' if is_reg else '分类 (Classification)'}")
    
    if args.metric == 'auc' and not is_reg:
        scores = 1 / (1 + np.exp(-y_pred)) # Sigmoid
        result = roc_auc_score(y_true, scores)
        logger.info(f">>> [{args.split}] 最终结果 ROC-AUC: {result:.6f}")
    else:
        # 预测值和真实值同时还原回原始物理量纲！
        y_real_pred = y_pred * train_std + train_mean
        y_real_true = y_true * train_std + train_mean
        print("y_real_true ", y_real_true[:5])
        print("y_true ", y_true[:5])
        print("y_real_pred ", y_real_pred[:5])
        print("y_pred ", y_pred[:5])

        
        mse_real = mean_squared_error(y_real_true, y_real_pred)
        rmse_real = np.sqrt(mse_real)
        mae_real = mean_absolute_error(y_real_true, y_real_pred)
        
        logger.info(f">>> [{args.split}] 最终结果 全量 RMSE (Total): {rmse_real:.6f}")
        logger.info(f">>> [{args.split}] 最终结果 全量 MAE (Total): {mae_real:.6f}")
        
        # 计算 Cliff-RMSE
        if len(is_cliff_list) == len(y_real_true):
            is_cliff_arr = np.array(is_cliff_list)
            cliff_idx = np.where(is_cliff_arr == 1)[0]
            non_cliff_idx = np.where(is_cliff_arr == 0)[0]
            
            if len(cliff_idx) > 0:
                cliff_rmse = np.sqrt(mean_squared_error(y_real_true[cliff_idx], y_real_pred[cliff_idx]))
                normal_rmse = np.sqrt(mean_squared_error(y_real_true[non_cliff_idx], y_real_pred[non_cliff_idx]))
                
                logger.info(f">>> [{args.split}] ---------------------------------")
                logger.info(f">>> [{args.split}] 悬崖分子数量 (is_cliff=1): {len(cliff_idx)}")
                logger.info(f">>> [{args.split}] 悬崖专项误差 (Cliff RMSE): {cliff_rmse:.6f}  <--- 论文核心指标！")
                logger.info(f">>> [{args.split}] 普通分子误差 (Normal RMSE): {normal_rmse:.6f}")
                logger.info(f">>> [{args.split}] ---------------------------------")
            
        result = rmse_real
    
    logger.info(f"==================================================\n")
    return result


def eval_single_checkpoint(args, checkpoint_path):
    logger.info(f"==================================================")
    logger.info(f"开始评估 Checkpoint: {checkpoint_path}")
    
    arg_overrides = {"data_path": args.data_path, "seed": 42}

    # 1. 加载模型 & 任务
    models, cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        [checkpoint_path],
        arg_overrides=arg_overrides,
    )
    model = models[0].eval().cuda()

    # 2. 加载数据集
    logger.info(f"正在从磁盘加载数据集 [{args.split}] ...")
    task.load_dataset(args.split)
    dataset = task.dataset(args.split)

    # 3. 读取 Min-Max 归一化参数
    import os
    train_mean, train_std = 0.0, 1.0
    dataset_name = cfg.task.dataset_name if hasattr(cfg, 'task') else cfg['task']['dataset_name']
    prop_name = dataset_name.split('-')[-1]
    split_path = os.path.join(args.data_path, prop_name, "split_82_cluster.pt")

    if os.path.exists(split_path):
        split_dict = torch.load(split_path)
        train_mean = split_dict.get('y_mean', 0.0)   
        train_std = split_dict.get('y_std', 1.0)     

    logger.info(f"[Eval] 使用归一化参数: mean={train_mean:.4f}, std={train_std:.4f}")

    # 4. dataloader
    if hasattr(cfg, 'dataset'):
        max_tokens = cfg.dataset.max_tokens_valid
        batch_size = args.batch_size if args.batch_size else cfg.dataset.batch_size_valid
    else:
        max_tokens = cfg['dataset']['max_tokens_valid']
        batch_size = args.batch_size if args.batch_size else cfg['dataset']['batch_size_valid']

    batch_iterator = task.get_batch_iterator(
        dataset=dataset,
        max_tokens=max_tokens,
        max_sentences=batch_size,
        num_workers=args.num_workers,
        seed=42,
    )
    itr = batch_iterator.next_epoch_itr(shuffle=False)

    # 5. 推理
    y_pred, y_true, is_cliff_list = [], [], []

    with torch.no_grad():
        for sample in progress_bar.progress_bar(itr, log_format="tqdm"):
            sample = utils.move_to_cuda(sample)

            # ===== 正确 forward（严格对齐 criterion）=====
            model_output = model(**sample["net_input"])
            logits = model_output[0]   # (B, T, C)

            # 只取 graph token（和训练一致）
            if logits.dim() == 3:
                logits = logits[:, 0, :]

            pred = logits.view(-1)
            target = sample["target"].view(-1)

            y_pred.extend(pred.cpu().numpy())
            y_true.extend(target.cpu().numpy())

            # is_cliff
            if 'batched_data' in sample.get('net_input', {}) and 'is_cliff' in sample['net_input']['batched_data']:
                is_cliff = sample['net_input']['batched_data']['is_cliff']
                is_cliff_list.extend(is_cliff.view(-1).cpu().numpy())

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    # ===== Sanity Check（非常重要）=====
    print("pred std:", np.std(y_pred))
    print("pred mean:", np.mean(y_pred))

    # 6. 反归一化（Min-Max）
    y_real_pred = y_pred * train_std + train_mean
    y_real_true = y_true * train_std + train_mean

    print("\n真实值:", y_real_true[:50])
    print("预测值:", y_real_pred[:50])

    # 7. 指标
    mse = mean_squared_error(y_real_true, y_real_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_real_true, y_real_pred)

    logger.info(f">>> [{args.split}] RMSE (Total): {rmse:.6f}")
    logger.info(f">>> [{args.split}] MAE  (Total): {mae:.6f}")

    # 8. Cliff 分析
    if len(is_cliff_list) == len(y_real_true):
        is_cliff_arr = np.array(is_cliff_list)
        cliff_idx = np.where(is_cliff_arr == 1)[0]
        normal_idx = np.where(is_cliff_arr == 0)[0]

        if len(cliff_idx) > 0:
            cliff_rmse = np.sqrt(mean_squared_error(y_real_true[cliff_idx], y_real_pred[cliff_idx]))
            normal_rmse = np.sqrt(mean_squared_error(y_real_true[normal_idx], y_real_pred[normal_idx]))

            logger.info(f">>> [{args.split}] ---------------------------------")
            logger.info(f">>> [{args.split}] Cliff RMSE: {cliff_rmse:.6f}")
            logger.info(f">>> [{args.split}] Normal RMSE: {normal_rmse:.6f}")
            logger.info(f">>> [{args.split}] ---------------------------------")

    return rmse

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