from dataclasses import dataclass
import math
from omegaconf import II

import torch
import torch.nn as nn
from fairseq import metrics, modules, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
import os
from dataclasses import dataclass, field

@dataclass
class GraphPredictionConfig(FairseqDataclass):
    tpu: bool = II("common.tpu")
    loss_type: str = field(
        default="L1", 
        metadata={"help": "loss type: L1, L2, or BCE"}
    )


@register_criterion("graph_prediction", dataclass=GraphPredictionConfig)
class GraphPredictionLoss(FairseqCriterion):
    """
    Implementation for the loss used in masked graph model (MGM) training.
    """

    def __init__(self, cfg: GraphPredictionConfig, task):
        super().__init__(task)
        self.tpu = cfg.tpu
        self.noise_scale = task.cfg.noise_scale
        self.loss_type = cfg.loss_type
        

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        sample_size = sample["nsamples"]

        with torch.no_grad():
            natoms = sample["net_input"]["batched_data"]['x'].shape[1]

        # add gaussian noise
        ori_pos = sample['net_input']['batched_data']['pos']
        noise = torch.randn(ori_pos.shape).to(ori_pos) * self.noise_scale
        noise_mask = (ori_pos == 0.0).all(dim=-1, keepdim=True)
        noise = noise.masked_fill_(noise_mask, 0.0)
        # sample['net_input']['batched_data']['pos'] = ori_pos + noise


        model_output = model(**sample["net_input"])
        logits, node_output = model_output[0], model_output[1]
        # logits = logits[:,0,:]
        if logits.dim() == 3:
            logits = logits[:, 0, :]
        elif logits.dim() == 2:
            logits = logits.squeeze(-1)
        
        targets = model.get_targets(sample, [logits])
        targets = targets.squeeze(-1)

        if self.loss_type in ["L1", "L2"]:
            y_mean = sample["net_input"]["batched_data"]["y_mean"].to(logits)
            y_std = sample["net_input"]["batched_data"]["y_std"].to(logits)

            # targets = (targets - y_mean) / (y_std + 1e-8)
        # if self.loss_type in ["L1", "L2"]:
        #     # 从 batched_data 中提取从 Dataset->Collator 传来的均值和标准差
        #     y_mean = sample["net_input"]["batched_data"]["y_mean"].to(logits)
        #     y_std = sample["net_input"]["batched_data"]["y_std"].to(logits)
            
        #     # 将 targets 映射到均值为 0, 标准差为 1 的空间
        #     # 这样模型输出的 logits (通常在 0 附近) 才能有效计算 Loss
        #     shifted_targets = (targets - y_mean) / (y_std + 1e-8)
        # else:
        #     shifted_targets = targets

        if self.loss_type == "BCE":
            # 针对分类任务 (BACE, HIV 等)
            # 使用 BCEWithLogitsLoss，它会自动处理 Sigmoid，数值更稳定
            # 注意：targets 需要转为 float，且形状要对齐
            loss = nn.BCEWithLogitsLoss(reduction='sum')(logits.view(-1), targets.view(-1).float())
        elif self.loss_type == "L2":
            # 针对回归任务的 MSE
            # loss = nn.MSELoss(reduction='sum')(logits.view(-1), targets.view(-1))
            # print("使用 SmoothL1Loss 替代 MSELoss")
            # loss = nn.SmoothL1Loss(reduction='sum')(logits.view(-1), targets.view(-1))
            loss = nn.SmoothL1Loss(beta=0.5, reduction='mean')(
                logits.view(-1), targets.view(-1)
            )

        else:
            # 默认为 L1 (适合大多数回归任务)
            loss = nn.L1Loss(reduction='sum')(logits.view(-1), targets.view(-1))


        if node_output is not None:
            node_mask = (node_output == 0.0).all(dim=-1).all(dim=-1)[:, None, None] + noise_mask
            node_output = node_output.masked_fill_(node_mask, 0.0)

            node_output_loss = (1.0 - nn.CosineSimilarity(dim=-1)(node_output.to(torch.float32), noise.masked_fill_(node_mask, 0.0).to(torch.float32)))
            node_output_loss = node_output_loss.masked_fill_(node_mask.squeeze(-1), 0.0).sum(dim=-1).to(torch.float16)

            tgt_count = (~node_mask).squeeze(-1).sum(dim=-1).to(node_output_loss)
            tgt_count = tgt_count.masked_fill_(tgt_count == 0.0, 1.0)
            node_output_loss = (node_output_loss / tgt_count).sum() * 1
        else:
            node_output_loss = (noise - noise).sum()

        logging_output = {
            "loss": loss.data,
            "node_output_loss": node_output_loss.data,
            "total_loss": loss.data + node_output_loss.data,
            "sample_size": sample_size,
            "nsentences": sample_size,
            "ntokens": natoms,
        }
        return loss + node_output_loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        node_output_loss_sum = sum(log.get("node_output_loss", 0) for log in logging_outputs)
        total_loss_sum = sum(log.get("total_loss", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size, sample_size, round=6
        )
        metrics.log_scalar(
            "node_output_loss", node_output_loss_sum / sample_size, sample_size, round=6
        )
        metrics.log_scalar(
            "total_loss", total_loss_sum / sample_size, sample_size, round=6
        )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True

