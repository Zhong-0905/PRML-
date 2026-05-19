''' Training script for Transformer '''
import os
import argparse
import math
import time
import numpy as np
import dill as pickle
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import transformer.Constants as Constants
from transformer.Models import Transformer
from transformer.Optim import ScheduledOptim

__author__ = "Yu-Hsiang Huang"


class CustomDataset(Dataset):
    """读取预处理生成的 CustomExample 对象列表"""
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx].src, self.examples[idx].trg


def paired_collate_fn(insts, src_stoi, trg_stoi):
    """将一整个 Batch 的文本序列动态进行 Padding 对齐，完美替代旧版 BucketIterator"""
    src_insts, trg_insts = zip(*insts)
    
    # 动态获取本地 Constants 里的真实常量
    pad_idx = getattr(Constants, 'PAD', getattr(Constants, 'PAD_IDX', 0))
    bos_idx = getattr(Constants, 'BOS', getattr(Constants, 'BOS_IDX', 2))
    eos_idx = getattr(Constants, 'EOS', getattr(Constants, 'EOS_IDX', 3))
    unk_idx = getattr(Constants, 'UNK', getattr(Constants, 'UNK_IDX', 1))

    # 将词序列转换为数字 ID 序列，并自动加上 BOS 和 EOS
    def encode_and_pad(insts, stoi):
        max_len = max(len(inst) for inst in insts) + 2
        batch_seq = np.full((len(insts), max_len), pad_idx, dtype=np.int64)
        for i, inst in enumerate(insts):
            seq = [stoi.get(w, unk_idx) for w in inst]
            seq = [bos_idx] + seq + [eos_idx]
            batch_seq[i, :len(seq)] = seq
        return torch.FloatTensor(batch_seq).long()

    src_seq = encode_and_pad(src_insts, src_stoi)
    trg_seq = encode_and_pad(trg_insts, trg_stoi)

    # 构造训练所需的金色标准标签（去掉首位的 BOS）
    gold = trg_seq[:, 1:].contiguous()
    return src_seq, trg_seq, gold


# ================== 核心训练/评估函数 ==================
def cal_performance(pred, gold, trg_pad_idx, smoothing=False):
    loss = cal_loss(pred, gold, trg_pad_idx, smoothing=smoothing)
    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(trg_pad_idx)
    n_correct = pred.eq(gold).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()
    return loss, n_correct, n_word


def cal_loss(pred, gold, trg_pad_idx, smoothing=False):
    gold = gold.contiguous().view(-1)
    if smoothing:
        eps = 0.1
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)
        non_pad_mask = gold.ne(trg_pad_idx)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=trg_pad_idx, reduction='sum')
    return loss


def train_epoch(model, training_data, optimizer, opt, device, pad_idx):
    model.train()
    total_loss, n_word_total, n_word_correct = 0, 0, 0

    desc = '  - (Training)    '
    for batch in tqdm(training_data, mininterval=2, desc=desc, leave=False):
        src_seq, trg_seq, gold = map(lambda x: x.to(device), batch)
        optimizer.zero_grad()
        
        # Transformer 预测时需要切掉目标的最后一个 Token 作为输入
        pred = model(src_seq, trg_seq[:, :-1])
        
        loss, n_correct, n_word = cal_performance(
            pred.view(-1, pred.size(-1)), gold, pad_idx, smoothing=opt.label_smoothing
        )
        loss.backward()
        optimizer.step_and_update_lr()

        total_loss += loss.item()
        n_word_correct += n_correct
        n_word_total += n_word

    loss_per_word = total_loss / n_word_total
    accuracy = n_word_correct / n_word_total
    return loss_per_word, accuracy


def eval_epoch(model, validation_data, device, pad_idx):
    model.eval()
    total_loss, n_word_total, n_word_correct = 0, 0, 0

    desc = '  - (Validation) '
    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2, desc=desc, leave=False):
            src_seq, trg_seq, gold = map(lambda x: x.to(device), batch)
            pred = model(src_seq, trg_seq[:, :-1])
            loss, n_correct, n_word = cal_performance(
                pred.view(-1, pred.size(-1)), gold, pad_idx, smoothing=False
            )

            total_loss += loss.item()
            n_word_correct += n_correct
            n_word_total += n_word

    loss_per_word = total_loss / n_word_total
    accuracy = n_word_correct / n_word_total
    return loss_per_word, accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_pkl', default=None, required=True)
    parser.add_argument('-epoch', type=int, default=10)
    parser.add_argument('-b', type=int, default=64)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-d_inner_hid', type=int, default=2048)
    parser.add_argument('-d_k', type=int, default=64)
    parser.add_argument('-d_v', type=int, default=64)
    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-warmup', type=int, default=4000)
    parser.add_argument('-lr_mul', type=float, default=2.0)
    parser.add_argument('-seed', type=int, default=None)
    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-output_dir', type=str, default='output')
    parser.add_argument('-use_tb', action='store_true')
    parser.add_argument('-label_smoothing', action='store_true')
    parser.add_argument('-embs_share_weight', action='store_true')
    parser.add_argument('-proj_share_weight', action='store_true')
    parser.add_argument('-scale_emb_or_prj', type=str, default='prj')

    opt = parser.parse_args()
    print(opt)

    if opt.seed is not None:
        torch.manual_seed(opt.seed)
        np.random.seed(opt.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Info] Using device: {device}")

    # ================== 载入预处理好的 Pickle 数据 ==================
    print(f"[Info] Loading data from {opt.data_pkl}")
    with open(opt.data_pkl, 'rb') as f:
        data = pickle.load(f)
        data['train'] = data['train'][:1000]
        data['valid'] = data['valid'][:200]

    src_stoi = data['vocab']['src'].vocab.stoi
    trg_stoi = data['vocab']['trg'].vocab.stoi
    opt.src_vocab_size = len(src_stoi)
    opt.trg_vocab_size = len(trg_stoi)

    # 包装并构建 PyTorch 规范的 Dataloader
    train_loader = DataLoader(
        CustomDataset(data['train']), batch_size=opt.b, shuffle=True,
        collate_fn=lambda x: paired_collate_fn(x, src_stoi, trg_stoi)
    )
    val_loader = DataLoader(
        CustomDataset(data['valid']), batch_size=opt.b, shuffle=False,
        collate_fn=lambda x: paired_collate_fn(x, src_stoi, trg_stoi)
    )

    # 自动提取你本地 Constants 里的真实 Padding 索引
    pad_idx = getattr(Constants, 'PAD', getattr(Constants, 'PAD_IDX', 0))

    # ================== 初始化模型 ==================
    transformer = Transformer(
        opt.src_vocab_size, opt.trg_vocab_size, src_pad_idx=pad_idx, trg_pad_idx=pad_idx,
        d_model=opt.d_model, d_inner=opt.d_inner_hid, n_layers=opt.n_layers, n_head=opt.n_head,
        d_k=opt.d_k, d_v=opt.d_v, dropout=opt.dropout, n_position=200,
        trg_emb_prj_weight_sharing=opt.proj_share_weight, emb_src_trg_weight_sharing=opt.embs_share_weight,
        scale_emb_or_prj=opt.scale_emb_or_prj
    ).to(device)

    optimizer = ScheduledOptim(
        optim.Adam(transformer.parameters(), betas=(0.9, 0.98), eps=1e-09),
        opt.lr_mul, opt.d_model, opt.warmup
    )

    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)

    # ================== 迭代训练流程 ==================
    log_train_file = os.path.join(opt.output_dir, 'train.log')
    log_valid_file = os.path.join(opt.output_dir, 'valid.log')

    print(f'[Info] Training Start. Checkpoints will be saved in {opt.output_dir}')
    with open(log_train_file, 'w') as f_train, open(log_valid_file, 'w') as f_valid:
        f_train.write('epoch,loss,ppl,accuracy\n')
        f_valid.write('epoch,loss,ppl,accuracy\n')

    best_acu = 0.0
    for epoch_i in range(opt.epoch):
        print(f'[ Epoch {epoch_i} ]')

        start = time.time()
        train_loss, train_acc = train_epoch(transformer, train_loader, optimizer, opt, device, pad_idx)
        print(f'  - (Training)   ppl: {math.exp(min(train_loss, 100)):8.5f}, accuracy: {train_acc*100:3.3f} %, elapse: {(time.time()-start)/60:3.3f} min')

        start = time.time()
        valid_loss, valid_acc = eval_epoch(transformer, val_loader, device, pad_idx)
        print(f'  - (Validation) ppl: {math.exp(min(valid_loss, 100)):8.5f}, accuracy: {valid_acc*100:3.3f} %, elapse: {(time.time()-start)/60:3.3f} min')

        with open(log_train_file, 'a') as f_train, open(log_valid_file, 'a') as f_valid:
            f_train.write(f'{epoch_i},{train_loss},{math.exp(min(train_loss,100))},{train_acc}\n')
            f_valid.write(f'{epoch_i},{valid_loss},{math.exp(min(valid_loss,100))},{valid_acc}\n')

        model_state_dict = transformer.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'settings': opt,
            'epoch': epoch_i
        }

        if valid_acc > best_acu:
            best_acu = valid_acc
            model_name = os.path.join(opt.output_dir, 'model_best.chkpt')
            torch.save(checkpoint, model_name)
            print(f'  - [Info] The checkpoint file has been updated with best accuracy.')


if __name__ == '__main__':
    main()