''' Translate input text with trained model. '''

import torch
import argparse
import dill as pickle
from tqdm import tqdm

import transformer.Constants as Constants
from transformer.Models import Transformer
from transformer.Translator import Translator

__author__ = "Yu-Hsiang Huang"


def load_model(opt, device, src_pad_idx, trg_pad_idx):
    """
    加载已训练的模型权重。
    显式指定 map_location='cpu' 以兼容在没有 GPU 的机器上进行反序列化，
    并通过参数显式传递词表动态算出的 Padding 索引。
    """
    # 强制映射到 CPU
    checkpoint = torch.load(opt.model, map_location='cpu')
    model_opt = checkpoint['settings']

    # 兼容处理
    d_word_vec = getattr(model_opt, 'd_word_vec', model_opt.d_model)

    model = Transformer(
        model_opt.src_vocab_size,
        model_opt.trg_vocab_size,

        src_pad_idx,
        trg_pad_idx,

        trg_emb_prj_weight_sharing=model_opt.proj_share_weight,
        emb_src_trg_weight_sharing=model_opt.embs_share_weight,
        d_k=model_opt.d_k,
        d_v=model_opt.d_v,
        d_model=model_opt.d_model,
        d_word_vec=d_word_vec,
        d_inner=model_opt.d_inner_hid,
        n_layers=model_opt.n_layers,
        n_head=model_opt.n_head,
        dropout=model_opt.dropout).to(device)

    model.load_state_dict(checkpoint['model'])
    print('[Info] Trained model state loaded.')
    return model 


def main():
    '''Main Function'''

    parser = argparse.ArgumentParser(description='translate.py')

    parser.add_argument('-model', required=True,
                        help='Path to model weight file')
    parser.add_argument('-data_pkl', required=True,
                        help='Pickle file with both instances and vocabulary.')
    parser.add_argument('-output', default='prediction.txt',
                        help="""Path to output the predictions (each line will
                        be the decoded sequence)""")
    parser.add_argument('-beam_size', type=int, default=5)
    parser.add_argument('-max_seq_len', type=int, default=100)
    parser.add_argument('-no_cuda', action='store_true')

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda

    # 载入预处理好的词表与测试集
    data = pickle.load(open(opt.data_pkl, 'rb'))
    SRC = data['vocab']['src']
    TRG = data['vocab']['trg']
    
    # 提取词表映射字典
    src_stoi = SRC.vocab.stoi
    trg_itos = TRG.vocab.itos

    opt.src_pad_idx = src_stoi[Constants.PAD_WORD]
    opt.trg_pad_idx = TRG.vocab.stoi[Constants.PAD_WORD]
    opt.trg_bos_idx = TRG.vocab.stoi[Constants.BOS_WORD]
    opt.trg_eos_idx = TRG.vocab.stoi[Constants.EOS_WORD]

    # 摆脱旧版 torchtext 依赖
    test_data = data['test']
    
    device = torch.device('cuda' if opt.cuda else 'cpu')
    
    # 将实时计算出的 Padding 索引正确传递给模型初始化函数
    translator = Translator(
        model=load_model(opt, device, opt.src_pad_idx, opt.trg_pad_idx),
        beam_size=opt.beam_size,
        max_seq_len=opt.max_seq_len,
        src_pad_idx=opt.src_pad_idx,
        trg_pad_idx=opt.trg_pad_idx,
        trg_bos_idx=opt.trg_bos_idx,
        trg_eos_idx=opt.trg_eos_idx).to(device)

    unk_idx = src_stoi[Constants.UNK_WORD]
    
    print('[Info] Translating test dataset...')
    with open(opt.output, 'w', encoding='utf-8') as f:
        for example in tqdm(test_data, mininterval=2, desc='  - (Test)', leave=False):
            # 将源语言文本（德语词列表）转换为对应的数字 ID 序列
            src_seq = [src_stoi.get(word, unk_idx) for word in example.src]
            
            # 使用 Beam Search 进行自回归解码
            pred_seq = translator.translate_sentence(torch.LongTensor([src_seq]).to(device))
            
            # 将生成的 ID 序列还原为目标语言（英语）文本
            pred_line = ' '.join(trg_itos[idx] for idx in pred_seq)
            pred_line = pred_line.replace(Constants.BOS_WORD, '').replace(Constants.EOS_WORD, '')
            
            f.write(pred_line.strip() + '\n')

    print(f'[Info] Finished. Predictions saved in {opt.output}')

if __name__ == "__main__":
    main()