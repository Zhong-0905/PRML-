''' Handling the data io '''
import os
import argparse
import logging
import dill as pickle
import urllib
from tqdm import tqdm
import sys
import codecs
import spacy
import torch
import tarfile
import torchtext

import transformer.Constants as Constants
from learn_bpe import learn_bpe
from apply_bpe import BPE

__author__ = "Yu-Hsiang Huang"

_TRAIN_DATA_SOURCES = [
    {"url": "http://data.statmt.org/wmt17/translation-task/" \
             "training-parallel-nc-v12.tgz",
     "trg": "news-commentary-v12.de-en.en",
     "src": "news-commentary-v12.de-en.de"},
    ]

_VAL_DATA_SOURCES = [
    {"url": "http://data.statmt.org/wmt17/translation-task/dev.tgz",
     "trg": "newstest2013.en",
     "src": "newstest2013.de"}]

_TEST_DATA_SOURCES = [
    {"url": "https://storage.googleapis.com/tf-perf-public/" \
                "official_transformer/test_data/newstest2014.tgz",
     "trg": "newstest2014.en",
     "src": "newstest2014.de"}]


class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def file_exist(dir_name, file_name):
    for sub_dir, _, files in os.walk(dir_name):
        if file_name in files:
            return os.path.join(sub_dir, file_name)
    return None


def download_and_extract(download_dir, url, src_filename, trg_filename):
    src_path = file_exist(download_dir, src_filename)
    trg_path = file_exist(download_dir, trg_filename)

    if src_path and trg_path:
        sys.stderr.write(f"Already downloaded and extracted {url}.\n")
        return src_path, trg_path

    compressed_file = _download_file(download_dir, url)

    sys.stderr.write(f"Extracting {compressed_file}.\n")
    with tarfile.open(compressed_file, "r:gz") as corpus_tar:
        corpus_tar.extractall(download_dir)

    src_path = file_exist(download_dir, src_filename)
    trg_path = file_exist(download_dir, trg_filename)

    if src_path and trg_path:
        return src_path, trg_path

    raise OSError(f"Download/extraction failed for url {url} to path {download_dir}")


def _download_file(download_dir, url):
    filename = url.split("/")[-1]
    if file_exist(download_dir, filename):
        sys.stderr.write(f"Already downloaded: {url} (at {filename}).\n")
    else:
        sys.stderr.write(f"Downloading from {url} to {filename}.\n")
        with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
            urllib.request.urlretrieve(url, filename=filename, reporthook=t.update_to)
    return filename


def get_raw_files(raw_dir, sources):
    raw_files = { "src": [], "trg": [], }
    for d in sources:
        src_file, trg_file = download_and_extract(raw_dir, d["url"], d["src"], d["trg"])
        raw_files["src"].append(src_file)
        raw_files["trg"].append(trg_file)
    return raw_files


def mkdir_if_needed(dir_name):
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)


def compile_files(raw_dir, raw_files, prefix):
    src_fpath = os.path.join(raw_dir, f"raw-{prefix}.src")
    trg_fpath = os.path.join(raw_dir, f"raw-{prefix}.trg")

    if os.path.isfile(src_fpath) and os.path.isfile(trg_fpath):
        sys.stderr.write(f"Merged files found, skip the merging process.\n")
        return src_fpath, trg_fpath

    sys.stderr.write(f"Merge files into two files: {src_fpath} and {trg_fpath}.\n")

    with open(src_fpath, 'w', encoding='utf-8') as src_outf, open(trg_fpath, 'w', encoding='utf-8') as trg_outf:
        for src_inf, trg_inf in zip(raw_files['src'], raw_files['trg']):
            sys.stderr.write(f'  Input files: \n'\
                    f'    - SRC: {src_inf}, and\n' \
                    f'    - TRG: {trg_inf}.\n')
            with open(src_inf, newline='\n', encoding='utf-8') as src_inf, open(trg_inf, newline='\n', encoding='utf-8') as trg_inf:
                cntr = 0
                for i, line in enumerate(src_inf):
                    cntr += 1
                    src_outf.write(line.replace('\r', ' ').strip() + '\n')
                for j, line in enumerate(trg_inf):
                    cntr -= 1
                    trg_outf.write(line.replace('\r', ' ').strip() + '\n')
                assert cntr == 0, 'Number of lines in two files are inconsistent.'
    return src_fpath, trg_fpath


def encode_file(bpe, in_file, out_file):
    sys.stderr.write(f"Read raw content from {in_file} and \n"\
            f"Write encoded content to {out_file}\n")
    
    with codecs.open(in_file, encoding='utf-8') as in_f:
        with codecs.open(out_file, 'w', encoding='utf-8') as out_f:
            for line in in_f:
                out_f.write(bpe.process_line(line))


def encode_files(bpe, src_in_file, trg_in_file, data_dir, prefix):
    src_out_file = os.path.join(data_dir, f"{prefix}.src")
    trg_out_file = os.path.join(data_dir, f"{prefix}.trg")

    if os.path.isfile(src_out_file) and os.path.isfile(trg_out_file):
        sys.stderr.write(f"Encoded files found, skip the encoding process ...\n")

    encode_file(bpe, src_in_file, src_out_file)
    encode_file(bpe, trg_in_file, trg_out_file)
    return src_out_file, trg_out_file


def main():
    print("Warning: Raw main() is deprecated. Please use main_wo_bpe().")


class CustomExample(object):
    """完美伪装原 torchtext.data.Example 对象，让后续模型训练能够通过 .src 和 .trg 属性读取数据"""
    def __init__(self, src_tokens, trg_tokens):
        self.src = src_tokens
        self.trg = trg_tokens


class CustomVocab(object):
    """极简词表类，用来替代旧版 Field 的字典属性，保持生成的 pkl 数据结构完全不崩"""
    def __init__(self):
        # 兼容不同版本的 Constants 属性名
        pad_idx = getattr(Constants, 'PAD', getattr(Constants, 'PAD_IDX', 0))
        bos_idx = getattr(Constants, 'BOS', getattr(Constants, 'BOS_IDX', 2))
        eos_idx = getattr(Constants, 'EOS', getattr(Constants, 'EOS_IDX', 3))
        unk_idx = getattr(Constants, 'UNK', getattr(Constants, 'UNK_IDX', 1))

        self.stoi = {
            Constants.PAD_WORD: pad_idx, 
            Constants.BOS_WORD: bos_idx, 
            Constants.EOS_WORD: eos_idx, 
            Constants.UNK_WORD: unk_idx
        }
        self.itos = [Constants.PAD_WORD, Constants.BOS_WORD, Constants.EOS_WORD, Constants.UNK_WORD]

    def build_vocab_from_tokens(self, all_tokens, min_freq):
        from collections import Counter
        counter = Counter(all_tokens)
        for word, freq in counter.items():
            if freq >= min_freq and word not in self.stoi:
                self.stoi[word] = len(self.itos)
                self.itos.append(word)


class CustomField(object):
    """极简 Field 类封装"""
    def __init__(self):
        self.vocab = CustomVocab()


def load_raw_data_and_tokenize(src_path, trg_path, tokenize_src_fn, tokenize_trg_fn, max_len):
    """纯 Python 的文本数据分词与过滤逻辑"""
    examples = []
    all_src_tokens = []
    all_trg_tokens = []
    
    with open(src_path, mode='r', encoding='utf-8') as src_file, \
         open(trg_path, mode='r', encoding='utf-8') as trg_file:
        for src_line, trg_line in zip(src_file, trg_file):
            src_line, trg_line = src_line.strip(), trg_line.strip()
            if src_line == '' or trg_line == '':
                continue
                
            src_tokens = tokenize_src_fn(src_line)
            trg_tokens = tokenize_trg_fn(trg_line)
            
            if len(src_tokens) <= max_len and len(trg_tokens) <= max_len:
                examples.append(CustomExample(src_tokens, trg_tokens))
                all_src_tokens.extend(src_tokens)
                all_trg_tokens.extend(trg_tokens)
                
    return examples, all_src_tokens, all_trg_tokens


def main_wo_bpe():
    '''
    Usage: python preprocess.py -lang_src de -lang_trg en -save_data multi30k_de_en.pkl -share_vocab
    '''
    spacy_support_langs = ['de', 'el', 'en', 'es', 'fr', 'it', 'lt', 'nb', 'nl', 'pt']

    parser = argparse.ArgumentParser()
    parser.add_argument('-lang_src', required=True, choices=spacy_support_langs)
    parser.add_argument('-lang_trg', required=True, choices=spacy_support_langs)
    parser.add_argument('-save_data', required=True)
    parser.add_argument('-data_src', type=str, default=None)
    parser.add_argument('-data_trg', type=str, default=None)

    parser.add_argument('-max_len', type=int, default=100)
    parser.add_argument('-min_word_count', type=int, default=3)
    parser.add_argument('-keep_case', action='store_true')
    parser.add_argument('-share_vocab', action='store_true')

    opt = parser.parse_args()
    print(opt)

    src_lang_model = spacy.load("de_core_news_sm" if opt.lang_src == "de" else "en_core_web_sm")
    trg_lang_model = spacy.load("de_core_news_sm" if opt.lang_trg == "de" else "en_core_web_sm")

    def tokenize_src(text):
        text = text.lower() if not opt.keep_case else text
        return [tok.text for tok in src_lang_model.tokenizer(text)]

    def tokenize_trg(text):
        text = text.lower() if not opt.keep_case else text
        return [tok.text for tok in trg_lang_model.tokenizer(text)]

    MAX_LEN = opt.max_len
    MIN_FREQ = opt.min_word_count

    data_dir = os.path.join('.data', 'multi30k')
    mkdir_if_needed(data_dir)
    
    train_src_path = os.path.join(data_dir, f'train.{opt.lang_src}')
    train_trg_path = os.path.join(data_dir, f'train.{opt.lang_trg}')
    val_src_path = os.path.join(data_dir, f'val.{opt.lang_src}')
    val_trg_path = os.path.join(data_dir, f'val.{opt.lang_trg}')
    test_src_path = os.path.join(data_dir, f'test2016.{opt.lang_src}')
    test_trg_path = os.path.join(data_dir, f'test2016.{opt.lang_trg}')

    if not os.path.exists(train_src_path):
        print(f"[Info] 正在尝试自动下载 Multi30k 原始翻译文本文件...")
        url = "https://raw.githubusercontent.com/neyo24/multi30k-dataset/master/data/task1/raw/"
        for prefix in ['train', 'val', 'test2016']:
            for lang in [opt.lang_src, opt.lang_trg]:
                f_name = f"{prefix}.{lang}"
                out_p = os.path.join(data_dir, f_name)
                if not os.path.exists(out_p):
                    print(f"Downloading {f_name}...")
                    urllib.request.urlretrieve(url + f_name + ".gz", out_p + ".gz")
                    with tarfile.open(out_p + ".gz", "r:gz") as f_gz:
                        f_gz.extractall(data_dir)

    print('[Info] 正在进行分词并清洗数据...')
    train_examples, train_src_tokens, train_trg_tokens = load_raw_data_and_tokenize(
        train_src_path, train_trg_path, tokenize_src, tokenize_trg, MAX_LEN
    )
    val_examples, _, _ = load_raw_data_and_tokenize(
        val_src_path, val_trg_path, tokenize_src, tokenize_trg, MAX_LEN
    )
    test_examples, _, _ = load_raw_data_and_tokenize(
        test_src_path, test_trg_path, tokenize_src, tokenize_trg, MAX_LEN
    )

    SRC_FIELD = CustomField()
    TRG_FIELD = CustomField()

    SRC_FIELD.vocab.build_vocab_from_tokens(train_src_tokens, MIN_FREQ)
    print('[Info] Get source language vocabulary size:', len(SRC_FIELD.vocab.itos))
    TRG_FIELD.vocab.build_vocab_from_tokens(train_trg_tokens, MIN_FREQ)
    print('[Info] Get target language vocabulary size:', len(TRG_FIELD.vocab.itos))

    if opt.share_vocab:
        print('[Info] Merging two vocabulary ...')
        for w, _ in SRC_FIELD.vocab.stoi.items():
            if w not in TRG_FIELD.vocab.stoi:
                TRG_FIELD.vocab.stoi[w] = len(TRG_FIELD.vocab.itos)
                TRG_FIELD.vocab.itos.append(w)
        SRC_FIELD.vocab.stoi = TRG_FIELD.vocab.stoi
        SRC_FIELD.vocab.itos = TRG_FIELD.vocab.itos
        print('[Info] Get merged vocabulary size:', len(TRG_FIELD.vocab.itos))

    data = {
        'settings': opt,
        'vocab': {'src': SRC_FIELD, 'trg': TRG_FIELD},
        'train': train_examples,
        'valid': val_examples,
        'test': test_examples}

    print('[Info] Dumping the processed data to pickle file', opt.save_data)
    pickle.dump(data, open(opt.save_data, 'wb'))
    print('[Info] 预处理圆满完成！')


if __name__ == '__main__':
    main_wo_bpe()