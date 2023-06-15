import os
import pandas as pd
from mxnet.gluon import nn, Block
import time
import argparse
import numpy as np
import mxnet as mx
import random
from mxnet import gluon
import gluonnlp as nlp
from gluonnlp.data import BERTTokenizer, ATISDataset, SNIPSDataset
from gluonnlp.data import BERTSentenceTransform
from seqeval.metrics import f1_score as ner_f1_score
# from bert import data
import io
# from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
from framework.models.base_model import BaseModel
import gluonnlp.data.batchify as bf
from utilities.helper import tokenize

bert_cfgs = {
    "bert_model": os.environ.get("BERT_MODEL"),
    "dataset_name": os.environ.get("BERT_DATASET_NAME"),
    "ctx": mx.gpu(0) if os.environ.get("DEVICE") == "gpu" else mx.cpu(),
    "save_dir": os.environ.get("MODEL_DIR"),
    "pretrain_name": os.environ.get("BERT_PRETRAIN_NAME"),
    "cased": False
}


class BERT(BaseModel):
    def __init__(self, cfgs, tokenize_text):
        super(BERT, self).__init__(cfgs, tokenize_text)
        self.cfgs = cfgs
        self.model = self.load_models(cfgs)
        self.tokenizer = None
        self.model = self.load_models(cfgs)

    def load_models(self, model_cfg):
        # dataset_name = 'book_corpus_wiki_en_cased' if model_cfg['cased'] else 'book_corpus_wiki_en_uncased'
        bert_model, bert_vocab = nlp.model.get_model(name=model_cfg['bert_model'],
                                                     dataset_name=model_cfg['dataset_name'],
                                                     pretrained=False,
                                                     ctx=model_cfg['ctx'],
                                                     use_pooler=True,
                                                     use_decoder=False,
                                                     use_classifier=False,
                                                     dropout=0.1,
                                                     embed_dropout=0.1)

        net = nlp.model.BERTClassifier(bert_model, num_classes=2, dropout=0.1)
        net.load_parameters(os.path.join(model_cfg['save_dir'], model_cfg['pretrain_name']), ctx=model_cfg['ctx'])

        self.tokenizer = BERTTokenizer(bert_vocab, lower=not self.cfgs['cased'])
        return net

    def inputs_pretreatment(self, inputs):
        max_len = 128
        token_ids = []
        valid_lengths = []
        segment_ids = []

        trans_data_bert = BERTSentenceTransform(self.tokenizer, max_len, vocab=None, pad=False, pair=False)
        for inp in inputs:
            token_id, valid_length, segment_id = trans_data_bert(tuple([inp]))
            token_ids.append(list(token_id))
            valid_lengths.append(valid_length)
            segment_ids.append(list(segment_id))
        # padding
        token_pad_ids = bf.Pad(axis=-1, pad_val=1)(token_ids[:])
        segment_pad_ids = bf.Pad(axis=-1, pad_val=0)(segment_ids[:])

        token_ids = mx.nd.array(token_pad_ids, ctx=self.cfgs['ctx']).astype(np.int32)
        segment_ids = mx.nd.array(segment_pad_ids, ctx=self.cfgs['ctx']).astype(np.int32)
        valid_lengths = mx.nd.array(valid_lengths, ctx=self.cfgs['ctx']).astype(np.float32)

        return token_ids, valid_lengths, segment_ids

    def forward(self, inputs):
        token_ids, valid_lengths, segment_ids = self.inputs_pretreatment(inputs)
        intent_scores = self.model(token_ids, segment_ids, valid_lengths)
        return list(map(self.softmax, intent_scores.asnumpy()))


bert_model = BERT(bert_cfgs, tokenize.en_tokenize)
