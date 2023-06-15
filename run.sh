#!/bin/bash
port=$1
#
## Dev environment setup
export PYTHONPATH=.
export DEVICE=cpu
export MODEL_DIR=parameters

export MXNET_ENGINE_TYPE=NaiveEngine

## For BERT method
export BERT_MODEL=bert_12_768_12
export BERT_DATASET_NAME=book_corpus_wiki_en_uncased
export BERT_PRETRAIN_NAME=bert-v3.params
export BERT_CASED=false

## Fasttext
export FASTTEXT_MODEL=fasttext-v3.bin

## Fasttext
export FASTTEXT_VI_MODEL=fasttext-vi.bin

## DistilBert
export DISTILBERT_MODEL=distbert-v3
export DISTILBERT_TOKENIZER=distilbert-base-uncased

## PhoBert
export PHOBERT_MODEL=phobert
export PHOBERT_TOKENIZER=vinai/phobert-base

## Ensemble
export ENSEMBLE_MODEL=ensemble_mlp-v3.pkl

## Ensemble VI
export ENSEMBLE_VI_MODEL=ensemble_mlp-vi.pkl

# Start server
python debug.py $1
