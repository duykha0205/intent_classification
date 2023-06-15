from framework.models.base_model import BaseModel
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utilities.helper import tokenize

phobert_cfgs = {
    "save_dir": os.environ.get("MODEL_DIR"),
    "phobert_model": os.environ.get("PHOBERT_MODEL"),
    "load_pretrain": True,
    "phobert_tokenizer": os.environ.get("PHOBERT_TOKENIZER"),
    "device": os.environ.get("DEVICE")
}


class PhoBert(BaseModel):
    def __init__(self, cfgs, tokenize_text):
        super(PhoBert, self).__init__(cfgs, tokenize_text)
        if cfgs["load_pretrain"]:
            self.load_models(cfgs)

        if cfgs["phobert_tokenizer"]:
            self.tokenizer = AutoTokenizer.from_pretrained(cfgs["phobert_tokenizer"])

    def load_models(self, model_cfg):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            os.path.join(model_cfg['save_dir'], model_cfg['phobert_model']))
        self.model.to(model_cfg['device'])

    def inputs_pretreatment(self, inputs):
        return self.tokenizer(inputs, padding="max_length", truncation=True, return_tensors="pt")

    def forward(self, inputs):
        tokenized = self.inputs_pretreatment(inputs)

        with torch.no_grad():
            predict = self.model(**tokenized).logits

        return list(map(self.softmax, predict.numpy()))


phobert_model = PhoBert(phobert_cfgs, tokenize.vi_tokenize)
