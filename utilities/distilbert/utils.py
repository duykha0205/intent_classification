from framework.models.base_model import BaseModel
import os
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from utilities.helper import tokenize

distilbert_cfgs = {
    "save_dir": os.environ.get("MODEL_DIR"),
    "distbert_model": os.environ.get("DISTILBERT_MODEL"),
    "load_pretrain": True,
    "distilbert_tokenizer": os.environ.get("DISTILBERT_TOKENIZER"),
    "device": os.environ.get("DEVICE")
}


class DistilBert(BaseModel):
    def __init__(self, cfgs, tokenize_text):
        super(DistilBert, self).__init__(cfgs, tokenize_text)
        if cfgs["load_pretrain"]:
            self.load_models(cfgs)

        if cfgs["distilbert_tokenizer"]:
            self.tokenizer = DistilBertTokenizer.from_pretrained(cfgs["distilbert_tokenizer"])

    def load_models(self, model_cfg):
        self.model = DistilBertForSequenceClassification.from_pretrained(
            os.path.join(model_cfg['save_dir'], model_cfg['distbert_model']))
        self.model.to(model_cfg['device'])

    def inputs_pretreatment(self, inputs):
        return self.tokenizer(inputs, padding="max_length", truncation=True, return_tensors="pt")

    def forward(self, inputs):
        tokenized = self.inputs_pretreatment(inputs)

        with torch.no_grad():
            predict = self.model(**tokenized).logits

        return list(map(self.softmax, predict.numpy()))


distilbert_model = DistilBert(distilbert_cfgs, tokenize.en_tokenize)
