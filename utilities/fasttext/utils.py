from framework.models.base_model import BaseModel
import fasttext
import os
from utilities.helper import tokenize

fasttext_cfgs = {
    "save_dir": os.environ.get("MODEL_DIR"),
    "fasttext_model": os.environ.get("FASTTEXT_MODEL"),
    "load_pretrain": True
}

# ignore warning of fasttext
try:
    fasttext.FastText.eprint = lambda *args, **kwargs: None
except:
    pass


def bound(a):
    if a > 1:
        return 1
    if a < 0:
        return 0
    return a


class Fasttext(BaseModel):
    def __init__(self, cfgs, tokenize_text):
        super(Fasttext, self).__init__(cfgs, tokenize_text)
        if cfgs["load_pretrain"]:
            self.load_models(cfgs)

    def load_models(self, model_cfg):
        self.model = fasttext.load_model(os.path.join(model_cfg['save_dir'], model_cfg['fasttext_model']))

    def forward(self, inputs):
        labels, probabilities = self.model.predict(inputs)
        output = []

        for i in range(len(inputs)):
            p = bound(probabilities[i][0])
            if labels[i][0] == '__label__no':
                output.append([p, 1 - p])
            else:
                output.append([1 - p, p])

        return output


fasttext_model = Fasttext(fasttext_cfgs, tokenize.en_tokenize)
