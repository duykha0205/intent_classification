import numpy as np
import os
import pickle
from utilities.helper import tokenize
from utilities.distilbert.utils import distilbert_model
from utilities.bert.utils import bert_model
from utilities.phobert.utils import phobert_model
from utilities.fasttext.utils import fasttext_model
from utilities.fasttext_vi.utils import fasttext_vi_model

ensemble_cfgs = {
    "save_dir": os.environ.get("MODEL_DIR"),
    "language": {
        "en": {
            "ensemble_model": os.environ.get("ENSEMBLE_MODEL"),
            "tokenize": tokenize.en_tokenize,
            "models": [
                ("fasttext", fasttext_model),
                ("distilbert", distilbert_model),
                ("bert", bert_model)
            ]
        },
        "vi": {
            "ensemble_model": os.environ.get("ENSEMBLE_VI_MODEL"),
            "tokenize": tokenize.vi_tokenize,
            "models": [
                ("fasttext", fasttext_vi_model),
                ("phobert", phobert_model)
            ]
        }
    },

}


def round(a):
    if a > 0.5:
        return 1
    return 0


class Ensemble:
    def __init__(self, ensemble_cfgs):
        self.model = {}
        self.estimators_ = {}
        self.tokenize_text = {}
        self.load_models(ensemble_cfgs)
        self.load_tokenizer(ensemble_cfgs)

    def load_tokenizer(self, model_cfg):
        for lang in model_cfg["language"]:
            self.tokenize_text[lang] = model_cfg["language"][lang]['tokenize']

    def load_models(self, model_cfg):
        for lang in model_cfg["language"]:
            self.estimators_[lang] = model_cfg["language"][lang]['models']

            with open(os.path.join(model_cfg['save_dir'], model_cfg["language"][lang]['ensemble_model']), 'rb') as file:
                self.model[lang] = pickle.load(file)

    def forward(self, tokenized_sentences, language):
        probabilities = []
        for clf in self.estimators_[language]:
            probabilities.append(np.array(clf[1].forward(tokenized_sentences)))

        return probabilities

    def voting(self, probabilities, language):
        predict_labels = np.array(probabilities).argmax(axis=-1)

        def vote(predictions):
            majority = np.argmax(np.bincount(predictions))
            return [1 - majority, majority]

        return np.apply_along_axis(vote, 0, predict_labels).T

    def stacking(self, probabilities, language):
        probabilities = np.array(probabilities)[:, :, -1].T
        return self.model[language].predict_proba(probabilities)

    def inference(self, sentences, language="vi", output_type="label", method=None):
        dict_map = {
            0: "no",
            1: "yes"
        }

        result = []
        tokenized = list(map(self.tokenize_text[language], sentences))
        probabilities = self.forward(tokenized, language)
        predict_labels = np.array(probabilities)
        ensemble_labels = method(probabilities, language)

        if output_type == "label":
            predict_labels = np.array(probabilities).argmax(axis=-1)
            ensemble_labels = np.array(ensemble_labels).argmax(axis=-1)

        for idx in range(len(sentences)):
            data = {
                "idx": idx + 1,
                "input": sentences[idx],
                "tokenized": tokenized[idx],
                "ensemble_output": dict_map[ensemble_labels[idx]] if output_type == "label" else float(ensemble_labels[idx][1])
            }

            for model in range(len(self.estimators_[language])):
                data[self.estimators_[language][model][0] + "_output"] = \
                    dict_map[predict_labels[model][idx]] if output_type == "label" else float(predict_labels[model][idx][1])

            result.append(data)

        return result


ensemble_model = Ensemble(ensemble_cfgs)
