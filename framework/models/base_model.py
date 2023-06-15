from . import MetaModel
import numpy as np


class BaseModel(MetaModel):
    def __init__(self, cfgs, tokenize_text):
        super(BaseModel, self).__init__()
        self.cfgs = cfgs
        self.tokenize_text = tokenize_text

    def softmax(self, vec):
        exponential = np.exp(vec)
        probabilities = exponential / np.sum(exponential)
        return probabilities

    def load_models(self, model_cfg):
        pass

    def inputs_pretreatment(self, inputs):
        pass

    def forward(self, inputs):
        pass

    def inference(self, sentences, output_type="label", method=None):
        dict_map = {
            0: "no",
            1: "yes"
        }
        result = []
        tokenized = list(map(self.tokenize_text, sentences))
        predict = self.forward(tokenized)
        pred = np.array(predict)

        if output_type == "label":
            pred = pred.argmax(axis=-1)

        for idx in range(len(sentences)):
            data = {
                "idx": idx + 1,
                "input": sentences[idx],
                "tokenized": tokenized[idx],
                "output": dict_map[pred[idx]] if output_type == "label" else float(pred[idx][1])
            }
            result.append(data)

        return result
