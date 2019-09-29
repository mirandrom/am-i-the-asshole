from overrides import overrides
from torch.nn import functional as F
from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor
import numpy as np
import torch


@Predictor.register('aita_regressor')
class AitaPredictor(Predictor):
    """"Predictor wrapper for the HumanScoreClassifier"""
    def predict_json(self, inputs: JsonDict) -> JsonDict:
        instance = self._json_to_instance(inputs)
        output_dict = self.predict_instance(instance)
        output_dict["all_labels"] = ["NTA", "YTA", "ESH", "NAH", "INFO"]
        output_dict["class_probabilities"] = F.sigmoid(torch.from_numpy(
            np.array(output_dict["logits"])
        )).tolist()
        return output_dict

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        title = json_dict['title']
        text = json_dict['text']
        return self._dataset_reader.text_to_instance(title=title, text=text)
