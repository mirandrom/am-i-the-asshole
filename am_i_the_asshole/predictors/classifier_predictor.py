from overrides import overrides
from torch.nn import functional as F
from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor
import numpy as np
import torch


@Predictor.register('aita_classifier')
class AitaPredictor(Predictor):
    def predict_json(self, inputs: JsonDict) -> JsonDict:
        instance = self._json_to_instance(inputs)
        output_dict = self.predict_instance(instance)
        output_dict["all_labels"] = ["NTA", "YTA", "ESH", "NAH", "INFO"]
        return output_dict

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        title = json_dict['title']
        text = json_dict['text']
        return self._dataset_reader.text_to_instance(title=title, text=text)


@Predictor.register('aita_binary_classifier')
class AitaBinaryPredictor(Predictor):
    def predict_json(self, inputs: JsonDict) -> JsonDict:
        instance = self._json_to_instance(inputs)
        output_dict = self.predict_instance(instance)
        label_dict = self._model.vocab.get_index_to_token_vocabulary('labels')
        all_labels = [label_dict[i] for i in range(len(label_dict))]
        output_dict["all_labels"] = all_labels
        return output_dict

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        title = json_dict['title']
        text = json_dict['text']
        return self._dataset_reader.text_to_instance(title=title, text=text)