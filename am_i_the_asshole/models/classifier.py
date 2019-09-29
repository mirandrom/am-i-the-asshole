from typing import Dict, Optional

import numpy
from overrides import overrides
import torch
import torch.nn.functional as F

from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import FeedForward, Seq2VecEncoder, TextFieldEmbedder
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy


@Model.register("aita_classifier")
class AitaClassifier(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 title_encoder: Seq2VecEncoder,
                 text_encoder: Seq2VecEncoder,
                 classifier_feedforward: FeedForward,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.title_encoder = title_encoder
        self.text_encoder = text_encoder
        self.classifier_feedforward = classifier_feedforward

        if text_field_embedder.get_output_dim() != title_encoder.get_input_dim():
            raise ConfigurationError(f"The output dimension of the text_field_"
                                     f"embedder must match the input dimension"
                                     f" of the summary_encoder. Found "
                                     f"{text_field_embedder.get_output_dim()} "
                                     f"and {title_encoder.get_input_dim()}, "
                                     f"respectively.")

        if text_field_embedder.get_output_dim() != text_encoder.get_input_dim():
            raise ConfigurationError(f"The output dimension of the text_field_"
                                     f"embedder must match the input dimension"
                                     f" of the summary_encoder. Found "
                                     f"{text_field_embedder.get_output_dim()} "
                                     f"and {text_encoder.get_input_dim()}, "
                                     f"respectively.")

        self.metrics = {
            "accuracy": CategoricalAccuracy(),
            "accuracy3": CategoricalAccuracy(top_k=3)
        }
        self.loss = torch.nn.CrossEntropyLoss()
        initializer(self)

    @overrides
    def forward(self,
                title: Dict[str, torch.LongTensor],
                text: Dict[str, torch.LongTensor],
                label: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        embedded_title = self.text_field_embedder(title)
        title_mask = util.get_text_field_mask(title)
        encoded_title = self.title_encoder(embedded_title, title_mask)

        embedded_text = self.text_field_embedder(text)
        text_mask = util.get_text_field_mask(text)
        encoded_text = self.text_encoder(embedded_text, text_mask)

        logits = self.classifier_feedforward(torch.cat([encoded_title, encoded_text], dim=-1))
        output_dict = {'logits': logits}
        if label is not None:
            loss = self.loss(logits, label)
            for metric in self.metrics.values():
                metric(logits, label)
            output_dict["loss"] = loss

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does a simple argmax over the class probabilities, converts indices to string labels, and
        adds a ``"label"`` key to the dictionary with the result.
        """
        class_probabilities = F.softmax(output_dict['logits'], dim=-1)
        output_dict['class_probabilities'] = class_probabilities

        predictions = class_probabilities.cpu().data.numpy()
        argmax_indices = numpy.argmax(predictions, axis=-1)
        labels = [self.vocab.get_token_from_index(x, namespace="labels")
                  for x in argmax_indices]
        output_dict['label'] = labels
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}
        return metrics