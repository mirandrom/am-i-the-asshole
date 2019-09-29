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
from allennlp.training.metrics import MeanAbsoluteError


@Model.register("aita_regressor")
class AitaRegressor(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 title_encoder: Seq2VecEncoder,
                 text_encoder: Seq2VecEncoder,
                 regressor_feedforward: FeedForward,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.title_encoder = title_encoder
        self.text_encoder = text_encoder
        self.regressor_feedforward = regressor_feedforward

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
                "MAE": MeanAbsoluteError(),
        }
        self.loss = torch.nn.BCEWithLogitsLoss()
        initializer(self)

    @overrides
    def forward(self,
                title: Dict[str, torch.LongTensor],
                text: Dict[str, torch.LongTensor],
                label: torch.FloatTensor = None) -> Dict[str, torch.Tensor]:
        embedded_title = self.text_field_embedder(title)
        title_mask = util.get_text_field_mask(title)
        encoded_title = self.title_encoder(embedded_title, title_mask)

        embedded_text = self.text_field_embedder(text)
        text_mask = util.get_text_field_mask(text)
        encoded_text = self.text_encoder(embedded_text, text_mask)

        logits = self.regressor_feedforward(torch.cat([encoded_title, encoded_text], dim=-1))
        output_dict = {'logits': logits}
        if label is not None:
            loss = self.loss(logits, label)
            for metric in self.metrics.values():
                metric(logits, label)
            output_dict["loss"] = loss

        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}
        return metrics