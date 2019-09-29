from allennlp.modules import Seq2VecEncoder
import torch
from overrides import overrides


@Seq2VecEncoder.register("bert-sentence-pooler")
class BertSentencePooler(Seq2VecEncoder):
    def __init__(self, bert_dim: int = 768, stateful: bool = False):
        self.bert_dim = bert_dim
        super().__init__(stateful=stateful)

    def forward(self, embs: torch.tensor, mask: torch.tensor = None) -> torch.tensor:
        # extract first token tensor
        return embs[:, 0]

    @overrides
    def get_output_dim(self) -> int:
        return self.bert_dim

    @overrides
    def get_input_dim(self) -> int:
        return self.bert_dim


