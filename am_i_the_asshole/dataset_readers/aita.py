from typing import Dict
import logging
import pandas as pd
import numpy as np
from typing import List

from overrides import overrides

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, ArrayField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("aita")
class AitaReader(DatasetReader):
    """
    Dataset reader for AITA data
    """

    def __init__(self,
                 categorical: bool = False,
                 binary_categorical: bool = False,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy)
        self.categorical = categorical
        self.label_itos = ["NTA", "YTA", "ESH", "NAH", "INFO"]
        if binary_categorical:
            self.label_itos = ["NTA", "YTA", "YTA", "NTA", "NTA"]
            self.categorical = True

        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {
            "tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path):
        df = pd.read_pickle(file_path)
        for _, r in df.iterrows():
            yield self.text_to_instance(r["title"], r["selftext"], r["label_probs"])

    @overrides
    def text_to_instance(self, title: str, text: str, label: List[float] = None) -> Instance:
        tokenized_title = self._tokenizer.tokenize(title)
        title_field = TextField(tokenized_title, self._token_indexers)
        tokenized_text = self._tokenizer.tokenize(text)
        text_field = TextField(tokenized_text, self._token_indexers)
        fields = {'title': title_field, 'text': text_field}
        if label is not None:
            if self.categorical:
                fields["label"] = LabelField(self.label_itos[np.array(label).argmax()])
            else:
                fields["label"] = ArrayField(np.array(label))
        return Instance(fields)
