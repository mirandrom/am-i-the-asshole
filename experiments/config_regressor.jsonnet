{
  "dataset_reader": {
    "type": "aita",
    "token_indexers": {
      "tokens": {
        "type": "bert-pretrained",
        "pretrained_model": "bert-base-uncased",
        "truncate_long_sequences": true,
      }
    },
    "tokenizer": {
      "type": "word"
    }
  },
  "train_data_path": "data/aita_2019_posts_labeled_train.pkl",
  "validation_data_path": "data/aita_2019_posts_labeled_test.pkl",
  "model": {
    "type": "aita_regressor",
    "text_field_embedder": {
      "tokens": {
        "type": "bert-pretrained",
        "pretrained_model": "bert-base-uncased",
        "requires_grad": true
      },
      "allow_unmatched_keys": true
    },
    "title_encoder": {
      "type": "bert-sentence-pooler",
    },
    "text_encoder": {
      "type": "bert-sentence-pooler",
    },
    "regressor_feedforward": {
      "input_dim": 768*2,
      "num_layers": 3,
      "hidden_dims": [768*2, 768, 5],
      "activations": ["relu", "relu", "linear"],
      "dropout": [0.2, 0.0, 0.0]
    }
  },
  "iterator": {
    "type": "bucket",
    "sorting_keys": [["text", "num_tokens"]],
    "batch_size": 4,
  },
  "trainer": {
    "num_epochs": 40,
    "patience": 10,
    "cuda_device": [0,1,2],
    "validation_metric": "-MAE",
    "optimizer": {
      "type": "adagrad"
    }
  }
}