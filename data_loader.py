from typing import Optional, Union, List, Dict
import json
from dataclasses import dataclass

import tensorflow as tf
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2Processor, AutoFeatureExtractor
from datasets import load_dataset, DatasetDict


def create_vocab_file(dataset):
    def extract_all_chars(batch):
        all_text = " ".join(batch["phonemes"])
        # phonemes are split by whitespace
        vocab = list(set(all_text.split())) + [" "]
        return {"vocab": [vocab], "all_text": [all_text]}

    vocabs = dataset.map(
        extract_all_chars,
        batched=True,
        batch_size=-1,
        keep_in_memory=True,
        remove_columns=dataset.column_names["train"]
    )

    vocab_list = list(set(vocabs["train"]["vocab"][0])
                      | set(vocabs["test"]["vocab"][0]))
    vocab_dict = {v: k for k, v in enumerate(vocab_list)}
    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)

    with open('vocab.json', 'w') as vocab_file:
        json.dump(vocab_dict, vocab_file)


def prepare_dataset(batch, processor):
    audio = batch["audio"]

    batch["input_values"] = processor(
        audio["array"], sampling_rate=16000).input_values[0]
    batch["input_length"] = len(batch["input_values"])

    return batch


def representative_dataset():
    pass


def create_data_loader():
    print('Loading dataset...')
    dataset = load_dataset("w11wo/ljspeech_phonemes", split='train')

    dataset = DatasetDict({'train': dataset})
    dataset = dataset .remove_columns(['id', 'text', 'normalized_text'])
    dataset = dataset['train'].train_test_split(test_size=0.1)

    print('Building vocabulary...')
    create_vocab_file(dataset)
    tokenizer = Wav2Vec2CTCTokenizer(
        "./vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        "facebook/wav2vec2-base-960h")
    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor, tokenizer=tokenizer)

    dataset = dataset.map(
        lambda x: prepare_dataset(x, processor), remove_columns=dataset .column_names["train"], num_proc=4)
    print('Done')
    import pdb
    pdb.set_trace()


@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], tf.Tensor]]]) -> Dict[str, tf.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]}
                          for feature in features]
        label_features = [{"input_ids": feature["labels"]}
                          for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="tf",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="tf",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch
