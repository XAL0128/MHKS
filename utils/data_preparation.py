import random
import argparse
import numpy as np
from transformers import BertTokenizer, XLNetTokenizer
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
import pickle


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, visual, acoustic, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.visual = visual
        self.acoustic = acoustic
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class CMUDataset:
    def __init__(self, config):
        self.config = config

    def prepare_bert_input(self, tokens, visual, acoustic, tokenizer):
        CLS = tokenizer.cls_token
        SEP = tokenizer.sep_token
        tokens = [CLS] + tokens + [SEP]

        # Pad zero vectors for acoustic / visual vectors to account for [CLS] / [SEP] tokens
        acoustic_zero = np.zeros((1, self.config.ACOUSTIC_DIM))
        acoustic = np.concatenate((acoustic_zero, acoustic, acoustic_zero))
        visual_zero = np.zeros((1, self.config.VISUAL_DIM))
        visual = np.concatenate((visual_zero, visual, visual_zero))

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        segment_ids = [0] * len(input_ids)
        input_mask = [1] * len(input_ids)

        pad_length = self.config.max_seq_length - len(input_ids)

        acoustic_padding = np.zeros((pad_length, self.config.ACOUSTIC_DIM))
        acoustic = np.concatenate((acoustic, acoustic_padding))

        visual_padding = np.zeros((pad_length, self.config.VISUAL_DIM))
        visual = np.concatenate((visual, visual_padding))

        padding = [0] * pad_length

        # Pad inputsae
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        return input_ids, visual, acoustic, input_mask, segment_ids

    def convert_to_features(self, examples, tokenizer):
        features = []

        for (ex_index, example) in enumerate(examples):

            (words, visual, acoustic), label_id, segment = example

            tokens, inversions = [], []
            for idx, word in enumerate(words):
                tokenized = tokenizer.tokenize(word)
                tokens.extend(tokenized)
                inversions.extend([idx] * len(tokenized))

            # Check inversion
            assert len(tokens) == len(inversions)

            aligned_visual = []
            aligned_audio = []

            for inv_idx in inversions:
                aligned_visual.append(visual[inv_idx, :])
                aligned_audio.append(acoustic[inv_idx, :])

            visual = np.array(aligned_visual)
            acoustic = np.array(aligned_audio)

            # Truncate input if necessary
            if len(tokens) > self.config.max_seq_length - 2:
                tokens = tokens[: self.config.max_seq_length - 2]
                acoustic = acoustic[: self.config.max_seq_length - 2]
                visual = visual[: self.config.max_seq_length - 2]

            input_ids, visual, acoustic, input_mask, segment_ids = self.prepare_bert_input(
                tokens, visual, acoustic, tokenizer
            )

            # Check input length
            assert len(input_ids) == self.config.max_seq_length
            assert len(input_mask) == self.config.max_seq_length
            assert len(segment_ids) == self.config.max_seq_length
            assert acoustic.shape[0] == self.config.max_seq_length
            assert visual.shape[0] == self.config.max_seq_length

            features.append(
                InputFeatures(
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    visual=visual,
                    acoustic=acoustic,
                    label_id=label_id,
                )
            )
        return features

    def get_appropriate_dataset(self, data):
        tokenizer = BertTokenizer.from_pretrained(self.config.pretrained)

        features = self.convert_to_features(data, tokenizer)
        all_input_ids = torch.tensor(
            [f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor(
            [f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor(
            [f.segment_ids for f in features], dtype=torch.long)
        all_visual = torch.tensor([f.visual for f in features], dtype=torch.float)
        all_acoustic = torch.tensor(
            [f.acoustic for f in features], dtype=torch.float)
        all_label_ids = torch.tensor(
            [f.label_id for f in features], dtype=torch.float)

        dataset = TensorDataset(
            all_input_ids,
            all_visual,
            all_acoustic,
            all_input_mask,
            all_segment_ids,
            all_label_ids,
        )
        return dataset


class SiMSDataset(Dataset):
    def __init__(self, dataset_path, mode='train'):
        self.mode = mode
        self.dataset_path = dataset_path

        self.__init_sims()

    def __init_sims(self):
        with open(self.dataset_path, "rb") as handle:
            data = pickle.load(handle)

        self.text = data[self.mode]['text_bert'].astype(np.float32)
        self.audio = data[self.mode]['audio'].astype(np.float32)
        self.vision = data[self.mode]['vision'].astype(np.float32)
        self.raw_text = data[self.mode]['raw_text']
        self.ids = data[self.mode]['id']

        self.labels = {'M': np.array(data[self.mode]['regression_labels']).astype(np.float32)}

        for m in "TAV":
            self.labels[m] = data[self.mode]['regression_labels_' + m].astype(np.float32)

        self.audio[self.audio == -np.inf] = 0

    def __len__(self):
        return len(self.labels['M'])

    def __getitem__(self, item):
        sample = {
            'raw_text': self.raw_text[item],
            'text': torch.Tensor(self.text[item]),
            'audio': torch.Tensor(self.audio[item]),
            'vision': torch.Tensor(self.vision[item]),
            'index': item,
            'id': self.ids[item],
            'labels': {k: torch.Tensor(v[item].reshape(-1)) for k, v in self.labels.items()}
        }

        return sample


def set_up_data_loader(config):
    dataset = config.dataset_name
    batch_size = config.batch_size

    if dataset == 'sims':
        datasets = {
            'train': SiMSDataset(dataset_path=config.data_path, mode='train'),
            'valid': SiMSDataset(dataset_path=config.data_path, mode='valid'),
            'test': SiMSDataset(dataset_path=config.data_path, mode='test')
        }

        dataLoader = {
            ds: DataLoader(datasets[ds],
                           batch_size=batch_size,
                           num_workers=4,
                           shuffle=True,
                           drop_last=True)
            for ds in datasets.keys()
        }
    else:
        with open(config.data_path, "rb") as handle:
            data = pickle.load(handle)

        train_data = data["train"]
        dev_data = data["dev"]
        test_data = data["test"]

        datasets = {
            'train': CMUDataset(config).get_appropriate_dataset(train_data),
            'valid': CMUDataset(config).get_appropriate_dataset(dev_data),
            'test': CMUDataset(config).get_appropriate_dataset(test_data)
        }

        dataLoader = {
            ds: DataLoader(datasets[ds],
                           batch_size=batch_size,
                           num_workers=4,
                           shuffle=True,
                           drop_last=True)
            for ds in datasets.keys()
        }

    return dataLoader
