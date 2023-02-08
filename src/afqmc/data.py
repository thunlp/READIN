import os.path as osp

import torch
from torch.utils.data import Dataset, DataLoader

import utils


def get_examples(file: str, set_type: str) -> list:
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(utils.iter_jsonl(file)):
        guid = "%s-%s" % (set_type, i)
        text_a = line['sentence1']
        text_b = line['sentence2']
        # label = str(line['label']) if set_type != 'test' else "0"
        label = str(line['label'])
        examples.append({
            'guid': guid, 
            'text': [text_a, text_b],
            'label': label,
        })
    return examples

def get_train_examples(data_dir):
    return get_examples(osp.join(data_dir, "train.json"), "train")
def get_dev_examples(data_dir):
    return get_examples(osp.join(data_dir, "dev.json"), "dev")
def get_test_examples(data_dir):
    return get_examples(osp.join(data_dir, "test.json"), "test")
def get_noisy_test_examples(data_dir):
    return get_examples(osp.join(data_dir, "noisy_test.json"), "noisy_test")


class AfqmcDataset(Dataset):
    def __init__(self, file: str, phase: str, tokenizer, max_seq_len: int, 
                 num_examples: int=None):
        self.file = file
        print('Loading examples from:', file)
        examples = get_examples(file, phase)[:num_examples]
        self.features = self.get_features(examples, tokenizer, max_seq_len)

    def get_features(self, examples: list, tokenizer, max_seq_len: int) -> dict:
        """
        Return list of examples (dict) into dict of features:
        ```
        {
            'input_ids': [...],
            'token_type_ids': [...],
            'attention_mask': [...],
            'labels': [...],
        }
        ```
        """
        label_list = ['0', '1']
        label_map = {label: i for i, label in enumerate(label_list)}

        features = []
        texts = [x['text'] for x in examples]
        features = tokenizer(
            texts,
            max_length=max_seq_len,
            truncation='longest_first',
            padding='max_length',
            return_tensors='pt')
        features['labels'] = torch.tensor([label_map[x['label']] for x in examples])
        return features

    def __getitem__(self, idx):
        return {
            k: self.features[k][idx] for k in 
            ['input_ids', 'token_type_ids', 'attention_mask', 'labels']
        }
    
    def __len__(self):
        return len(self.features['input_ids'])

class AfqmcSeq2SeqDataset(Dataset):
    def __init__(self, file: str, phase: str, tokenizer, num_examples: int=None):
        # self.verbalizer = ['不等价', '等价']
        self.verbalizer = ['nonequivalent', 'equivalent']
        self.file = file
        self.tokenizer = tokenizer
        self.phase = phase

        examples = get_examples(file, phase)[:num_examples]
        self.features = self.get_features(examples, tokenizer)

    def get_features(self, examples: list, tokenizer) -> dict:
        '''
        A feature for seq2seq is a pair of input_ids and labels.

        input text template:  "句子1：{}，句子2：{}。"
        output text template: "{}"

        Return:
        ```
        {
            'input_ids': [...],
            'attention_mask': [...],
            'labels': [...],
            'label_ids': [...],
        }
        ```
        '''
        source_template = '句子1：{}，句子2：{}。'
        texts = [source_template.format(ex['text'][0], ex['text'][1]) for ex in examples]
        label_ids = [int(ex['label']) for ex in examples]
        labels = [self.verbalizer[label_id] for label_id in label_ids]
        inputs = tokenizer(texts, padding='longest', return_tensors='pt')  # {'input_ids': Tensor, 'attention_mask': Tensor}
        labels = tokenizer(labels, padding='longest', return_tensors='pt').input_ids
        inputs['labels'] = labels
        # inputs['label_ids'] = torch.tensor(label_ids)
        inputs['label_ids'] = label_ids
        return inputs

    def __getitem__(self, idx):
        if 'test' in self.phase:
            return {
                k: self.features[k][idx] for k in 
                ['input_ids', 'attention_mask', 'labels', 'label_ids']
            }
        else:
            return {
                k: self.features[k][idx] for k in 
                ['input_ids', 'attention_mask', 'labels']
            }

    def __len__(self):
        return len(self.features['input_ids'])
