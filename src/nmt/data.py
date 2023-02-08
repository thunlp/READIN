from torch.utils.data import Dataset

from utils import load_jsonl

class NmtDataset(Dataset):
    def __init__(self, 
        examples_file, 
        tokenizer, 
        max_len: int=1024, 
        num_examples: int=None):
        self.tokenizer = tokenizer
        self.examples = load_jsonl(examples_file)[:num_examples]
        self.max_len = max_len
        self.features = self.get_features(self.examples, tokenizer)

    def get_features(self, examples, tokenizer):
        zh_lines = [x['zh'] for x in examples]
        # en_lines = [x['en'] for x in examples]
        features = tokenizer(zh_lines, max_length=self.max_len, padding=True, truncation=True, return_tensors="pt")
        return features

    def __getitem__(self, idx: int) -> dict:
        return {
            key: self.features[key][idx] for key in self.features
        }

    def __len__(self) -> int:
        return len(self.features["input_ids"])