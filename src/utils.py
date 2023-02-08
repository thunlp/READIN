# coding: utf8
import json
from argparse import Namespace
from pathlib import Path
import random

import torch
import numpy as np


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def iter_jsonl(file):
    for line in open(file, 'r', encoding='utf8'):
        yield json.loads(line.strip())


def dump_jsonl(data, file: Path, **kwargs):
    with open(file, 'w', encoding='utf8') as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False, **kwargs) + '\n')


def load_jsonl(file, cnt=None):
    if cnt:
        data = []
        for line in open(file, 'r', encoding='utf8'):
            data.append(json.loads(line))
            if len(data) == cnt:
                break
        return data
    else:
        return [json.loads(line) for line in open(file, 'r', encoding='utf8')]


def dump_json(data, file: Path, **kwargs):
    json.dump(data, open(file, 'w', encoding='utf8'),
              ensure_ascii=False, **kwargs)


def load_json(file: Path):
    return json.load(open(file, 'r', encoding='utf8'))


def dump_str(data, file: Path):
    file.parent.mkdir(parents=True, exist_ok=True)
    with file.open('w') as f:
        f.write(str(data))


def dump_args(args: Namespace, file: Path):
    file.parent.mkdir(parents=True, exist_ok=True)
    s = json.dumps(vars(args), indent=2, ensure_ascii=False)
    file.open('w').write(s)
    print(s)


def get_acc(preds: np.array, labels: np.array) -> float:
    assert len(preds) == len(labels)
    if isinstance(preds, np.array) and isinstance(labels, np.array):
        return np.mean(np.argmax(preds, axis=1) == labels)
    else:
        correct = 0
        for a, b in zip(preds, labels):
            if a == b:
                correct += 1
        return correct / len(preds)


def get_param_count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


class Logger:
    def __init__(self, out_file: Path):
        self.out_path = Path(out_file)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_file = self.out_path.open('w')

    def log(self, *args, **kwargs):
        print(*args, **kwargs, flush=True)
        print(*args, **kwargs, file=self.out_file, flush=True)
