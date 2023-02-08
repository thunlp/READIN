import torch
assert torch.cuda.is_available()

from pathlib import Path

from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

from utils import load_jsonl, dump_jsonl, get_param_count, Logger
from nmt.data import NmtDataset


logger = None
def log(*args, **kwargs): logger.log(*args, **kwargs)


def get_dataset(tokenizer, file, num_examples=None):
    return NmtDataset(file, tokenizer, num_examples=num_examples)


def test(model, dataset, output_dir: Path, batch_size=16):
    log('Building dataloader...')
    dataloader = DataLoader(dataset, batch_size=batch_size)

    # Results
    preds = []
    
    log('*** Testing ***')
    log(f'Batch size: {batch_size}')
    log(f'# steps: {len(dataloader)}')
    log(f'# examples: {len(dataset)}')
    for batch in tqdm(dataloader):
        for k, v in batch.items(): batch[k] = v.cuda()  # Move to GPU
        generated_tokens = model.generate(**batch)
        output = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)  # Will move to CPU automatically
        preds += output
    log('*** Done testing ***')

    # Dump results
    output_dir.mkdir(exist_ok=True, parents=True)
    log(f'Dumping result to {output_dir}')
    dump_jsonl(preds, output_dir / 'preds.json')


def test_all(model, tokenizer, data_dir: Path, output_dir: Path):
    log('Testing all...')
    
    # Test clean 
    file_examples = data_dir / 'test_clean.json'
    log(f'*** Testing phase: clean ***')
    log(f'Loading from {file_examples}')
    dataset = get_dataset(tokenizer, file_examples)
    test(model, dataset, output_dir / 'test_clean')
    
    # Test noisy
    for noise_type in ['keyboard', 'asr']:
        for i in range(1, 4):
            test_name = f'test_noisy_{noise_type}_{i}'
            log(f'*** Testing phase: {test_name} ***')
            file_examples = data_dir / f'{test_name}.json'
            dataset = get_dataset(tokenizer, file_examples)
            test(model, dataset, output_dir / test_name)


if __name__ == '__main__':
    model_path = "facebook/mbart-large-50-many-to-one-mmt"

    output_dir = Path('results/nmt/mbart-large')
    data_dir = Path('../data/realtypo/nmt')
    log_file = output_dir / 'test.log'

    logger = Logger(log_file)

    log('Getting model...')
    model = MBartForConditionalGeneration.from_pretrained(model_path).cuda()
    log('Getting tokenizer...')
    tokenizer = MBart50TokenizerFast.from_pretrained(model_path)
    tokenizer.src_lang = "zh_CN"
    log(f'# params: {get_param_count(model)}')
    
    # Test
    test_all(model, tokenizer, data_dir, output_dir)
