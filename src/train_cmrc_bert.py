# coding: utf8
import torch
assert torch.cuda.is_available()

from pathlib import Path

from transformers import BertForQuestionAnswering, BertTokenizer
from cmrc2018 import CMRC2018Dataset, Trainer
import utils
import arguments


def test(trainer: Trainer, dataset: CMRC2018Dataset, output_dir: Path, desc: str):
    '''
    Test model on given dataset
    '''
    eval_output = trainer.evaluate(dataset, output_dir, is_labeled=True, desc=f'Testing {desc}')

    # Save results
    result = eval_output['result']
    preds = eval_output['preds']
    print(result, flush=True)
    utils.dump_json(result, output_dir / 'result.json', indent=2)
    utils.dump_json(preds, output_dir / 'preds.json', indent=2)


def test_all(trainer: Trainer, data_dir: Path, output_dir: Path, tok_name: str):
    print('Loading best model', flush=True)
    trainer.load_best_model(output_dir)
    
    # Clean
    print('*** Testing clean ***')
    file_examples = data_dir / 'test_clean.json'
    data = CMRC2018Dataset(trainer.tokenizer, file_examples, has_labels=True, 
                           tok_name=tok_name)
    test(trainer, data, output_dir / 'test_clean', 'clean')
    
    for noise_type in [
        'keyboard',
        'asr',
    ]:
        for i in range(1, 4):
            phase_name = f'test_noisy_{noise_type}_{i}'
            print(f'*** Testing {phase_name} ***')
            file_examples = data_dir / f'{phase_name}.json'
            dataset = CMRC2018Dataset(trainer.tokenizer, file_examples, 
                has_labels=True, tok_name=tok_name)
            test(trainer, dataset, output_dir=output_dir / phase_name, 
                desc=phase_name)


def main():
    args = arguments.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Model
    print('Loading model', flush=True)
    model = BertForQuestionAnswering.from_pretrained(args.model_path)
    tokenizer = BertTokenizer.from_pretrained(args.model_path)
    tok_name = args.model_path.split('/')[-1]

    print(f'# params: {utils.get_param_count(model)}', flush=True)

    utils.set_seed(0)
    trainer = Trainer(model, tokenizer, args)
    if 'train' in args.mode:
        # Data
        data_dir = Path(args.train_dir)
        print('Loading train and dev data', flush=True)
        train_dataset = CMRC2018Dataset(tokenizer, data_dir / 'train.json', 
            has_labels=True, tok_name=tok_name)
        eval_dataset = CMRC2018Dataset(tokenizer, data_dir / 'dev.json', 
            has_labels=True, tok_name=tok_name)
        trainer.train(train_dataset, eval_dataset)
    if 'test' in args.mode:
        data_dir = Path(args.test_dir)
        test_all(trainer, data_dir, output_dir, tok_name)


if __name__ == '__main__':
    # from multiprocessing import freeze_support
    # freeze_support()
    main()