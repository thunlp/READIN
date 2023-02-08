'''
Train a BERT model on AFQMC data.

This uses the Trainer class of the Huggingface Transformers library. 

Some constant setups:
- Evaluate and save a checkpoints after each epoch.
- Use linear LR scheduler.
- Disable other loggings, and tqdm. 
'''
import torch
assert torch.cuda.is_available()

from pathlib import Path
from argparse import Namespace

from transformers import BertForSequenceClassification, BertTokenizer
from transformers.trainer import Trainer, TrainingArguments
import numpy as np

from afqmc.data import AfqmcDataset
import utils
from arguments import parse_args



def _get_dataset(file: Path, phase: str, **kwargs) -> AfqmcDataset:
    return AfqmcDataset(file, phase, max_seq_len=512, **kwargs)


def  get_dataset(data_dir: Path, phase: str, **kwargs) -> AfqmcDataset:
    return _get_dataset(data_dir / f'{phase}.json', phase, **kwargs)


def get_test_acc(preds: np.array, labels: np.array) -> float:
    return (np.argmax(preds, axis=1) == labels).mean()


def get_trainer(model: BertForSequenceClassification, tokenizer: BertTokenizer,
                data_dir: Path, output_dir: Path, args: Namespace) -> Trainer:
    kwargs = {'tokenizer': tokenizer, 'num_examples': args.num_examples}
    train_dataset = get_dataset(data_dir, 'train', **kwargs)
    eval_dataset = get_dataset(data_dir, 'dev', **kwargs)
    
    # Hyperparameters
    warmup_ratio = 0.1
    
    train_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True, # TODO: remove on release
        do_train=True,
        do_predict=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_acc_steps,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        learning_rate=args.lr,
        num_train_epochs=args.num_epochs,
        lr_scheduler_type='linear',
        warmup_ratio=warmup_ratio,
        logging_first_step=True,
        logging_steps=args.log_interval,
        logging_strategy='steps',
        report_to='none',
        disable_tqdm=True,
        seed=args.seed,
    )
    trainer = Trainer(
        model,
        train_args,
        train_dataset=train_dataset, 
        eval_dataset=eval_dataset,
    )
    return trainer


def predict(trainer: Trainer, dataset: AfqmcDataset, output_dir: Path):
    def parse_result(result):
        return {'loss': result.metrics['test_loss'],
                'acc': get_test_acc(result.predictions, result.label_ids)}

    result = trainer.predict(dataset)
    result_dict = parse_result(result)
    print('Test result:')
    print('loss:', result_dict['loss'])
    print('acc:', result_dict['acc'])

    # Save result
    output_dir.mkdir(exist_ok=True, parents=True)
    preds = np.argmax(result.predictions, axis=1) # (N, C) -> (N)
    utils.dump_str(list(preds), output_dir / 'preds.txt')
    utils.dump_json(result_dict, output_dir / 'result.json')


def train(trainer: Trainer, args: Namespace):
    train_args = {}
    if args.resume_from_ckpt:
        print(f"Resumption from: {args.resume_from_ckpt}")
        train_args['resume_from_checkpoint'] = args.resume_from_ckpt
    trainer.train(**train_args)


def test(trainer, tokenizer, data_dir):
    # Test clean
    print('\nTesting phase: clean', flush=True)
    file_examples = data_dir / 'test_clean'/ 'test.json'
    data = AfqmcDataset(file_examples, 'test_clean', tokenizer, 512)
    predict(trainer, data, output_dir / 'test_clean')
    
    # Test noisy
    for noise_type in [
        'keyboard', 
        'asr',
        ]:
        for i in range(1, 4):
            test_name = f'test_noisy_{noise_type}_{i}'
            print(f'\nTesting phase: {test_name}', flush=True)
            file_examples = data_dir / test_name / 'test.json'
            data = AfqmcDataset(file_examples, test_name, tokenizer, 512)
            predict(trainer, data, output_dir / test_name)

    # for noise_type in [
    #     'glyph_50',
    #     'glyph_100',
    #     'phonetic_0',
    #     'phonetic_10',
    #     'phonetic_20',
    #     'phonetic_30',
    #     'phonetic_40',
    #     'phonetic_50',
    #     ]:
    #     phase_name = f'test_synthetic_noise_{noise_type}'
    #     file_examples = Path('../data/synthetic_noise/afqmc_unbalanced', noise_type, 'test.json')
    #     data = AfqmcDataset(file_examples, phase_name, tokenizer, 512)
    #     predict(trainer, data, output_dir / phase_name)

    print('Done testing', flush=True)


if __name__ == "__main__":
    # Setup
    args = parse_args()
    output_dir = Path(args.output_dir)
    train_dir = Path(args.train_dir)
    test_dir = Path(args.test_dir)
    # os.environ["WANDB_DISABLED"] = "true"

    utils.set_seed(args.seed)
    utils.dump_args(args, output_dir / 'train_args.json')

    # Model
    print(f"Loading model from: {args.model_path}", flush=True)
    tokenizer = BertTokenizer.from_pretrained(args.model_path)
    model = BertForSequenceClassification.from_pretrained(args.model_path).cuda()
    print('# params:', utils.get_param_count(model), flush=True)

    # Train
    trainer = get_trainer(model, tokenizer, train_dir, output_dir, args)
    if 'train' in args.mode:
        train(trainer, args)
    if 'test' in args.mode:
        test(trainer, tokenizer, test_dir)
