from pathlib import Path
from argparse import Namespace

import torch
from torch.utils.data import DataLoader
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

from data.afqmc import AfqmcSeq2SeqDataset
import utils


def _get_dataset(file: Path, phase: str, **kwargs) -> AfqmcSeq2SeqDataset:
    return AfqmcSeq2SeqDataset(file, phase, **kwargs)


def get_dataset(data_dir: Path, phase: str, **kwargs) -> AfqmcSeq2SeqDataset:
    return _get_dataset(data_dir / f'{phase}.json', phase, **kwargs)


def predict(
    # trainer: Seq2SeqTrainer, 
    model, tokenizer,
    dataset: AfqmcSeq2SeqDataset, 
    args: Namespace,
    ) -> tuple:
    '''
    Return (predictions: list, result: dict)
    '''
    def get_test_acc(preds: torch.Tensor, labels: torch.Tensor) -> float:
        '''
        preds: (#examples)
        labels: (#examples)
        '''
        assert len(preds) == len(labels)
        count = len(labels)
        correct = 0
        for i in range(count):
            if preds[i] == labels[i]:
                correct += 1
        return correct / count

    def prediction_step(model, batch: dict, max_gen_len: int) -> tuple:
        # return trainer.prediction_step(trainer.model, inputs=batch, 
            # prediction_loss_only=False)
        output = model.generate(
            input_ids=batch['input_ids'].cuda(),
            attention_mask=batch['attention_mask'].cuda(),
            max_length=max_gen_len,
            do_sample=False,
        )
        return output

    text_to_label = {text: i for i, text in enumerate(dataset.verbalizer)}
    label_ids = tokenizer(dataset.verbalizer).input_ids
    max_gen_len = max([len(x) for x in label_ids])
    model.eval()
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        # collate_fn=collate_fn
    )
    
    print('*** Testing ***')
    print(f'# examples: {len(dataset)}')
    print(f'max_gen_len: {max_gen_len}')
    print(f'verbalizer: {dataset.verbalizer}')

    total_loss = 0
    acc = 0
    num_steps = 0
    all_preds = []
    for batch in dataloader:
        output_seqs = prediction_step(model, batch, max_gen_len)

        output_texts = tokenizer.batch_decode(output_seqs, skip_special_tokens=True)
        for t in output_texts:
            if t not in text_to_label:
                print(output_texts)
                print(text_to_label)
                exit()
        preds = [text_to_label[t] for t in output_texts]
        all_preds += preds

        # total_loss += loss.item()
        # print(batch)
        # label_ids = [x['label_ids'] for x in batch]
        acc += get_test_acc(preds, batch['label_ids'])
        num_steps += 1

    # Get result
    acc /= num_steps
    # loss = total_loss / num_steps
    result = {
        'acc': acc,
        # 'loss': loss,
    }
    return all_preds, result


def get_trainer(model, tokenizer, data_dir: Path, output_dir: Path, 
                args: Namespace) -> Seq2SeqTrainer:
    '''Return a huggingface Trainer instance.'''
    kwargs = {'tokenizer': tokenizer, 'num_examples': args.num_examples}
    train_dataset = get_dataset(data_dir, 'train', **kwargs)
    eval_dataset = get_dataset(data_dir, 'dev', **kwargs)
    print('verbalizer:', train_dataset.verbalizer)
    utils.dump_json(train_dataset.verbalizer, output_dir / 'verbalizer.json')
    print('# train examples:', len(train_dataset))
    print('# eval examples:', len(eval_dataset))

    # Hyperparameters
    batch_size = args.batch_size
    grad_acc_steps = args.grad_acc_steps
    num_epochs = args.num_epochs
    warmup_ratio = 0.1
    lr = args.lr
    
    train_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True, # TODO: remove on release
        do_train=True,
        do_predict=True,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_acc_steps,
        # Move predictions to CPU often because vocab is very large.
        eval_accumulation_steps=128,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        learning_rate=lr,
        num_train_epochs=num_epochs,
        lr_scheduler_type='linear',
        optim='adafactor',
        warmup_ratio=warmup_ratio,
        report_to='none',
        logging_first_step=True,
        logging_steps=args.log_interval,
        disable_tqdm=not args.tqdm,
        fp16=args.fp16,
        bf16=args.bf16,
        seed=args.seed,
    )
    trainer = Seq2SeqTrainer(
        model,
        train_args,
        train_dataset=train_dataset, 
        eval_dataset=eval_dataset,
    )
    return trainer