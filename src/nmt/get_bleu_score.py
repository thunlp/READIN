import json
from pathlib import Path
import logging

from sacrebleu import corpus_bleu, sentence_bleu

logging.getLogger('sacrebleu').setLevel(logging.ERROR)


def avg(x): return sum(x) / len(x)


def load_jsonl(file):
    return [json.loads(line) for line in open(file, 'r')]


def get_noisy_ids(file):
    return [int(json.loads(line)['id']) for line in open(file, 'r')]


def get_result(output_dir, noise_type, en_lines, num=None):
    # Clean
    print('---- Clean ----')
    hyps = load_jsonl(output_dir / 'test_clean' / 'preds.json')
    refs = [en_lines]  # Yes, it is a list of list(s) as required by sacreBLEU

    bleu = corpus_bleu(hyps, refs)
    print(bleu.precisions, bleu.counts, bleu.totals)
    print(bleu)
    # print(vars(bleu))
    # import math
    # bp = bleu.bp
    # score = bp * math.exp(
    #         sum([math.log(p) for p in bleu.precisions[:4]]) / 4)
    # print(score)
    # exit()
    
    scores = []
    all_scores = []  # List of list of sentence BLEU scores
    ids = get_noisy_ids(f'../../data/realtypo/nmt/test_noisy_{noise_type}_1.json')
    en_lines = [en_lines[i] for i in ids]
    for i in range(1, 4):
        test_name = f'test_noisy_{noise_type}_{i}'
        print(f'---- {test_name} ----')
        test_dir = output_dir / test_name
        hyps = load_jsonl(test_dir / 'preds.json')[:num]
        refs = [en_lines[:num]]  # Yes, it is a list of list(s) as required by sacreBLEU

        bleu = corpus_bleu(hyps, refs)
        score = bleu.score
        scores.append(score)
        print(bleu)
        print(bleu.precisions, bleu.counts, bleu.totals)
        print('Micro average')
        sent_scores = []
        for hyp, ref in zip(hyps, refs[0]):
            bleu = sentence_bleu(hyp, [ref], use_effective_order=False)
            # bleu = corpus_bleu([hyp], [[ref]])
            sent_scores.append(bleu.score)
        print(f'BLEU: {avg(sent_scores)}')
        
        all_scores.append(sent_scores)
        json.dump(sent_scores, open(test_dir / 'bleu_sent.json', 'w'), indent=2)

    print('---- Average ----')
    print(sum(scores) / 3)
    print('---- Worst group BLEU ----')
    worst_scores = []
    for sent_scores in zip(*all_scores):
        worst_scores.append(min(sent_scores))
    print(sum(worst_scores) / len(worst_scores))


if __name__ == '__main__':
    example_file = '../../data/realtypo/nmt/test_clean.json'
    en_lines = [x['en'] for x in load_jsonl(example_file)]

    result_dir = Path('../results/nmt')
    model_name = 'm2m100_418M'
    output_dir = result_dir / model_name
    noise_type = 'asr'
    get_result(output_dir, noise_type, en_lines)



