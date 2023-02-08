from pathlib import Path
from time import time

from trainer import Trainer
from .data import AfqmcDataset

from utils import dump_json, get_acc

class AfqmcTrainer(Trainer):
    def on_eval_end(self, 
        dataset: AfqmcDataset,
        desc: str='dev',
        output_dir: Path=None) -> dict:
        '''
        Get result, log and return.
        '''
        # Process gathered result
        output_dir.mkdir(exist_ok=True, parents=True)
        # TODO: remove on release
        dump_json(self.all_preds, 'preds.json')  
        dump_json(self.all_labels, 'labels.json')

        acc = get_acc(self.all_labels, self.all_preds)

        result = {
            'acc': acc,
            'loss': self.total_loss / self.num_eval_steps,   # This must be provided for choosing best model
            'time_elapsed': time() - self.eval_start_time,
        }
        self.log(result)
        return {
            'result': result,
            'preds': self.all_preds,
        }
