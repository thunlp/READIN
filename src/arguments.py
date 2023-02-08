import argparse


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', required=True)
    # p.add_argument('--data_dir', required=True)
    p.add_argument('--train_dir', required=True)
    p.add_argument('--test_dir', required=True)
    p.add_argument('--output_dir', required=True)

    # Hyperparameters
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--num_epochs', type=int, default=10)
    p.add_argument('--lr', type=float, default=2e-5)
    p.add_argument('--grad_acc_steps', type=int, default=16)
    p.add_argument('--fp16', action='store_true',
                   help='Not working on Google\'s models')
    p.add_argument('--bf16', action='store_true',
                   help='Not working on Google\'s models')
    p.add_argument('--seed', type=int, default=0)

    # Saving/loading
    p.add_argument('--resume_from_ckpt')

    # Logging
    p.add_argument('--log_interval', type=int, default=5)
    p.add_argument('--tqdm', action='store_true')

    # Debugging
    p.add_argument('--num_examples', type=int)
    p.add_argument('--mode', default='train_test')

    return p.parse_args()
