import os
import time
import argparse
import pickle


def str_to_bool(value):
    """
    Converts a string representation of a boolean value to its corresponding boolean value.

    Args:
        value (str): The string representation of the boolean value.

    Returns:
        bool: The boolean value corresponding to the input string.

    Raises:
        ValueError: If the input string is not a valid boolean value.
    """
    if isinstance(value, bool):
        return value
    if value.lower() in {"false", "f", "0", "no", "n"}:
        return False
    elif value.lower() in {"true", "t", "1", "yes", "y"}:
        return True
    raise ValueError(f"{value} is not a valid boolean value")


def get_public_config():
    """
    Get the public configuration parser.

    Returns:
        argparse.ArgumentParser: The argument parser object.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, help="train or test", default="train")
    parser.add_argument("--n_exp", type=int, default=0, help="experiment index")
    parser.add_argument("--gpu", type=int, default=6, help="which gpu to run")
    parser.add_argument("--seed", type=int, default=0)

    # data
    parser.add_argument("--dataset", type=str, default="base")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--aug", type=float, default=1.0)
    parser.add_argument("--seq_len", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--input_dim", type=int, default=20)  # 4+13+3=20
    parser.add_argument("--output_dim", type=int, default=1)

    # training
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--save_iter", type=int, default=0)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument(
        "--wandb", type=str_to_bool, default=True, help="whether to use wandb"
    )

    # test
    parser.add_argument("--save_preds", type=bool, default=False)
    return parser
