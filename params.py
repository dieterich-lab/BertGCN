import argparse


def check_positive(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("Values must be >= 1.")
    return ivalue


def parse_args(*args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--testonly",
        action="store_true",
        help="If true, only tests the model and skips training.",
    )
    parser.add_argument(
        "--interpret_mode",
        type=str,
        default="gcn_bert",
        choices=["gcn_only", "gcn_bert"],
    )
    parser.add_argument("--bertmodel", type=str, default="medbert", choices=["medbert"])
    parser.add_argument(
        "--doclevel",
        type=str,
        default="letter",
        choices=["letter", "diagnosis", "riskfactor", "anamnesis"],
    )
    parser.add_argument("--batchsize", default=1, type=int)
    parser.add_argument("--nepochs", default=50, type=int)
    parser.add_argument(
        "--mixfactor",
        default=0.5,
        type=float,
        help="Denotes how much percentage of the GCN are taken.",
    )
    parser.add_argument(
        "--testunklar",
        action="store_true",
        help="Uses the '_unklar labels as test set and rest as train set.'",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="If false, don't use stop words to filter the dataset for the BERT model during training.",
    )
    parser.add_argument(
        "--report", action="store_true", help="Report dataset statistics."
    )
    parser.add_argument("--dev", nargs="?", const=32, default=False, type=int)

    if args:
        args = parser.parse_args(*args)
    else:
        args = parser.parse_args()
    return args
