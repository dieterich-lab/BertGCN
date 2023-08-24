import argparse


def check_positive(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("Values must be >= 1.")
    return ivalue


def parse_args(*args):
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str, choices=["MIC", "CSC", "Patho"], help="`MIC` = Medication indication classification, `CSC` = Cardio:DE section classification, `Patho` = Aurélie's data.")
    # parser.add_argument("model", type=str, choices=["gbert", "medbert"])
    parser.add_argument(
        "--testonly", action="store_true", help="If true, only tests the model and skips training."
    )
    parser.add_argument(
        "--noarznei", action="store_true", help="Trains the model without the sequence prefix `Arznei [...] &`."
    )
    parser.add_argument("--interpret_mode", type=str, default="gcn_bert", choices=["gcn_only", "gcn_bert"])
    parser.add_argument("--bertmodel", type=str, default="medbert", choices=["gbert", "medbert"])
    # parser.add_argument("--bertmodel", type=str, default=None, choices=["gbert", "medbert"])
    parser.add_argument("--doclevel", type=str, default="letter", choices=["letter", "diagnosis", "riskfactor", "anamnesis"])
    parser.add_argument("--attention_window", type=int, default=512, help="Attention window")
    parser.add_argument("--gpus", "-g", type=int, nargs="+", default=[0], choices=[0, 1, 2, 3])
    parser.add_argument("--batchsize", default=1, type=int)
    parser.add_argument("--nepochs", default=50, type=int)
    parser.add_argument("--mixfactor", default=.5, type=float, help="Denotes how much percentage of the GCN are taken.")
    parser.add_argument("--gradacc", default=32, type=int)
    parser.add_argument("--checkpoint", type=int, help="Load model from a specific checkpoint.")
    parser.add_argument("-a", "--accelerator", default="gpu", type=str, choices=["gpu", "cpu"])
    parser.add_argument("-s", "--simple", action="store_true", help="Uses simplified transformations for rnn and trans.")
    parser.add_argument("--testunklar", action="store_true", help="Uses the '_unklar labels as test set and rest as train set.'")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="If false, don't use stop words to filter the dataset for the BERT model during training.",
    )
    parser.add_argument("--suppressupdates", action="store_true", help="If true, updating graph features is suppressed.")
    parser.add_argument("--report", action="store_true", help="Report dataset statistics.")
    parser.add_argument("--optimize", action="store_true", help="Use optuna library to grid search the best mixfactor.")
    parser.add_argument("--reportraw", action="store_true", help="Report raw (untokenized) dataset statistics.")
    parser.add_argument("--savebest", action="store_true", help="Only save the current checkpoint as best model.")
    parser.add_argument("--equalsets", action="store_true", help="Forces train, val and test set to be the same.")
    parser.add_argument("--sample", action="store_true", help="Sample from the data when in 'dev' mode.")
    parser.add_argument(
        "--noweightedsampling",
        action="store_true",
        help="Don't use the custom Trainer that re-weights the cross entropy to compensate for skewed label distribution.",
    )
    parser.add_argument("-p", "--proceed", action="store_true", help="Continue training with current best model.")
    parser.add_argument("--allowtqdm", action="store_true", help="Show tqdm progressbar.")
    parser.add_argument(
        "--fromscratch", action="store_true", help="Trains the coding task with a freshly initialized model."
    )
    parser.add_argument(
        "--cdscentered",
        action="store_true",
        help="Trains the coding task on data that is centered around '-CDS_end-'-token.",
    )
    parser.add_argument("-o", "--overlap", default=50, type=int)
    parser.add_argument("-c", "--chunklen", default=200, type=int)
    parser.add_argument("-l", "--learningrate", default=5e-5, type=float)
    parser.add_argument(
        "-m", "--maxchunks", default=1000, type=check_positive, help="Determines the maximum number of chunks."
    )
    parser.add_argument("--nlayers", default=1, type=int)
    parser.add_argument("--nheads", default=4, type=int)
    parser.add_argument("--penfactor", default=0.5, type=float, help="Coefficient for the penalization term.")
    parser.add_argument("--dev", nargs="?", const=32, default=False, type=int)
    parser.add_argument("--patience", nargs="?", const=3, default=3, type=int)

    if args:
        args = parser.parse_args(*args)
    else:
        args = parser.parse_args()
    return args
