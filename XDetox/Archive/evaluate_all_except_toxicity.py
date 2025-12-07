
from .bertscore import *
from .bleu import *
from .perplexity import *
import numpy as np
import argparse

def evaluate_all(references, hypotheses, eval_ref=True, use_corpus_bleu=True):
    # BERTScore (F1)
    bs = np.nanmean(get_bert_scores(zip(hypotheses, references))["f1"])  # note: (pred, ref)

    # BLEU-4
    if use_corpus_bleu:
        bleu = get_bleu([[r] for r in references], hypotheses)
    else:
        bleu = calc_bleu(references, hypotheses)

    # Perplexity
    perp_hyp = np.nanmean(get_perplexity(hypotheses))
    perp_ref = np.nanmean(get_perplexity(references)) if eval_ref else None

    return (bs, bleu, perp_hyp, perp_ref)

def get_data(args):
    with open(args.orig_path, "r") as f:
        orig = [s.strip() for s in f.readlines()]
    with open(args.gen_path, "r") as f:
        gen = [s.strip() for s in f.readlines()]
    if len(orig) != len(gen):
        raise ValueError(f"Line count mismatch: {len(orig)} refs vs {len(gen)} hyps")
    return orig, gen

def eval_args(args):
    orig, gen = get_data(args)
    metrics = evaluate_all(orig, gen, eval_ref=not args.skip_ref)

    save_path = args.gen_path[:-4] + "_stats_notox.txt"
    items = [
        "bertscore",
        "bleu4",
        "perplexity gen",
        "perplexity orig" if not args.skip_ref else "perplexity orig (skipped)",
    ]
    with open(save_path, "w") as f:
        for i, m in zip(items, metrics):
            print(i, ":", m)
            f.write(i + ": " + str(m) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--orig_path", required=True)
    parser.add_argument("--gen_path", required=True)
    parser.add_argument("--skip_ref", action="store_true", help="Skip computing metrics on references")
    eval_args(parser.parse_args())
