import argparse
import json
import os
try:
    from rouge_score import rouge_scorer
except ImportError:
    print("rouge_score package not found. Please install via: pip install rouge-score")
    exit(1)

def evaluate_rouge(generated_text, reference_text):
    """
    Computes ROUGE-1, ROUGE-2, and ROUGE-L scores between a generated text and a reference.
    """
    print("--- ROUGE Score Evaluation ---")
    print(f"Reference: '{reference_text}'")
    print(f"Generated: '{generated_text}'\n")

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference_text, generated_text)

    for metric_name, score in scores.items():
        print(f"{metric_name.upper()}:")
        print(f"  Precision: {score.precision:.4f}")
        print(f"  Recall:    {score.recall:.4f}")
        print(f"  F1 Score:  {score.fmeasure:.4f}")
    
    return scores

def main():
    parser = argparse.ArgumentParser(description="Evaluate text outputs with ROUGE")
    parser.add_argument("--gen", required=True, help="Generated text (model output)")
    parser.add_argument("--ref", required=True, help="Reference text (ground truth)")
    args = parser.parse_args()

    evaluate_rouge(args.gen, args.ref)

if __name__ == "__main__":
    main()
