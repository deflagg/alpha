import argparse
import os
import random
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from src.toy_wiki import ToyWikiSpec, build_aliases, build_relation_templates, sample_triples, render_paragraph

def train_tokenizer(out_path: str, vocab_size: int, num_samples: int, seed: int):
    spec = ToyWikiSpec(num_entities=1000, num_relations=50, aliases_per_entity=3, templates_per_relation=5)
    aliases = build_aliases(spec, seed)
    templates = build_relation_templates(spec, seed)
    rng = random.Random(seed)

    print(f"Generating {num_samples} samples for tokenizer training...")
    texts = []
    for _ in range(num_samples):
        # Sample 1-3 triples per paragraph
        k = rng.randint(1, 3)
        triples = sample_triples(rng, spec, k)
        texts.append(render_paragraph(rng, triples, aliases, templates))

    print("Training BPE tokenizer...")
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[PAD]", "[BOS]", "[EOS]", "[UNK]", "[SEP]"]
    )
    
    tokenizer.train_from_iterator(texts, trainer)
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tokenizer.save(out_path)
    print(f"Tokenizer saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="artifacts/bpe32k.json")
    parser.add_argument("--vocab_size", type=int, default=32000)
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    train_tokenizer(args.out, args.vocab_size, args.samples, args.seed)
