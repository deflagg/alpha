import argparse
import yaml
import torch
import random
import os
from tqdm import tqdm
from tokenizers import Tokenizer
from src.toy_wiki import ToyWikiSpec, build_aliases, build_relation_templates, sample_triples, render_paragraph, canonical_entity_name

def generate_dataset(config_path: str):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    ds_config = config['dataset']
    spec_config = ds_config['spec']
    gen_config = ds_config['generation']
    seq_config = ds_config['sequences']

    spec = ToyWikiSpec(
        num_entities=spec_config['num_entities'],
        num_relations=spec_config['num_relations'],
        aliases_per_entity=spec_config['aliases_per_entity'],
        templates_per_relation=spec_config['templates_per_relation']
    )

    seed = gen_config['dataset_seed']
    aliases = build_aliases(spec, seed)
    templates = build_relation_templates(spec, seed)
    
    tokenizer = Tokenizer.from_file(ds_config['paths']['tokenizer_path'])
    bos_id = tokenizer.token_to_id("[BOS]")
    eos_id = tokenizer.token_to_id("[EOS]")
    pad_id = tokenizer.token_to_id("[PAD]")
    sep_id = tokenizer.token_to_id("[SEP]")

    # Semantic vocab mapping:
    # 0: PAD, 1: BOS, 2: EOS, 3: SEP
    # 4 to 4 + num_entities - 1: Entities
    # 4 + num_entities to 4 + num_entities + num_relations - 1: Relations
    SEM_PAD, SEM_BOS, SEM_EOS, SEM_SEP = 0, 1, 2, 3
    ENTITY_OFFSET = 4
    RELATION_OFFSET = ENTITY_OFFSET + spec.num_entities

    def encode_semantic(triples):
        tokens = [SEM_BOS]
        for s, r, o in triples:
            tokens.extend([ENTITY_OFFSET + s, RELATION_OFFSET + r, ENTITY_OFFSET + o])
        tokens.append(SEM_SEP) # End of triples block
        tokens.append(SEM_EOS)
        return tokens

    def process_split(num_samples, split_seed):
        rng = random.Random(split_seed)
        text_ids_list = []
        sem_ids_list = []
        
        print(f"Generating {num_samples} samples...")
        for _ in tqdm(range(num_samples)):
            # Randomly sample 1-5 triples per sample to vary complexity
            k = rng.randint(1, 4)
            triples = sample_triples(rng, spec, k)
            
            # Text encoding
            text = render_paragraph(rng, triples, aliases, templates)
            enc = tokenizer.encode(text)
            t_ids = [bos_id] + enc.ids + [eos_id]
            if len(t_ids) > seq_config['seq_len_text']:
                t_ids = t_ids[:seq_config['seq_len_text']]
            else:
                t_ids = t_ids + [pad_id] * (seq_config['seq_len_text'] - len(t_ids))
            text_ids_list.append(t_ids)
            
            # Semantic encoding
            s_ids = encode_semantic(triples)
            if len(s_ids) > seq_config['seq_len_sem']:
                s_ids = s_ids[:seq_config['seq_len_sem']]
            else:
                s_ids = s_ids + [SEM_PAD] * (seq_config['seq_len_sem'] - len(s_ids))
            sem_ids_list.append(s_ids)
            
        return torch.LongTensor(text_ids_list), torch.LongTensor(sem_ids_list)

    train_text, train_sem = process_split(gen_config['train_samples'], seed + 1)
    val_text, val_sem = process_split(gen_config['val_samples'], seed + 2)

    # Build Evaluation Set (Relation Completion)
    eval_config = config['eval']['rel_completion']
    eval_rng = random.Random(eval_config['eval_seed'])
    eval_triples = sample_triples(eval_rng, spec, eval_config['num_questions'])
    
    e_s, e_r, e_o = [], [], []
    e_candidates = []
    
    for s, r, o in eval_triples:
        e_s.append(s)
        e_r.append(r)
        e_o.append(o)
        
        # Negative candidates
        negatives = []
        while len(negatives) < eval_config['num_choices'] - 1:
            neg = eval_rng.randint(0, spec.num_entities - 1)
            if neg != o and neg not in negatives:
                negatives.append(neg)
        
        choices = [o] + negatives
        eval_rng.shuffle(choices)
        e_candidates.append(choices)

    entity_names = [canonical_entity_name(i) for i in range(spec.num_entities)]
    relation_names = [f"Relation_{i:03d}" for i in range(spec.num_relations)]

    dataset = {
        "meta": {
            "spec": spec_config,
            "seeds": {"dataset": seed, "eval": eval_config['eval_seed']},
            "tokenizer_path": ds_config['paths']['tokenizer_path']
        },
        "entity_names": entity_names,
        "relation_names": relation_names,
        "train": {"text_ids": train_text, "sem_ids": train_sem},
        "val": {"text_ids": val_text, "sem_ids": val_sem},
        "eval_rel_completion": {
            "s": torch.LongTensor(e_s),
            "r": torch.LongTensor(e_r),
            "o": torch.LongTensor(e_o),
            "candidates": torch.LongTensor(e_candidates)
        }
    }

    os.makedirs(os.path.dirname(ds_config['paths']['dataset_path']), exist_ok=True)
    torch.save(dataset, ds_config['paths']['dataset_path'])
    print(f"Dataset saved to {ds_config['paths']['dataset_path']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    generate_dataset(args.config)
