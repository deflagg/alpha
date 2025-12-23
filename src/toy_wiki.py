import random
from typing import List, Tuple, Dict
import numpy as np

class ToyWikiSpec:
    def __init__(self, num_entities: int, num_relations: int, aliases_per_entity: int, templates_per_relation: int):
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.aliases_per_entity = aliases_per_entity
        self.templates_per_relation = templates_per_relation

def build_aliases(spec: ToyWikiSpec, seed: int) -> Dict[int, List[str]]:
    rng = random.Random(seed)
    aliases = {}
    for i in range(spec.num_entities):
        canonical = f"Entity_{i:05d}"
        entity_aliases = [canonical]
        for j in range(spec.aliases_per_entity - 1):
            entity_aliases.append(f"Alias_{i:05d}_{j}")
        aliases[i] = entity_aliases
    return aliases

def build_relation_templates(spec: ToyWikiSpec, seed: int) -> Dict[int, List[str]]:
    rng = random.Random(seed)
    templates = {}
    for i in range(spec.num_relations):
        rel_name = f"Relation_{i:03d}"
        rel_templates = [
            f"{{s}} is connected to {{o}} via {rel_name}.",
            f"The {rel_name} of {{s}} is {{o}}.",
            f"{{s}} relates to {{o}} through {rel_name}.",
            f"{{o}} is the {rel_name} for {{s}}.",
            f"In the context of {rel_name}, {{s}} is paired with {{o}}."
        ][:spec.templates_per_relation]
        templates[i] = rel_templates
    return templates

def sample_triples(rng: random.Random, spec: ToyWikiSpec, k: int) -> List[Tuple[int, int, int]]:
    triples = []
    for _ in range(k):
        s = rng.randint(0, spec.num_entities - 1)
        r = rng.randint(0, spec.num_relations - 1)
        o = rng.randint(0, spec.num_entities - 1)
        while o == s:
            o = rng.randint(0, spec.num_entities - 1)
        triples.append((s, r, o))
    return triples

def render_paragraph(rng: random.Random, triples: List[Tuple[int, int, int]], aliases: Dict[int, List[str]], templates: Dict[int, List[str]]) -> str:
    sentences = []
    for s_idx, r_idx, o_idx in triples:
        s_name = rng.choice(aliases[s_idx])
        o_name = rng.choice(aliases[o_idx])
        template = rng.choice(templates[r_idx])
        sentences.append(template.format(s=s_name, o=o_name))
    
    # Shuffle sentences to make it slightly more "natural"
    rng.shuffle(sentences)
    return " ".join(sentences)

def canonical_entity_name(eid: int) -> str:
    return f"Entity_{eid:05d}"

def canonical_relation_template(rid: int) -> str:
    # Use the first template as the canonical one for evaluation
    return "{s} is connected to {o} via Relation_" + f"{rid:03d}."
