import torch
import torch.nn.functional as F
from src.toy_wiki import canonical_relation_template

def evaluate_lm_loss(model, val_tensor, pad_id, batch_size=32):
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for i in range(0, len(val_tensor), batch_size):
            batch = val_tensor[i:i+batch_size].to(next(model.parameters()).device)
            # batch: [B, T]
            logits = model(batch)
            
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = batch[:, 1:].contiguous()
            
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), 
                                   shift_labels.view(-1), 
                                   ignore_index=pad_id, 
                                   reduction='sum')
            
            total_loss += loss.item()
            num_tokens = (shift_labels != pad_id).sum().item()
            total_tokens += num_tokens
            
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
    return avg_loss

def evaluate_rel_completion_semantic(model, eval_set, entity_names, relation_names, device):
    """
    eval_set: s, r, o, candidates
    semantic tokens: BOS, S, R, O
    """
    model.eval()
    s = eval_set['s'].to(device)
    r = eval_set['r'].to(device)
    o = eval_set['o'].to(device)
    candidates = eval_set['candidates'].to(device)
    
    ENTITY_OFFSET = 4
    RELATION_OFFSET = ENTITY_OFFSET + len(entity_names)
    SEM_BOS = 1
    
    correct = 0
    
    with torch.no_grad():
        # Build prompt: [BOS, ent(S), rel(R)]
        prompt = torch.stack([
            torch.full_like(s, SEM_BOS),
            s + ENTITY_OFFSET,
            r + RELATION_OFFSET
        ], dim=1) # [N, 3]
        
        logits = model(prompt) # [N, 3, V]
        last_logits = logits[:, -1, :] # Prediction for token after R (which should be O)
        
        for i in range(len(s)):
            cand_ids = candidates[i] + ENTITY_OFFSET
            cand_logits = last_logits[i, cand_ids]
            pred_idx = cand_ids[cand_logits.argmax()]
            if pred_idx == o[i] + ENTITY_OFFSET:
                correct += 1
                
    return correct / len(s)

def evaluate_rel_completion_traditional(model, tokenizer, eval_set, entity_names, relation_names, device):
    model.eval()
    s = eval_set['s']
    r = eval_set['r']
    o = eval_set['o']
    candidates = eval_set['candidates']
    
    correct = 0
    bos_id = tokenizer.token_to_id("[BOS]")
    pad_id = tokenizer.token_to_id("[PAD]")
    
    with torch.no_grad():
        for i in range(len(s)):
            s_name = entity_names[s[i]]
            # Use deterministic template
            template = canonical_relation_template(r[i].item())
            prompt_text = template.split("{o}")[0].format(s=s_name)
            
            # Encode prompt
            prompt_ids = [bos_id] + tokenizer.encode(prompt_text).ids
            prompt_tensor = torch.LongTensor([prompt_ids]).to(device)
            
            cand_scores = []
            for cand_idx in candidates[i]:
                cand_name = entity_names[cand_idx]
                # Log-prob of candidate name given prompt
                cand_ids = tokenizer.encode(cand_name).ids
                
                # Full sequence: prompt + candidate
                full_ids = prompt_ids + cand_ids
                full_tensor = torch.LongTensor([full_ids]).to(device)
                
                logits = model(full_tensor) # [1, T, V]
                # We care about the log-probs of the candidate tokens
                # prompt ends at index len(prompt_ids) - 1
                # first cand token is at index len(prompt_ids)
                # so we look at logits from len(prompt_ids)-1 to end-1
                relevant_logits = logits[0, len(prompt_ids)-1:-1, :]
                relevant_targets = full_tensor[0, len(prompt_ids):]
                
                log_probs = F.log_softmax(relevant_logits, dim=-1)
                cand_tokens_log_prob = log_probs[torch.arange(len(relevant_targets)), relevant_targets].sum().item()
                cand_scores.append(cand_tokens_log_prob)
                
            if candidates[i][torch.argmax(torch.tensor(cand_scores))] == o[i]:
                correct += 1
                
    return correct / len(s)
