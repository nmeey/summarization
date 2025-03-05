from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import math


def preprocess_for_pretrain(doc):
    # split into sentences
    sentences = [s for s in doc.split('.') if s]
    if len(sentences) <= 1:
        # if only one sentence, use first half as "summary" and second half as "doc"
        words = doc.split()
        half = len(words)//2 if len(words)//2 > 0 else len(words)
        summary_text = ' '.join(words[:half])
        doc_text = ' '.join(words[half:]) or ' '.join(words[:half])
    else:
        L = 3  # number of sentences to use as pseudo-summary
        summary_text = '.'.join(sentences[:L])
        doc_text = '.'.join(sentences[L:]) 
        if not doc_text:
            doc_text = summary_text  # edge case: if article has <=3 sentences, just duplicate as dummy
    return summary_text, doc_text

def pretrain_summarizer_epoch(docs, optimizer, tokenizer, model, batch_size=16):
    model.train()
    total_loss = 0.0
    random.shuffle(docs)
    for i in range(0, len(docs), batch_size):
        batch_docs = docs[i:i+batch_size]
        inputs = []
        targets = []
        for doc in batch_docs:
            pseudo_sum, remaining_doc = preprocess_for_pretrain(doc)
            # Tokenize
            enc = tokenizer(remaining_doc, truncation=True, max_length=512, return_tensors='pt')
            dec = tokenizer(pseudo_sum, truncation=True, max_length=128, return_tensors='pt')
            # Prepare decoder input: BART uses <s> as start of target and requires it explicitly
            # We can use model to do this by providing labels (Bart will internally shift and mask).
            input_ids = enc["input_ids"].squeeze(0)
            target_ids = dec["input_ids"].squeeze(0)
            inputs.append(input_ids)
            targets.append(target_ids)
        # Pad sequences in batch
        inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=PAD_TOKEN_ID)
        targets_padded = pad_sequence(targets, batch_first=True, padding_value=PAD_TOKEN_ID)
        inputs_padded = inputs_padded.to(device)
        targets_padded = targets_padded.to(device)
        # Forward pass (Bart allows passing labels to compute CE loss directly)
        outputs = model(input_ids=inputs_padded, labels=targets_padded)
        loss = outputs.loss  # CrossEntropyLoss over the padded sequence
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(batch_docs)
    return total_loss / len(docs)

# Example: run one pretraining epoch on a subset of data for demonstration
#print("Pretraining summarizer on pseudo summaries...")
#pretrain_loss = pretrain_summarizer_epoch(train_docs[:100], optimizer_sum, tokenizer, summarizer, batch_size=8)
#print(f"Pretraining loss: {pretrain_loss:.4f}")


def train_contrastive_and_reviewer_batch(docs_batch):
    """
    Train contrastive encoder and writing reviewer on a batch of documents.
    Each document in docs_batch is a raw text string.
    """
    # Generate one summary for each document using current summarizer (greedy for efficiency)
    tokenizer.padding_side = 'left'  # ensure padding on left if needed (Bart might expect decoder inputs differently)
    batch_enc = tokenizer(docs_batch, return_tensors='pt', padding=True, truncation=True, max_length=512)
    input_ids_batch = batch_enc['input_ids'].to(device)
    # Generate summaries (greedy)
    with torch.no_grad():
        # using summarizer generate function for speed; ensure model is not in training mode for generation
        summarizer.eval()
        summary_ids = summarizer.generate(input_ids_batch, max_length=128, num_beams=1, do_sample=False)
    summarizer.train()
    # Prepare contrastive loss
    contrastive_loss = 0.0
    # Prepare writing reviewer loss
    reviewer_loss = 0.0
    # We'll accumulate gradient manually for each item then average (for simplicity in explanation)
    optimizer_con.zero_grad()
    optimizer_wri.zero_grad()
    for idx, doc in enumerate(docs_batch):
        # True document and negatives
        pos_doc = doc
        negatives = generate_negatives(doc, n_negatives=3)  # generate 3 negatives for now
        # Tokenize positive and negatives
        docs_tokens = [tokenizer.encode(text, truncation=True, max_length=512) for text in [pos_doc] + negatives]
        # Add CLS token at beginning if not present (Bart encode already adds bos at start by default, but when using encode we should ensure it)
        # In this case, tokenizer.encode for Bart does include bos and eos by default. If not, we would prepend CLS_TOKEN_ID.
        # Convert to tensor and pad to same length
        lengths = [len(seq) for seq in docs_tokens]
        max_len = max(lengths)
        docs_pad = [seq + [PAD_TOKEN_ID]*(max_len-len(seq)) for seq in docs_tokens]
        docs_tensor = torch.tensor(docs_pad, device=device)
        # Encode summary and documents
        summary_tokens = summary_ids[idx:idx+1]  # the generated summary token IDs for this doc
        summary_tokens = summary_tokens.to(device)
        v_s = encode_sequence(summary_tokens, encoder_type="summary")  # (1, d_model_summary)
        # Encode each document candidate
        v_c_list = []
        # Note: encode_sequence expects 2D tensor (batch, seq_len), we can feed all candidates at once
        v_c = encode_sequence(docs_tensor, encoder_type="doc")  # (K, d_model_doc), where K = 1+negatives
        # Now split positive vs negatives
        v_c_pos = v_c[0:1]        # embedding of true document
        v_c_negs = v_c[1:]        # embeddings of negatives
        # Compute cosine similarities
        # Expand v_s to shape (K,) via broadcasting for cos sim
        v_s_expanded = v_s.expand(v_c.shape[0], -1)
        cos_sims = F.cosine_similarity(v_s_expanded, v_c, dim=-1)  # (K,)
        # Contrastive loss (InfoNCE): target is index 0 among K
        cos_sims_div = cos_sims / tau
        # We need to make it a 2D tensor for cross_entropy: shape (1, K)
        cos_sims_div = cos_sims_div.unsqueeze(0)
        target_index = torch.tensor([0], dtype=torch.long, device=device)
        item_con_loss = F.cross_entropy(cos_sims_div, target_index)
        contrastive_loss += item_con_loss
        
        # Writing reviewer training:
        # We need a positive human-written text s* and the negative summary \hat{s}.
        human_text = pos_doc  # for simplicity, use the source doc or first H sentences as human-written sample
        # Optionally, we could pick a random other document segment for variety.
        # Truncate human_text to H sentences (to simulate summary length somewhat)
        human_sentences = [s for s in human_text.split('.') if s]
        human_text_sample = '.'.join(human_sentences[:3]) if len(human_sentences) > 3 else human_text
        # Tokenize human and summary
        human_ids = tokenizer.encode(human_text_sample, truncation=True, max_length=128, add_special_tokens=True)
        summary_ids_list = summary_ids[idx].tolist()
        # Convert to tensors
        # Pad to same length
        max_len = max(len(human_ids), len(summary_ids_list))
        human_ids_pad = human_ids + [PAD_TOKEN_ID]*(max_len - len(human_ids))
        summary_ids_pad = summary_ids_list + [PAD_TOKEN_ID]*(max_len - len(summary_ids_list))
        human_tensor = torch.tensor(human_ids_pad, device=device).unsqueeze(0)
        summary_tensor = torch.tensor(summary_ids_pad, device=device).unsqueeze(0)
        # Reviewer forward
        _, human_score = writing_reviewer(human_tensor)       # mean score for human text (positive)
        _, summary_score = writing_reviewer(summary_tensor)   # mean score for generated summary (negative)
        # Compute reviewer WGAN loss: we want reviewer to assign higher to human, lower to summary
        # So the loss (to minimize) = (score_fake - score_real) + gradient_penalty
        # Actually, since we want to maximize f_wri(s*) - f_wri(s^), minimizing (f_wri(s^) - f_wri(s*)) achieves that.
        item_wri_loss = (summary_score - human_score).mean()
        # Gradient penalty:
        # Interpolate between human and summary embedding representations
        # For interpolation, we'll do in the embedding space of reviewer (we can interpolate the input token embeddings).
        epsilon = random.random()
        # Get embeddings for both sequences from reviewer
        emb_human = writing_reviewer.module.embed(human_tensor) if isinstance(writing_reviewer, nn.DataParallel) else writing_reviewer.embed(human_tensor)
        emb_summary = writing_reviewer.module.embed(summary_tensor) if isinstance(writing_reviewer, nn.DataParallel) else writing_reviewer.embed(summary_tensor)
        # Interpolate embeddings
        interpolated = epsilon * emb_human + (1 - epsilon) * emb_summary
        interpolated.requires_grad_(True)
        # Pass through LSTM and linear to get score
        out, _ = (writing_reviewer.module.lstm(interpolated) if isinstance(writing_reviewer, nn.DataParallel) 
                  else writing_reviewer.lstm(interpolated))
        interp_scores = (writing_reviewer.module.linear(out) if isinstance(writing_reviewer, nn.DataParallel) 
                         else writing_reviewer.linear(out)).squeeze(-1)
        # Take mean score of interpolated
        interp_mean = interp_scores.mean()
        # Compute gradient of this score w.rt interpolated embeddings
        grad = torch.autograd.grad(outputs=interp_mean, inputs=interpolated,
                                   grad_outputs=torch.ones_like(interp_mean),
                                   create_graph=True, retain_graph=True)[0]
        grad_norm = grad.norm(2)
        gp_loss = lambda_gp * ((grad_norm - 1)**2)
        item_wri_loss = item_wri_loss + gp_loss
        reviewer_loss += item_wri_loss
    # Average losses over batch
    contrastive_loss = contrastive_loss / len(docs_batch)
    reviewer_loss = reviewer_loss / len(docs_batch)
    # Backpropagate and update encoders and reviewer
    contrastive_loss.backward()
    reviewer_loss.backward()
    optimizer_con.step()
    optimizer_wri.step()
    return contrastive_loss.item(), reviewer_loss.item()

# Pretrain contrastive encoder & reviewer on a small sample (for demonstration)
#sample_docs = train_docs[:4]
#c_loss, w_loss = train_contrastive_and_reviewer_batch(sample_docs)
#print(f"Contrastive loss (pretrain): {c_loss:.4f}, Reviewer loss (pretrain): {w_loss:.4f}")




def train_summarizer_with_rl(docs_batch):
    summarizer.train()
    optimizer_sum.zero_grad()
    batch_size = len(docs_batch)
    # Tokenize inputs
    enc = tokenizer(docs_batch, return_tensors='pt', padding=True, truncation=True, max_length=512)
    input_ids = enc['input_ids'].to(device)
    attention_mask = enc['attention_mask'].to(device)
    # Generate one sampled summary and one greedy summary for each input
    sampled_summaries = []
    sampled_logprobs = []
    greedy_summaries = []
    # We will do auto-regressive generation manually to also get log-probs for the sampled sequence.
    # Initialize decoder input with BOS token for each sequence
    dec_input = torch.tensor([[CLS_TOKEN_ID]] * batch_size, device=device)  # shape (batch, 1)
    finished = [False] * batch_size
    max_len = 128
    # To store log-probs
    log_probs = [0.0] * batch_size
    for t in range(max_len):
        # Get logits for next token from summarizer
        outputs = summarizer(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=dec_input)
        next_token_logits = outputs.logits[:, -1, :]  # (batch, vocab_size) logits for last time step
        # Sample a token for each sequence
        probs = F.softmax(next_token_logits, dim=-1)
        # For reproducibility, one might set a random seed here or use distribution with temperature if needed.
        m = torch.distributions.Categorical(probs)
        sampled_tokens = m.sample()  # (batch,)
        token_logprobs = m.log_prob(sampled_tokens)  # (batch,)
        # Append sampled tokens to decoder input for next step
        dec_input = torch.cat([dec_input, sampled_tokens.unsqueeze(1)], dim=1)
        # Accumulate log-probabilities
        for i, lp in enumerate(token_logprobs):
            if not finished[i]:
                log_probs[i] += lp  # add log-prob for this token
        # Check for EOS token
        for i, tok in enumerate(sampled_tokens.tolist()):
            if tok == EOS_TOKEN_ID:
                finished[i] = True
        if all(finished):
            break
    # Now dec_input contains BOS + generated tokens for each sequence.
    sampled_summaries = dec_input  # includes BOS at index 0 and maybe EOS within
    sampled_logprobs = torch.stack([lp for lp in log_probs])  # shape (batch,)
    # Also get greedy summaries for baseline (we can use model.generate for that)
    with torch.no_grad():
        greedy_ids = summarizer.generate(input_ids, max_length=128, num_beams=1, do_sample=False)
    greedy_summaries = greedy_ids  # token IDs

    # Compute rewards for each sequence
    rewards = []
    for i in range(batch_size):
        # Get contrastive reward
        # Prepare true doc and negatives
        doc_text = docs_batch[i]
        negatives = generate_negatives(doc_text, n_negatives=3)
        # Tokenize doc and negatives
        docs_tokens = [tokenizer.encode(text, truncation=True, max_length=512) for text in [doc_text] + negatives]
        max_doc_len = max(len(seq) for seq in docs_tokens)
        docs_pad = [seq + [PAD_TOKEN_ID]*(max_doc_len-len(seq)) for seq in docs_tokens]
        docs_tensor = torch.tensor(docs_pad, device=device)
        # Encode summary (sampled and greedy) and docs
        # Remove the BOS token from the summary encoding (our encode_sequence assumes BOS at pos0, our sampled_summaries already include BOS at start)
        summary_ids = sampled_summaries[i:i+1]
        greedy_ids = greedy_summaries[i:i+1]
        v_s_sample = encode_sequence(summary_ids, encoder_type="summary")
        v_s_greedy = encode_sequence(greedy_ids, encoder_type="summary")
        v_c_all = encode_sequence(docs_tensor, encoder_type="doc")
        v_c_pos = v_c_all[0:1]
        v_c_negs = v_c_all[1:]
        # Compute contrastive loss for sample and greedy
        # (We reuse similar code as before, but now we want the numerical value of loss.)
        def contrastive_loss_for(v_s):
            # cosine similarities
            v_s_expand = v_s.expand(v_c_all.size(0), -1)
            cos_sims = F.cosine_similarity(v_s_expand, v_c_all, dim=-1)
            sims = cos_sims / tau
            sims = sims.unsqueeze(0)
            # cross entropy loss with target 0
            loss_val = F.cross_entropy(sims, torch.tensor([0], device=device))
            return loss_val.item()
        loss_sample = contrastive_loss_for(v_s_sample)
        loss_greedy = contrastive_loss_for(v_s_greedy)
        r_con = -loss_sample  # negative loss as reward
        baseline_con = -loss_greedy
        # Contrastive advantage
        adv_con = r_con - baseline_con
        # Writing quality reward
        # Use writing reviewer to score both
        _, mean_score_sample = writing_reviewer(sampled_summaries[i:i+1])
        _, mean_score_greedy = writing_reviewer(greedy_summaries[i:i+1])
        r_wri = mean_score_sample.item()
        # (Optionally subtract baseline: difference between sample and greedy)
        adv_wri = r_wri  # - mean_score_greedy.item() (we can subtract if desired)
        # Total reward advantage
        advantage = adv_con + adv_wri
        rewards.append(advantage)
    rewards = torch.tensor(rewards, device=device)
    # Policy gradient loss = - E[ (r - baseline) * log_prob ]
    policy_loss = - (rewards * sampled_logprobs).mean()
    policy_loss.backward()
    optimizer_sum.step()
    return policy_loss.item()

# Run an RL training step on a small batch (for demo)
#sample_docs_batch = train_docs[:2]
#rl_loss = train_summarizer_with_rl(sample_docs_batch)
#print(f"Summarizer policy loss: {rl_loss:.4f}")

if __name__ == "__main__":
  # Optimizers
  lr = 1e-4
  optimizer_sum = torch.optim.Adam(summarizer.parameters(), lr=lr)
  optimizer_con = torch.optim.Adam(contrastive_params, lr=lr)
  optimizer_wri = torch.optim.Adam(writing_reviewer.parameters(), lr=lr)
  
  # Hyperparameters
  tau = 1.0
  lambda_gp = 10.0
  alpha_baseline = 1.0
  batch_size = 16
  num_epochs = 4
  for epoch in range(num_epochs):
    for batch in training_data:
        # Train critics
        train_contrastive_and_reviewer_batch(batch)
        # Train summarizer
        train_summarizer_with_rl(batch)

