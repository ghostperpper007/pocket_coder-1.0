import torch
import torch.nn.functional as F
import random
import numpy as np
from model import (CodeEncoder, SSMDecoder, CodeGNN,
                   TokenAttention, ReasoningBlock,
                   ASTDiagnosticSystem, get_edge_index_sequential)

# ── Determinism ───────────────────────────────────────────────────────────────
def set_deterministic(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

DEVICE    = torch.device('cpu')
EMB_DIM   = 256
STATE_DIM = 32
MAX_SEQ_LEN = 512
CKPT_PATH = 'checkpoint (4).pt'

set_deterministic(42)

# ── Build model ───────────────────────────────────────────────────────────────
encoder    = CodeEncoder(EMB_DIM, MAX_SEQ_LEN).to(DEVICE)
decoder    = SSMDecoder(EMB_DIM, vocab_size=encoder.vocab_size, state_dim=STATE_DIM).to(DEVICE)
gnn        = CodeGNN(EMB_DIM).to(DEVICE)
token_attn = TokenAttention(EMB_DIM, num_heads=8).to(DEVICE)
reasoner   = ReasoningBlock(EMB_DIM).to(DEVICE)
diagnostic = ASTDiagnosticSystem(encoder, EMB_DIM).to(DEVICE)

ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
encoder.load_state_dict(ckpt['encoder'])
decoder.load_state_dict(ckpt['decoder'])
gnn.load_state_dict(ckpt['gnn'])
token_attn.load_state_dict(ckpt['token_attn'])
reasoner.load_state_dict(ckpt['reasoner'])
diagnostic.load_state_dict(ckpt['diagnostic'])

encoder.eval(); decoder.eval(); gnn.eval()
token_attn.eval(); reasoner.eval(); diagnostic.eval()

print("Model loaded.")

# ── Classical attention over state history (no learned params) ────────────────
def classical_context(history: list[torch.Tensor], query: torch.Tensor) -> torch.Tensor:
    """
    Pure dot-product attention over state history.
    No learned Q/K/V projections — just geometry in embedding space.

    Args:
        history : list of tensors each shape (1, EMB_DIM), oldest → newest
        query   : tensor shape (1, EMB_DIM), the current last hidden state

    Returns:
        context : tensor shape (1, EMB_DIM), weighted sum of history
    """
    if len(history) == 1:
        return history[0]

    # Stack history → (N, EMB_DIM)
    keys = torch.cat(history, dim=0)                        # (N, D)
    q    = F.normalize(query, dim=-1)                       # (1, D)
    k    = F.normalize(keys,  dim=-1)                       # (N, D)

    # Cosine similarity scores → softmax weights
    scores  = (q @ k.T).squeeze(0)                         # (N,)
    weights = F.softmax(scores, dim=-1)                     # (N,)

    # Weighted sum of raw (un-normalised) history vectors
    context = (weights.unsqueeze(-1) * keys).sum(dim=0, keepdim=True)  # (1, D)
    return context


# ── N-gram blocker (classical, no params) ─────────────────────────────────────
def get_banned_tokens(tokens: list[int], ngram_size: int = 3) -> set[int]:
    """
    Find every token that would complete a repeated n-gram.
    Returns a set of token ids to set to -inf before sampling.
    """
    banned = set()
    n = ngram_size - 1                  # prefix length to match
    if len(tokens) < n:
        return banned
    last_prefix = tuple(tokens[-n:])
    for i in range(len(tokens) - n):
        if tuple(tokens[i:i + n]) == last_prefix:
            banned.add(tokens[i + n])
    return banned


# ── Sampling (actually samples, unlike the old version) ───────────────────────
def sample_top_p(logits: torch.Tensor, top_p: float) -> int:
    """
    Nucleus (top-p) sampling. Truly samples from the filtered distribution.
    Falls back to argmax only when the entire mass collapses to one token.
    """
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    probs   = F.softmax(sorted_logits, dim=-1)
    cumsum  = torch.cumsum(probs, dim=-1)

    # Remove tokens beyond the nucleus
    remove                    = cumsum - probs > top_p
    remove[0]                 = False           # always keep the top token
    sorted_logits[remove]     = float('-inf')

    filtered_probs = F.softmax(sorted_logits, dim=-1)

    # Actual multinomial sample (not argmax)
    sampled_pos  = torch.multinomial(filtered_probs, num_samples=1).item()
    return sorted_indices[sampled_pos].item()


# ── Generation loop ───────────────────────────────────────────────────────────
def generate(
    prompt        : str,
    max_new_tokens: int   = 100,
    temperature   : float = 0.7,
    top_p         : float = 0.9,
    use_greedy    : bool  = False,
    rep_penalty   : float = 1.2,
    ngram_size    : int   = 3,
    max_history   : int   = 10,
) -> str:
    tokens = encoder.tokenizer.encode(prompt)

    # Single-vector states carried between steps — shape (1, EMB_DIM)
    high_h     = torch.zeros(1, EMB_DIM, device=DEVICE)
    low_h      = torch.zeros(1, EMB_DIM, device=DEVICE)
    feedback   = torch.zeros(1, EMB_DIM, device=DEVICE)
    delta_accum= torch.zeros(1, EMB_DIM, device=DEVICE)

    # Rolling history of last-token states — used by classical_context
    high_h_hist      = []
    low_h_hist       = []
    feedback_hist    = []
    delta_accum_hist = []

    with torch.no_grad():
        for _ in range(max_new_tokens):
            if len(tokens) >= MAX_SEQ_LEN:
                break

            # ── Embed current sequence ────────────────────────────────────────
            id_tensor  = torch.tensor(tokens, dtype=torch.long, device=DEVICE)
            positions  = torch.arange(len(id_tensor), device=DEVICE)
            embeddings = encoder.norm(
                encoder.token_emb(id_tensor) + encoder.pos_emb(positions)
            )                                                   # (T, D)

            seq_len    = embeddings.size(0)
            edge_index = get_edge_index_sequential(seq_len).to(DEVICE)
            edge_index = edge_index.clamp(max=seq_len - 1)

            # ── Expand single-vector states to full sequence length ───────────
            # We keep states as (1, D) between steps, expand only for the
            # forward pass, and collapse back to (1, D) afterwards.
            H = high_h.expand(seq_len, -1).contiguous()        # (T, D)
            L = low_h.expand(seq_len, -1).contiguous()         # (T, D)
            F_= feedback.expand(seq_len, -1).contiguous()      # (T, D)
            D = delta_accum.expand(seq_len, -1).contiguous()   # (T, D)

            # ── Reasoning passes ──────────────────────────────────────────────
            # Pass 1 — attend over raw embeddings
            features           = token_attn(embeddings)
            H, L, b_bias, c_bias, delta = reasoner(features, H, L, F_)
            gnn_out            = gnn(H, edge_index)
            D                  = (D * 0.8 + delta * 0.2).clamp(-1, 1)
            logits             = decoder(features, b_bias, c_bias)

            # Pass 2 — attend over GNN-enriched features
            delta_scale        = (gnn_out.norm() / (D.norm() + 1e-6)).clamp(max=1.0)
            refined            = gnn_out + D * delta_scale * 0.1
            H, L, b_bias, c_bias, delta = reasoner(refined, H, L, F_)
            D                  = (D * 0.8 + delta * 0.2).clamp(-1, 1)
            logits             = decoder(refined, b_bias, c_bias)

            feedback, _        = diagnostic.get_feedback(logits.detach())

            # ── Collapse back to last-token state ─────────────────────────────
            last_H    = H[-1:].clone()                          # (1, D)
            last_L    = L[-1:].clone()
            last_F    = feedback[-1:].clone()
            last_D    = D[-1:].clone()

            # ── Update history ────────────────────────────────────────────────
            high_h_hist.append(last_H)
            low_h_hist.append(last_L)
            feedback_hist.append(last_F)
            delta_accum_hist.append(last_D)

            if len(high_h_hist) > max_history:
                high_h_hist.pop(0)
                low_h_hist.pop(0)
                feedback_hist.pop(0)
                delta_accum_hist.pop(0)

            # ── Classical context: attend over full history ───────────────────
            high_h      = classical_context(high_h_hist,      last_H)
            low_h       = classical_context(low_h_hist,       last_L)
            feedback    = classical_context(feedback_hist,    last_F)
            delta_accum = classical_context(delta_accum_hist, last_D)

            # ── Token selection ───────────────────────────────────────────────
            last_logits = logits[-1].clone()                    # (vocab,)

            # Repetition penalty (multiplicative, proper form)
            for tid in set(tokens):
                if last_logits[tid] > 0:
                    last_logits[tid] /= rep_penalty
                else:
                    last_logits[tid] *= rep_penalty

            # N-gram blocking
            for tid in get_banned_tokens(tokens, ngram_size):
                last_logits[tid] = float('-inf')

            # Temperature
            last_logits = last_logits / temperature

            if use_greedy:
                next_token = last_logits.argmax().item()
            else:
                next_token = sample_top_p(last_logits, top_p)

            tokens.append(next_token)

            if next_token == encoder.tokenizer.eos_token_id:
                break

    return encoder.tokenizer.decode(tokens, skip_special_tokens=True)


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    prompt = "def calculate_sum(a, b):"
    print(f"\nPrompt: {prompt}\n")

    print("=== GREEDY (deterministic) ===")
    r1 = generate(prompt, max_new_tokens=50, temperature=0.1, use_greedy=True)
    r2 = generate(prompt, max_new_tokens=50, temperature=0.1, use_greedy=True)
    print(f"Run 1: {r1}")
    print(f"Run 2: {r2}")
    print(f"Identical: {r1 == r2}")

    print("\n=== SAMPLING ===")
    r3 = generate(prompt, max_new_tokens=50, temperature=0.7, top_p=0.9)
    print(f"Run 1: {r3}")
    r4 = generate(prompt, max_new_tokens=50, temperature=0.7, top_p=0.9)
    print(f"Run 2: {r4}")

    try:
        ckpt = torch.load(CKPT_PATH, map_location='cpu')
        print(f"\nCheckpoint — step: {ckpt['step']}, tokens: {ckpt.get('total_tokens', 0):,}")
    except Exception as e:
        print(f"Checkpoint info unavailable: {e}")