"""
preprocess.py  —  run this ONCE before training
Saves: /kaggle/working/dataset.pt
  A list of dicts: {
      "ids"        : LongTensor (seq_len,)
      "ast_signal" : FloatTensor (8,)
      "edge_index" : LongTensor (2, num_edges)
  }
"""

import ast
import math
import torch
import itertools
from transformers import AutoTokenizer
from datasets import load_dataset

# ── config (must match training script) ──────────────────────────────────────
MAX_SEQ_LEN  = 512
MIN_TOKENS   = 2
SAVE_PATH    = "/kaggle/working/dataset.pt"
MAX_SAMPLES  = 500_000   # set to None to use everything
# ─────────────────────────────────────────────────────────────────────────────

tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono")

# ── AST helpers ───────────────────────────────────────────────────────────────

def _ast_depth(node, depth=0):
    children = list(ast.iter_child_nodes(node))
    if not children:
        return depth
    return max(_ast_depth(c, depth + 1) for c in children)

def _token_entropy(ids):
    if len(ids) == 0:
        return 0.0
    counts      = torch.bincount(torch.tensor(ids, dtype=torch.long))
    probs       = counts.float() / counts.sum()
    probs       = probs[probs > 0]
    entropy     = -(probs * probs.log()).sum().item()
    max_entropy = math.log(len(ids) + 1e-9)
    return entropy / max_entropy if max_entropy > 0 else 0.0

def compute_ast_signal(code_str: str, ids: list) -> torch.Tensor:
    syntax_ok = error_position = depth_score = 0.0
    node_diversity = return_present = func_def_present = undefined_ratio = 0.0

    try:
        tree             = ast.parse(code_str)
        syntax_ok        = 1.0
        all_nodes        = list(ast.walk(tree))
        node_types       = [type(n).__name__ for n in all_nodes]
        total_nodes      = max(len(all_nodes), 1)
        raw_depth        = _ast_depth(tree)
        depth_score      = min(raw_depth / 20.0, 1.0)
        node_diversity   = len(set(node_types)) / total_nodes
        return_present   = 1.0 if any(isinstance(n, ast.Return)      for n in all_nodes) else 0.0
        func_def_present = 1.0 if any(isinstance(n, ast.FunctionDef) for n in all_nodes) else 0.0
        assigned         = {n.id for n in ast.walk(tree)
                            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
        used             = {n.id for n in ast.walk(tree)
                            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        undefined        = used - assigned - {
            'True','False','None','print','range','len','int','str',
            'float','list','dict','set','tuple','type','zip','map','enumerate'
        }
        undefined_ratio  = len(undefined) / max(len(used), 1)
    except SyntaxError as e:
        total_lines    = max(len(code_str.splitlines()), 1)
        error_line     = getattr(e, 'lineno', 0) or 0
        error_position = min(error_line / total_lines, 1.0)

    token_ent = _token_entropy(ids)

    return torch.tensor([
        syntax_ok, error_position, depth_score, node_diversity,
        return_present, func_def_present, undefined_ratio, token_ent,
    ], dtype=torch.float)

def compute_edge_index(code_str: str, n_tokens: int) -> torch.Tensor:
    try:
        tree     = ast.parse(code_str)
        edges    = set()
        nodes    = list(ast.walk(tree))
        node_ids = {id(n): i for i, n in enumerate(nodes)}
        for node in nodes:
            for child in ast.iter_child_nodes(node):
                p = node_ids[id(node)]
                c = node_ids[id(child)]
                if p != c:
                    edges.add((p, c))
                    edges.add((c, p))
        if edges:
            row, col = zip(*edges)
            ei = torch.tensor([list(row), list(col)], dtype=torch.long)
            return ei.clamp(max=n_tokens - 1)
    except:
        pass
    n   = n_tokens
    row = list(range(n - 1)) + list(range(1, n))
    col = list(range(1, n)) + list(range(n - 1))
    return torch.tensor([row, col], dtype=torch.long)

# ── dataset ───────────────────────────────────────────────────────────────────

def get_dataset():
    codesearchnet = load_dataset("Nan-Do/code-search-net-python",    streaming=True, split="train")
    codeparrot    = load_dataset("codeparrot/codeparrot-clean-train", streaming=True, split="train")
    return itertools.chain(codesearchnet, codeparrot)

def get_code(sample) -> str:
    code = sample.get("whole_func_string", "") or sample.get("content", "")
    first_line = next((l.strip() for l in code.splitlines() if l.strip()), "")
    if not any(first_line.startswith(kw) for kw in
               ("def ", "class ", "import ", "from ", "#", "@")):
        return ""
    return code

# ── main ──────────────────────────────────────────────────────────────────────

print("Starting preprocessing...")
records = []
skipped = 0

for i, sample in enumerate(get_dataset()):
    if MAX_SAMPLES and i >= MAX_SAMPLES:
        break

    code = get_code(sample)
    if len(code.strip()) < 10:
        skipped += 1
        continue

    ids = tokenizer.encode(code, truncation=True, max_length=MAX_SEQ_LEN)
    if len(ids) < MIN_TOKENS:
        skipped += 1
        continue

    records.append({
        "ids"        : torch.tensor(ids, dtype=torch.long),
        "ast_signal" : compute_ast_signal(code, ids),
        "edge_index" : compute_edge_index(code, len(ids)),
    })

    if len(records) % 10_000 == 0:
        print(f"  {len(records):,} samples processed (skipped {skipped:,})...")

torch.save(records, SAVE_PATH)
print(f"\nDone. Saved {len(records):,} samples → {SAVE_PATH}")
print(f"Skipped {skipped:,} invalid samples.")


import torch
import torch.nn as nn
import torch.nn.functional as F
import ast
import math
import time
import os
import random
from dataclasses import dataclass
from typing import Tuple
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

EMB_DIM      = 256
STATE_DIM    = 32
MAX_SEQ_LEN  = 512
LR           = 3e-4
GRAD_CLIP    = 1.0
CKPT_EVERY   = 300
CKPT_PATH    = "/kaggle/working/checkpoint.pt"
LOAD_PATH    = "/kaggle/input/models/arjimbob/checkpoint7/pytorch/default/1/checkpoint (7).pt"
LOG_EVERY    = 50
DATASET_PATH = "/kaggle/working/dataset.pt"  # output of preprocess.py

WARMUP_STEPS = 500
TOTAL_STEPS  = 50000
ACCUM_STEPS  = 8

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ─────────────────────────────────────────────────────────────────────────────
# MODEL  (completely unchanged)
# ─────────────────────────────────────────────────────────────────────────────

class CodeEncoder(nn.Module):
    def __init__(self, embedding_dim=EMB_DIM, max_seq_len=MAX_SEQ_LEN):
        super().__init__()
        self.tokenizer   = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono")
        self.vocab_size  = len(self.tokenizer)
        self.max_seq_len = max_seq_len
        self.token_emb   = nn.Embedding(self.vocab_size, embedding_dim)
        self.pos_emb     = nn.Embedding(max_seq_len, embedding_dim)
        self.norm        = nn.LayerNorm(embedding_dim)

    def encode_ids(self, id_tensor: torch.Tensor):
        """Fast path: pre-tokenized ids, skips string processing."""
        device    = next(self.parameters()).device
        id_tensor = id_tensor.to(device)
        positions = torch.arange(len(id_tensor), device=device)
        x = self.token_emb(id_tensor) + self.pos_emb(positions)
        return self.norm(x), id_tensor

    def encode(self, source: str):
        """Original string path — kept for inference."""
        ids       = self.tokenizer.encode(source, truncation=True, max_length=self.max_seq_len)
        device    = next(self.parameters()).device
        id_tensor = torch.tensor(ids, dtype=torch.long, device=device)
        positions = torch.arange(len(id_tensor), device=device)
        x = self.token_emb(id_tensor) + self.pos_emb(positions)
        return self.norm(x), id_tensor

    def decode(self, ids: list) -> str:
        ids = [max(0, min(i, self.vocab_size - 1)) for i in ids]
        return self.tokenizer.decode(ids, skip_special_tokens=True)


class TokenAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads=8):
        super().__init__()
        self.attn  = nn.MultiheadAttention(embedding_dim, num_heads, batch_first=True, dropout=0.1)
        self.norm  = nn.LayerNorm(embedding_dim)
        self.ff    = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim * 4, embedding_dim),
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        x_seq       = x.unsqueeze(0)
        attn_out, _ = self.attn(x_seq, x_seq, x_seq)
        x_seq       = self.norm(attn_out + x_seq)
        x_seq       = self.norm2(self.ff(x_seq) + x_seq)
        return x_seq.squeeze(0)


class CodeGNN(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'w_self':  nn.Linear(embedding_dim, embedding_dim),
                'w_neigh': nn.Linear(embedding_dim, embedding_dim),
                'attn':    nn.Linear(embedding_dim * 2, 1),
            }) for _ in range(2)
        ])
        self.norms      = nn.ModuleList([nn.LayerNorm(embedding_dim) for _ in range(2)])
        self.activation = nn.GELU()

    def forward(self, x, edge_index):
        num_nodes = x.size(0)
        row, col  = edge_index
        for layer, norm in zip(self.layers, self.norms):
            out_self   = layer['w_self'](x)
            neigh_feat = layer['w_neigh'](x[row])
            attn_score = torch.sigmoid(layer['attn'](torch.cat([x[col], neigh_feat], dim=-1)))
            weighted   = neigh_feat * attn_score
            agg        = torch.zeros_like(out_self)
            agg.index_add_(0, col, weighted)
            attn_sum   = torch.zeros(num_nodes, 1, device=x.device, dtype=x.dtype)
            attn_sum.index_add_(0, col, attn_score.to(attn_sum.dtype))
            x = norm(self.activation(out_self + agg / (attn_sum + 1e-6)) + x)
        return x


class ReasoningBlock(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.high_a_down   = nn.Linear(embedding_dim, 64)
        self.high_a_up     = nn.Linear(64, embedding_dim)
        self.gate          = nn.Linear(embedding_dim * 2, embedding_dim)
        self.low_a         = nn.Parameter(torch.randn(embedding_dim, embedding_dim) * 0.02)
        self.b_injector    = nn.Linear(embedding_dim, embedding_dim)
        self.c_injector    = nn.Linear(embedding_dim, embedding_dim)
        self.delta_proj    = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x, high_hidden, low_hidden, diagnostic_feedback):
        new_high_hidden = torch.tanh(self.high_a_up(self.high_a_down(x)))
        gate_val        = torch.sigmoid(self.gate(torch.cat([high_hidden, diagnostic_feedback], dim=-1)))
        refinement      = torch.matmul(low_hidden, self.low_a) * gate_val
        new_low_hidden  = torch.tanh(refinement + diagnostic_feedback)
        b_inject        = self.b_injector(new_low_hidden)
        c_inject        = self.c_injector(new_high_hidden)
        delta           = torch.tanh(self.delta_proj(new_low_hidden))
        return new_high_hidden, new_low_hidden, b_inject, c_inject, delta


class SSMDecoder(nn.Module):
    def __init__(self, embedding_dim: int, vocab_size: int, state_dim: int = STATE_DIM):
        super().__init__()
        self.d_model        = embedding_dim
        self.d_state        = state_dim
        self.in_proj        = nn.Linear(embedding_dim, embedding_dim * 2)
        self.log_A          = nn.Parameter(
            torch.log(torch.arange(1, state_dim + 1, dtype=torch.float)
                      .unsqueeze(0).expand(embedding_dim, -1))
        )
        self.B_proj         = nn.Linear(embedding_dim, state_dim, bias=False)
        self.C_proj         = nn.Linear(embedding_dim, state_dim, bias=False)
        self.delta_proj     = nn.Linear(embedding_dim, embedding_dim)
        self.delta_bias     = nn.Parameter(torch.randn(embedding_dim) * 0.01)
        self.b_inject_proj  = nn.Linear(embedding_dim, state_dim, bias=False)
        self.c_inject_proj  = nn.Linear(embedding_dim, state_dim, bias=False)
        self.out_norm       = nn.LayerNorm(embedding_dim)
        self.out_proj       = nn.Linear(embedding_dim, embedding_dim)
        self.to_vocab       = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x, b_inject, c_inject):
        seq_len, d = x.shape
        xz      = self.in_proj(x)
        x_in    = xz[:, :d]
        z       = torch.sigmoid(xz[:, d:])
        delta   = F.softplus(self.delta_proj(x_in) + self.delta_bias)
        A       = -torch.exp(self.log_A)
        A_bar   = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0))
        B_raw   = self.B_proj(x_in)
        B_scale = 1.0 + torch.tanh(self.b_inject_proj(b_inject))
        B_seq   = B_raw * B_scale
        inv_A   = 1.0 / A
        B_bar   = (A_bar - 1.0) * inv_A.unsqueeze(0) * B_seq.unsqueeze(1)
        C_raw   = self.C_proj(x_in)
        C_scale = 1.0 + torch.tanh(self.c_inject_proj(c_inject))
        C_seq   = C_raw * C_scale

        p = 1
        while p < seq_len:
            p <<= 1

        a = torch.ones(p, d, self.d_state, device=x.device, dtype=x.dtype)
        b = torch.zeros(p, d, self.d_state, device=x.device, dtype=x.dtype)
        a[:seq_len] = A_bar
        b[:seq_len] = B_bar

        step = 1
        while step < p:
            left  = torch.arange(step - 1, p, step * 2)
            right = left + step
            mask  = right < p
            l, r  = left[mask], right[mask]
            a[r]  = a[r] * a[l]
            b[r]  = a[r] * b[l] + b[r]
            step <<= 1

        a[p - 1] = 1.0
        b[p - 1] = 0.0
        step = p >> 1
        while step >= 1:
            left  = torch.arange(step - 1, p, step * 2)
            right = left + step
            mask  = right < p
            l, r  = left[mask], right[mask]
            old_al = a[l].clone()
            old_bl = b[l].clone()
            old_ar = a[r].clone()
            old_br = b[r].clone()
            a[l] = old_ar
            b[l] = old_ar * old_bl + old_br
            a[r] = old_ar * old_al
            b[r] = old_ar * old_bl + old_br
            step >>= 1

        h_seq = b[:seq_len]
        y_seq = (h_seq * C_seq.unsqueeze(1)).sum(-1)
        out   = self.out_norm(y_seq * z + x_in)
        out   = self.out_proj(out)
        return self.to_vocab(out)


@dataclass
class ASTReport:
    score_vec : torch.Tensor
    status    : str
    details   : dict

def _ast_depth(node, depth=0):
    children = list(ast.iter_child_nodes(node))
    if not children:
        return depth
    return max(_ast_depth(c, depth + 1) for c in children)

def _token_entropy(ids):
    if len(ids) == 0:
        return 0.0
    counts      = torch.bincount(torch.tensor(ids, dtype=torch.long))
    probs       = counts.float() / counts.sum()
    probs       = probs[probs > 0]
    entropy     = -(probs * probs.log()).sum().item()
    max_entropy = math.log(len(ids) + 1e-9)
    return entropy / max_entropy if max_entropy > 0 else 0.0


class ASTDiagnosticSystem(nn.Module):
    def __init__(self, encoder, embedding_dim: int):
        super().__init__()
        self.encoder       = encoder
        self.embedding_dim = embedding_dim
        self.signal_dim    = 8
        self.signal_proj   = nn.Linear(self.signal_dim, embedding_dim)

    def feedback_from_signal(self, signal: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Fast path: uses pre-computed signal instead of live AST parsing."""
        signal_seq = signal.unsqueeze(0).expand(seq_len, -1)
        return self.signal_proj(signal_seq)

    def get_feedback(self, logits: torch.Tensor) -> Tuple[torch.Tensor, ASTReport]:
        """Original live path — available for inference/eval."""
        ids      = logits.argmax(dim=-1).tolist()
        code_str = self.encoder.decode(ids)
        seq_len  = logits.size(0)

        syntax_ok = error_position = depth_score = 0.0
        node_diversity = return_present = func_def_present = undefined_ratio = 0.0
        status = "unparsed"

        try:
            tree             = ast.parse(code_str)
            syntax_ok        = 1.0
            status           = "Valid AST"
            all_nodes        = list(ast.walk(tree))
            node_types       = [type(n).__name__ for n in all_nodes]
            total_nodes      = max(len(all_nodes), 1)
            raw_depth        = _ast_depth(tree)
            depth_score      = min(raw_depth / 20.0, 1.0)
            node_diversity   = len(set(node_types)) / total_nodes
            return_present   = 1.0 if any(isinstance(n, ast.Return)      for n in all_nodes) else 0.0
            func_def_present = 1.0 if any(isinstance(n, ast.FunctionDef) for n in all_nodes) else 0.0
            assigned         = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
            used             = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
            undefined        = used - assigned - {'True','False','None','print','range','len',
                                                   'int','str','float','list','dict','set',
                                                   'tuple','type','zip','map','enumerate'}
            undefined_ratio  = len(undefined) / max(len(used), 1)
        except SyntaxError as e:
            total_lines    = max(len(code_str.splitlines()), 1)
            error_line     = getattr(e, 'lineno', 0) or 0
            error_position = min(error_line / total_lines, 1.0)
            status         = f"SyntaxError line {error_line}: {e.msg}"

        token_ent  = _token_entropy(ids)
        signal     = torch.tensor([
            syntax_ok, error_position, depth_score, node_diversity,
            return_present, func_def_present, undefined_ratio, token_ent,
        ], dtype=torch.float, device=logits.device)
        signal_seq = signal.unsqueeze(0).expand(seq_len, -1)
        feedback   = self.signal_proj(signal_seq)
        details    = {
            "syntax_ok": syntax_ok, "error_position": error_position,
            "depth_score": depth_score, "node_diversity": f"{node_diversity:.3f}",
            "return_present": return_present, "func_def_present": func_def_present,
            "undefined_ratio": f"{undefined_ratio:.3f}", "token_entropy": f"{token_ent:.3f}",
        }
        return feedback, ASTReport(score_vec=signal, status=status, details=details)


# ─────────────────────────────────────────────────────────────────────────────
# BUILD MODEL
# ─────────────────────────────────────────────────────────────────────────────

encoder    = CodeEncoder().to(DEVICE)
decoder    = SSMDecoder(EMB_DIM, vocab_size=encoder.vocab_size, state_dim=STATE_DIM).to(DEVICE)
gnn        = CodeGNN(EMB_DIM).to(DEVICE)
token_attn = TokenAttention(EMB_DIM, num_heads=8).to(DEVICE)
reasoner   = ReasoningBlock(EMB_DIM).to(DEVICE)
diagnostic = ASTDiagnosticSystem(encoder, EMB_DIM).to(DEVICE)

all_params = (
    list(encoder.parameters())    +
    list(decoder.parameters())    +
    list(gnn.parameters())        +
    list(token_attn.parameters()) +
    list(reasoner.parameters())   +
    list(diagnostic.parameters())
)

optimizer = torch.optim.AdamW(all_params, lr=LR, weight_decay=0.01)
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=WARMUP_STEPS,
    num_training_steps=TOTAL_STEPS,
)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD CHECKPOINT
# ─────────────────────────────────────────────────────────────────────────────

start_step   = 0
total_tokens = 0

if os.path.exists(LOAD_PATH):
    print(f"Resuming from: {LOAD_PATH}")
    ckpt = torch.load(LOAD_PATH, map_location=DEVICE)
    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])
    gnn.load_state_dict(ckpt["gnn"])
    token_attn.load_state_dict(ckpt["token_attn"])
    reasoner.load_state_dict(ckpt["reasoner"], strict=False)
    diagnostic.load_state_dict(ckpt["diagnostic"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    else:
        resumed_step = ckpt["step"]
        for _ in range(min(resumed_step, TOTAL_STEPS)):
            scheduler.step()
        print(f"  [scheduler] fast-forwarded to step {resumed_step}")
    start_step   = ckpt["step"]
    total_tokens = ckpt.get("total_tokens", 0)
    print(f"Resumed at step {start_step}, tokens seen: {total_tokens:,}")
else:
    print("No checkpoint found — starting fresh")

# ─────────────────────────────────────────────────────────────────────────────
# LOAD PRE-COMPUTED DATASET
# ─────────────────────────────────────────────────────────────────────────────

print(f"Loading dataset from {DATASET_PATH}...")
dataset  = torch.load(DATASET_PATH)
indices  = list(range(len(dataset)))
random.shuffle(indices)
data_pos = 0
print(f"Loaded {len(dataset):,} samples.")

def next_sample():
    global data_pos, indices
    if data_pos >= len(indices):
        random.shuffle(indices)
        data_pos = 0
        print("Dataset reshuffled for next epoch.")
    record   = dataset[indices[data_pos]]
    data_pos += 1
    return record

# ─────────────────────────────────────────────────────────────────────────────
# SAVE CHECKPOINT
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(step, total_tokens):
    torch.save({
        "encoder"     : encoder.state_dict(),
        "decoder"     : decoder.state_dict(),
        "gnn"         : gnn.state_dict(),
        "token_attn"  : token_attn.state_dict(),
        "reasoner"    : reasoner.state_dict(),
        "diagnostic"  : diagnostic.state_dict(),
        "optimizer"   : optimizer.state_dict(),
        "scheduler"   : scheduler.state_dict(),
        "step"        : step,
        "total_tokens": total_tokens,
    }, CKPT_PATH)
    print(f"  [ckpt saved] step={step} | tokens={total_tokens:,}")

# ─────────────────────────────────────────────────────────────────────────────
# TRAIN STEP  (model logic identical — only feedback source changed)
# ─────────────────────────────────────────────────────────────────────────────

def train_step(record: dict):
    id_tensor  = record["ids"].to(DEVICE)
    ast_signal = record["ast_signal"].to(DEVICE)  # (8,) pre-computed, no parsing
    edge_index = record["edge_index"].to(DEVICE)

    if id_tensor.size(0) < 2:
        return None, 0

    embeddings, target_ids = encoder.encode_ids(id_tensor)
    max_node   = embeddings.size(0) - 1
    edge_index = edge_index.clamp(max=max_node)

    high_h      = torch.zeros(embeddings.size(0), EMB_DIM, device=DEVICE)
    low_h       = torch.zeros(embeddings.size(0), EMB_DIM, device=DEVICE)
    delta_accum = torch.zeros(embeddings.size(0), EMB_DIM, device=DEVICE)
    gnn_base    = None

    # pre-compute feedback from cached signal — no AST parsing, no decode
    feedback = diagnostic.feedback_from_signal(ast_signal, embeddings.size(0))

    for i in range(3):
        if i == 0:
            features                              = token_attn(embeddings)
            high_h, low_h, b_bias, c_bias, delta = reasoner(features, high_h, low_h, feedback)
            gnn_base                              = gnn(high_h, edge_index)
            delta_accum                           = (delta_accum * 0.7 + delta * 0.3).clamp(-1, 1)
            logits                                = decoder(features, b_bias, c_bias)
        else:
            high_h, low_h, b_bias, c_bias, delta = reasoner(
                gnn_base + delta_accum, high_h, low_h, feedback
            )
            delta_accum = (delta_accum * 0.7 + delta * 0.3).clamp(-1, 1)
            delta_scale = gnn_base.norm() / (delta_accum.norm() + 1e-6)
            features    = gnn_base + delta_accum * delta_scale * 0.2
            logits      = decoder(features, b_bias, c_bias)

    loss = F.cross_entropy(logits[:-1], target_ids[1:]) / ACCUM_STEPS
    loss.backward()
    return loss.item() * ACCUM_STEPS, embeddings.size(0)

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

print("\nStarting training...\n")

step           = start_step
last_ckpt_time = time.time()
loss_accum     = 0.0
loss_count     = 0

optimizer.zero_grad()

while True:
    record         = next_sample()
    loss, n_tokens = train_step(record)

    if loss is None:
        continue

    total_tokens += n_tokens
    loss_accum   += loss
    loss_count   += 1

    if loss_count % ACCUM_STEPS == 0:
        torch.nn.utils.clip_grad_norm_(all_params, GRAD_CLIP)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        step += 1

        if step % LOG_EVERY == 0:
            avg_loss   = loss_accum / loss_count
            current_lr = scheduler.get_last_lr()[0]
            print(f"step {step:>7} | loss {avg_loss:.4f} | lr {current_lr:.2e} | tokens {total_tokens:>12,}")
            loss_accum = 0.0
            loss_count = 0

        if time.time() - last_ckpt_time >= CKPT_EVERY:
            save_checkpoint(step, total_tokens)
            last_ckpt_time = time.time()