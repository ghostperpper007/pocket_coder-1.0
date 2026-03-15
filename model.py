import torch
import torch.nn as nn
from transformers import GPT2Tokenizer

import torch
import torch.nn as nn
import torch.nn.functional as F
import ast
import math
from dataclasses import dataclass
from typing import Tuple, Optional

class CodeEncoder(nn.Module):
    """
    Uses GPT2 vocabulary (50257 tokens) for encoding/decoding text.
    All weights are your own — only the vocab dictionary is borrowed.
    Output: (seq_len, embedding_dim)
    """
    def __init__(self, embedding_dim=600, max_seq_len=512):
        super().__init__()
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono")
        self.vocab_size = len(self.tokenizer)
        self.max_seq_len = max_seq_len

        # Your own weights — nothing from GPT2
        self.token_emb = nn.Embedding(len(self.tokenizer), embedding_dim)
        self.pos_emb   = nn.Embedding(max_seq_len, embedding_dim)
        self.norm      = nn.LayerNorm(embedding_dim)

    def encode(self, source: str):
        ids       = self.tokenizer.encode(source, truncation=True, max_length=self.max_seq_len)
        device    = next(self.parameters()).device          # ← add this
        id_tensor = torch.tensor(ids, dtype=torch.long, device=device)   # ← add device=
        positions = torch.arange(len(id_tensor), device=device)
        x = self.token_emb(id_tensor) + self.pos_emb(positions)
        return self.norm(x)                              # (seq_len, embedding_dim)

    def decode(self, ids: list) -> str:
        # Clamp ids to valid vocab range
        ids = [max(0, min(i, self.vocab_size - 1)) for i in ids]
        return self.tokenizer.decode(ids, skip_special_tokens=True)


source_complex = """
def calculate_area(radius):
    pi = 3.14159
    return pi * radius
"""

# ── Edge Index Helpers ────────────────────────────────────────────────────────

def get_edge_index(source):
    try:
        tree = ast.parse(source)
    except:
        return get_edge_index_sequential(source)

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

    if not edges:
        return get_edge_index_sequential(len(encoder.tokenizer.encode(source)))

    row, col = zip(*edges)
    return torch.tensor([list(row), list(col)], dtype=torch.long)

def get_edge_index_sequential(n_tokens):
    if isinstance(n_tokens, str):
        n_tokens = len(n_tokens.split())
    row = list(range(n_tokens - 1)) + list(range(1, n_tokens))
    col = list(range(1, n_tokens)) + list(range(n_tokens - 1))
    return torch.tensor([row, col], dtype=torch.long)

# ── Token Attention ───────────────────────────────────────────────────────────

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
        x_seq            = x.unsqueeze(0)
        attn_out, _      = self.attn(x_seq, x_seq, x_seq)
        x_seq            = self.norm(attn_out + x_seq)
        x_seq            = self.norm2(self.ff(x_seq) + x_seq)
        return x_seq.squeeze(0)


# ── GNN with Edge Attention ───────────────────────────────────────────────────

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

            agg      = torch.zeros_like(out_self)
            agg.index_add_(0, col, weighted)

            attn_sum = torch.zeros(num_nodes, 1, device=x.device, dtype=x.dtype)
            attn_sum.index_add_(0, col, attn_score.to(attn_sum.dtype))

            x = norm(self.activation(out_self + agg / (attn_sum + 1e-6)) + x)

        return x


# ── Reasoning Block ───────────────────────────────────────────────────────────

class ReasoningBlock(nn.Module):
    def __init__(self, embedding_dim, rank=64):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.rank          = rank
        self.high_a_down   = nn.Linear(embedding_dim, rank)
        self.high_a_up     = nn.Linear(rank, embedding_dim)
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


# ── True SSM (S4/Mamba-style) Decoder ────────────────────────────────────────

class SSMDecoder(nn.Module):
    def __init__(self, embedding_dim: int, vocab_size: int, state_dim: int = 64):
        super().__init__()
        self.d_model   = embedding_dim
        self.d_state   = state_dim

        self.in_proj   = nn.Linear(embedding_dim, embedding_dim * 2)

        self.log_A     = nn.Parameter(
            torch.log(torch.arange(1, state_dim + 1, dtype=torch.float)
                      .unsqueeze(0).expand(embedding_dim, -1))
        )
        self.B_proj    = nn.Linear(embedding_dim, state_dim, bias=False)
        self.C_proj    = nn.Linear(embedding_dim, state_dim, bias=False)
        self.delta_proj = nn.Linear(embedding_dim, embedding_dim)
        self.delta_bias = nn.Parameter(torch.randn(embedding_dim) * 0.01)

        self.b_inject_proj = nn.Linear(embedding_dim, state_dim, bias=False)
        self.c_inject_proj = nn.Linear(embedding_dim, state_dim, bias=False)

        self.out_norm  = nn.LayerNorm(embedding_dim)
        self.out_proj  = nn.Linear(embedding_dim, embedding_dim)
        self.to_vocab  = nn.Linear(embedding_dim, vocab_size)

    def forward(
    self,
    x: torch.Tensor,
    b_inject: torch.Tensor,
    c_inject: torch.Tensor,
) -> torch.Tensor:

        seq_len, d = x.shape
    
        xz     = self.in_proj(x)
        x_in   = xz[:, :d]
        z      = torch.sigmoid(xz[:, d:])
    
        delta  = F.softplus(self.delta_proj(x_in) + self.delta_bias)
        A      = -torch.exp(self.log_A)
        A_bar  = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0))
    
        B_raw   = self.B_proj(x_in)
        B_scale = 1.0 + torch.tanh(self.b_inject_proj(b_inject))
        B_seq   = B_raw * B_scale
    
        inv_A  = 1.0 / A
        B_bar  = (A_bar - 1.0) * inv_A.unsqueeze(0) * B_seq.unsqueeze(1)
    
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
    
        # Up-sweep (reduce)
        step = 1
        while step < p:
            left  = torch.arange(step - 1, p, step * 2)
            right = left + step
            mask  = right < p
            l, r  = left[mask], right[mask]
            a[r]  = a[r] * a[l]
            b[r]  = a[r] * b[l] + b[r]
            step <<= 1
    
        # Down-sweep
        a[p - 1] = 1.0
        b[p - 1] = 0.0
        step = p >> 1
        while step >= 1:
            left  = torch.arange(step - 1, p, step * 2)
            right = left + step
            mask  = right < p
            l, r  = left[mask], right[mask]
            old_al = a[l].clone()  # save a[l] before touching it
            old_bl = b[l].clone()  # save b[l] before touching it
            old_ar = a[r].clone()  # save a[r] before touching it
            old_br = b[r].clone()  # save b[r] before touching it
    
            a[l] = old_ar
            b[l] = old_ar * old_bl + old_br
            a[r] = old_ar * old_al
            b[r] = old_ar * old_bl + old_br
            step >>= 1
    
        h_seq  = b[:seq_len]
        y_seq  = (h_seq * C_seq.unsqueeze(1)).sum(-1)
    
        out    = self.out_norm(y_seq * z + x_in)
        out    = self.out_proj(out)
    
        return self.to_vocab(out)


# ── Rich AST Diagnostic System ────────────────────────────────────────────────

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
    counts = torch.bincount(torch.tensor(ids, dtype=torch.long))
    probs  = counts.float() / counts.sum()
    probs  = probs[probs > 0]
    entropy = -(probs * probs.log()).sum().item()
    max_entropy = math.log(len(ids) + 1e-9)
    return entropy / max_entropy if max_entropy > 0 else 0.0

# FIX 1: extends nn.Module + super().__init__() so signal_proj is registered
class ASTDiagnosticSystem(nn.Module):
    def __init__(self, encoder, embedding_dim: int):
        super().__init__()                                   # ← FIX 1
        self.encoder       = encoder
        self.embedding_dim = embedding_dim
        self.signal_dim    = 8
        self.signal_proj   = nn.Linear(self.signal_dim, embedding_dim)

    def get_feedback(self, logits: torch.Tensor) -> Tuple[torch.Tensor, ASTReport]:
        """
        logits : (seq_len, vocab_size)  — passed in detached at call site
        returns: feedback (seq_len, embedding_dim), report
        """
        ids      = logits.argmax(dim=-1).tolist()
        code_str = self.encoder.decode(ids)
        seq_len  = logits.size(0)

        syntax_ok        = 0.0
        error_position   = 0.0
        depth_score      = 0.0
        node_diversity   = 0.0
        return_present   = 0.0
        func_def_present = 0.0
        undefined_ratio  = 0.0
        status           = "unparsed"

        try:
            tree = ast.parse(code_str)
            syntax_ok = 1.0
            status    = "Valid AST"

            all_nodes   = list(ast.walk(tree))
            node_types  = [type(n).__name__ for n in all_nodes]
            total_nodes = max(len(all_nodes), 1)

            raw_depth        = _ast_depth(tree)
            depth_score      = min(raw_depth / 20.0, 1.0)
            node_diversity   = len(set(node_types)) / total_nodes
            return_present   = 1.0 if any(isinstance(n, ast.Return)      for n in all_nodes) else 0.0
            func_def_present = 1.0 if any(isinstance(n, ast.FunctionDef) for n in all_nodes) else 0.0

            assigned = {n.id for n in ast.walk(tree)
                        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
            used     = {n.id for n in ast.walk(tree)
                        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
            undefined = used - assigned - {'True','False','None','print','range',
                                            'len','int','str','float','list','dict',
                                            'set','tuple','type','zip','map','enumerate'}
            undefined_ratio = len(undefined) / max(len(used), 1)

        except SyntaxError as e:
            total_lines    = max(len(code_str.splitlines()), 1)
            error_line     = getattr(e, 'lineno', 0) or 0
            error_position = min(error_line / total_lines, 1.0)
            status         = f"SyntaxError line {error_line}: {e.msg}"

        token_ent = _token_entropy(ids)

        signal = torch.tensor([
            syntax_ok, error_position, depth_score, node_diversity,
            return_present, func_def_present, undefined_ratio, token_ent,
        ], dtype=torch.float)

        signal_seq = signal.unsqueeze(0).expand(seq_len, -1)
        feedback   = self.signal_proj(signal_seq)            # differentiable

        details = {
            "syntax_ok"       : syntax_ok,
            "error_position"  : error_position,
            "depth_score"     : depth_score,
            "node_diversity"  : f"{node_diversity:.3f}",
            "return_present"  : return_present,
            "func_def_present": func_def_present,
            "undefined_ratio" : f"{undefined_ratio:.3f}",
            "token_entropy"   : f"{token_ent:.3f}",
        }

        report = ASTReport(score_vec=signal, status=status, details=details)
        # FIX 2: no .detach() here — logits are detached at the call site instead,
        # keeping signal_proj gradients alive
        return feedback, report


#from this point on its training loop
'''
    print(f"Pass {i+1}:")
    print(f"  AST Status    : {report.status}")
    print(f"  Syntax OK     : {report.details['syntax_ok']}")
    print(f"  Depth Score   : {report.details['depth_score']:.3f}")
    print(f"  Node Diversity: {report.details['node_diversity']}")
    print(f"  Return Present: {report.details['return_present']}")
    print(f"  Undef Ratio   : {report.details['undefined_ratio']}")
    print(f"  Token Entropy : {report.details['token_entropy']}")
    print(f"  Delta Norm    : {delta_accum.norm().item():.4f}")
    print(f"  Unique IDs    : {len(set(logits.argmax(dim=-1).tolist()))}/{logits.size(0)}")
    output_text = encoder.decode(logits.argmax(dim=-1).tolist())
    print(f"  Output        : {output_text[:80]}")
    print()
'''
#commented out for inference