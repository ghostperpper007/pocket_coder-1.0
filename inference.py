import torch
import torch.nn.functional as F
from model import (CodeEncoder, SSMDecoder, CodeGNN,
                   TokenAttention, ReasoningBlock,
                   ASTDiagnosticSystem, get_edge_index)

DEVICE = torch.device('cpu')
EMB_DIM = 256
STATE_DIM = 32
MAX_SEQ_LEN = 512
CKPT_PATH = 'checkpoint (3).pt'

# ── Build model ───────────────────────────────────────────────────────────────
encoder    = CodeEncoder(EMB_DIM, MAX_SEQ_LEN).to(DEVICE)
decoder    = SSMDecoder(EMB_DIM, vocab_size=encoder.vocab_size, state_dim=STATE_DIM).to(DEVICE)
gnn        = CodeGNN(EMB_DIM).to(DEVICE)
token_attn = TokenAttention(EMB_DIM, num_heads=8).to(DEVICE)
reasoner   = ReasoningBlock(EMB_DIM).to(DEVICE)
diagnostic = ASTDiagnosticSystem(encoder, EMB_DIM).to(DEVICE)

# ── Load checkpoint ───────────────────────────────────────────────────────────
ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
encoder.load_state_dict(ckpt['encoder'])
decoder.load_state_dict(ckpt['decoder'])
gnn.load_state_dict(ckpt['gnn'])
token_attn.load_state_dict(ckpt['token_attn'])
reasoner.load_state_dict(ckpt['reasoner'])
diagnostic.load_state_dict(ckpt['diagnostic'])

encoder.eval()
decoder.eval()
gnn.eval()
token_attn.eval()
reasoner.eval()
diagnostic.eval()

print("Model loaded successfully")

# ── Generation function ───────────────────────────────────────────────────────
def generate(prompt: str, max_new_tokens: int = 100, temperature: float = 0.8, top_p: float = 0.95):
    tokens = encoder.tokenizer.encode(prompt)
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            if len(tokens) >= MAX_SEQ_LEN:
                break

            # build current sequence
            source = encoder.tokenizer.decode(tokens)
            id_tensor = torch.tensor(tokens, dtype=torch.long, device=DEVICE)
            positions = torch.arange(len(id_tensor), device=DEVICE)
            embeddings = encoder.norm(encoder.token_emb(id_tensor) + encoder.pos_emb(positions))

            edge_index = get_edge_index(source).to(DEVICE)
            max_node   = embeddings.size(0) - 1
            edge_index = edge_index.clamp(max=max_node)

            high_h      = torch.zeros(embeddings.size(0), EMB_DIM, device=DEVICE)
            low_h       = torch.zeros(embeddings.size(0), EMB_DIM, device=DEVICE)
            feedback    = torch.zeros(embeddings.size(0), EMB_DIM, device=DEVICE)
            delta_accum = torch.zeros(embeddings.size(0), EMB_DIM, device=DEVICE)
            gnn_base    = None

            for i in range(3):
                if i == 0:
                    features = token_attn(embeddings)
                    high_h, low_h, b_bias, c_bias, delta = reasoner(features, high_h, low_h, feedback)
                    gnn_base    = gnn(high_h, edge_index)
                    delta_accum = (delta_accum * 0.7 + delta * 0.3).clamp(-1, 1)
                    logits      = decoder(features, b_bias, c_bias)
                else:
                    high_h, low_h, b_bias, c_bias, delta = reasoner(
                        gnn_base + delta_accum, high_h, low_h, feedback
                    )
                    delta_accum = (delta_accum * 0.7 + delta * 0.3).clamp(-1, 1)
                    delta_scale = gnn_base.norm() / (delta_accum.norm() + 1e-6)
                    features    = gnn_base + delta_accum * delta_scale * 0.2
                    logits      = decoder(features, b_bias, c_bias)

                feedback, _ = diagnostic.get_feedback(logits.detach())

            # sample from last token logits only
            last_logits = logits[-1] / temperature

            # top-p sampling
            sorted_logits, sorted_indices = torch.sort(last_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs - F.softmax(sorted_logits, dim=-1) > top_p
            sorted_logits[sorted_indices_to_remove] = float('-inf')
            probs     = F.softmax(sorted_logits, dim=-1)
            next_token = sorted_indices[torch.multinomial(probs, 1)].item()

            tokens.append(next_token)

            # stop at eos
            if next_token == encoder.tokenizer.eos_token_id:
                break

    return encoder.tokenizer.decode(tokens, skip_special_tokens=True)


# ── Run ───────────────────────────────────────────────────────────────────────
prompt = "def calculate_sum(a, b):"
print(f"\nPrompt: {prompt}\n")
print("Generated:")
print(generate(prompt, max_new_tokens=150, temperature=0.8, top_p=0.95))