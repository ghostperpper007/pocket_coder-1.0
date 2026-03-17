import torch
import torch.nn.functional as F
from model import (CodeEncoder, SSMDecoder, CodeGNN,
                   TokenAttention, ReasoningBlock,
                   ASTDiagnosticSystem, get_edge_index, get_edge_index_sequential)

DEVICE = torch.device('cpu')
EMB_DIM = 256
STATE_DIM = 32
MAX_SEQ_LEN = 512
CKPT_PATH = 'checkpoint (7).pt'

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

# ── Generation function (Option C: Full Debug) ───────────────────────────────────
def generate(prompt: str, max_new_tokens: int = 100, temperature: float = 0.8, top_p: float = 0.9):
    tokens = encoder.tokenizer.encode(prompt)
    
    # Initialize states ONCE outside the generation loop
    high_h = torch.zeros(1, EMB_DIM, device=DEVICE)
    low_h = torch.zeros(1, EMB_DIM, device=DEVICE)
    feedback = torch.zeros(1, EMB_DIM, device=DEVICE)
    delta_accum = torch.zeros(1, EMB_DIM, device=DEVICE)
    gnn_base = None

    print(f"Initial prompt: '{prompt}'")
    print(f"Initial tokens: {tokens[:10]}...")  # Show first 10 tokens
    
    with torch.no_grad():
        for step in range(max_new_tokens):
            if len(tokens) >= MAX_SEQ_LEN:
                break

            id_tensor = torch.tensor(tokens, dtype=torch.long, device=DEVICE)
            positions = torch.arange(len(id_tensor), device=DEVICE)
            embeddings = encoder.norm(encoder.token_emb(id_tensor) + encoder.pos_emb(positions))

            # Enhanced edge index handling with logging
            current_code = encoder.tokenizer.decode(tokens, skip_special_tokens=True)
            try:
                edge_index = get_edge_index(current_code).to(DEVICE)
                edge_type = "AST"
                if edge_index.size(1) == 0:
                    raise ValueError("Empty AST edges")
            except:
                edge_index = get_edge_index_sequential(len(tokens)).to(DEVICE)
                edge_type = "Sequential"
            
            max_node = embeddings.size(0) - 1
            edge_index = edge_index.clamp(max=max_node)

            # Expand states to current sequence length
            current_seq_len = embeddings.size(0)
            if high_h.size(0) != current_seq_len:
                high_h = high_h[:1].expand(current_seq_len, -1).contiguous()
                low_h = low_h[:1].expand(current_seq_len, -1).contiguous()
                feedback = feedback[:1].expand(current_seq_len, -1).contiguous()
                delta_accum = delta_accum[:1].expand(current_seq_len, -1).contiguous()

            for i in range(3):
                if i == 0:
                    features = token_attn(embeddings)
                    high_h, low_h, b_bias, c_bias, delta = reasoner(features, high_h, low_h, feedback)
                    gnn_base = gnn(high_h, edge_index)
                    delta_accum = (delta_accum * 0.7 + delta * 0.3).clamp(-1, 1)
                    logits = decoder(features, b_bias, c_bias)
                else:
                    high_h, low_h, b_bias, c_bias, delta = reasoner(
                        gnn_base + delta_accum, high_h, low_h, feedback
                    )
                    delta_accum = (delta_accum * 0.7 + delta * 0.3).clamp(-1, 1)
                    delta_scale = gnn_base.norm() / (delta_accum.norm() + 1e-6)
                    features = gnn_base + delta_accum * delta_scale * 0.2
                    logits = decoder(features, b_bias, c_bias)

                feedback, _ = diagnostic.get_feedback(logits.detach())

            # Enhanced sampling with comprehensive logging
            last_logits = logits[-1] / temperature
            
            # Stronger repetition prevention
            if len(tokens) > 0:
                last_logits[tokens[-1]] -= 2.0  # Heavy penalty for immediate repetition
                if len(tokens) > 1:
                    last_logits[tokens[-2]] -= 1.0  # Light penalty for 2-gram repetition
            
            # Existing token penalty
            for token_id in set(tokens):
                last_logits[token_id] /= 1.5

            # Get top tokens for debugging
            top_tokens = torch.topk(last_logits, 10)
            top_decoded = [encoder.tokenizer.decode([t.item()]) for t in top_tokens.indices]
            top_probs = F.softmax(last_logits, dim=-1)[top_tokens.indices]
            
            print(f"\n--- Step {step+1} ---")
            print(f"Edge type: {edge_type}")
            print(f"Current code: '{current_code[-50:]}'")  # Show last 50 chars
            print(f"Top 10 tokens: {list(zip(top_decoded, top_probs.tolist()))[:5]}")
            print(f"Feedback norm: {feedback.norm().item():.4f}")
            print(f"High hidden norm: {high_h.norm().item():.4f}")
            print(f"Low hidden norm: {low_h.norm().item():.4f}")

            # More conservative top-p sampling
            sorted_logits, sorted_indices = torch.sort(last_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs - F.softmax(sorted_logits, dim=-1) > top_p
            sorted_logits[sorted_indices_to_remove] = float('-inf')
            probs = F.softmax(sorted_logits, dim=-1)
            
            # Add small epsilon to prevent zero probability issues
            probs = probs + 1e-8
            probs = probs / probs.sum()
            
            next_token = sorted_indices[torch.multinomial(probs, 1)].item()
            next_token_str = encoder.tokenizer.decode([next_token])
            
            print(f"Selected token: '{next_token_str}' (ID: {next_token})")

            tokens.append(next_token)

            if next_token == encoder.tokenizer.eos_token_id:
                print("EOS token generated - stopping")
                break

            # Keep only the last token's state for next iteration
            high_h = high_h[-1:].contiguous()
            low_h = low_h[-1:].contiguous()
            feedback = feedback[-1:].contiguous()
            delta_accum = delta_accum[-1:].contiguous()

    final_result = encoder.tokenizer.decode(tokens, skip_special_tokens=True)
    print(f"\n=== FINAL RESULT ===")
    print(f"Generated: {final_result}")
    return final_result

# ── Run ───────────────────────────────────────────────────────────────────────
prompt = "def calculate_sum(a, b):"
print(f"\n=== OPTION C: FULL DEBUG ===")
print(f"Prompt: {prompt}\n")
print("Starting generation with detailed logging...")
result = generate(prompt, max_new_tokens=50, temperature=0.8, top_p=0.9)  # Reduced tokens for debugging
