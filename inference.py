import torch
import torch.nn.functional as F
import random
import numpy as np
from model import (CodeEncoder, SSMDecoder, CodeGNN,
                   TokenAttention, ReasoningBlock,
                   ASTDiagnosticSystem, get_edge_index_sequential)

# Set seeds for deterministic behavior
def set_deterministic(seed=42):
    """
    Why seed=42? 
    - 42 is the "Answer to the Ultimate Question of Life, the Universe, and Everything" from Hitchhiker's Guide
    - It's a commonly used default seed that makes results reproducible across different codebases
    - Any fixed number would work, but 42 has become a de facto standard
    - The actual value doesn't matter as long as it's consistent
    
    Why set multiple seeds?
    - torch.manual_seed(): PyTorch CPU operations
    - torch.cuda.manual_seed_all(): PyTorch GPU operations (all GPUs)
    - np.random.seed(): NumPy operations (used by some PyTorch functions)
    - random.seed(): Python's built-in random (used by some libraries)
    - torch.backends.cudnn.deterministic: Forces deterministic CuDNN algorithms
    - torch.backends.cudnn.benchmark: Disables auto-tuning for consistency
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

DEVICE = torch.device('cpu')
EMB_DIM = 256
STATE_DIM = 32
MAX_SEQ_LEN = 512
CKPT_PATH = 'checkpoint (4).pt'

# Set deterministic
set_deterministic(42)

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

print("Model loaded successfully with improved state management")

# ── Simple Effective State Management ───────────────────────────────────────────
def compress_recent_states(states_history, num_recent=5):
    """
    Simple weighted average of recent states
    More effective than complex attention for this use case
    """
    if len(states_history) <= num_recent:
        # Not enough history, use last state
        return states_history[-1]
    
    # Take last N states and weight by recency
    recent_states = torch.cat(states_history[-num_recent:], dim=0)
    
    # Exponential weights: most recent = highest weight
    weights = torch.tensor([0.1, 0.15, 0.2, 0.25, 0.3], device=states_history[0].device)
    
    # Weighted average
    compressed_state = torch.sum(recent_states * weights.unsqueeze(-1), dim=0, keepdim=True)
    
    return compressed_state

# ── Generation function (Updated with simple effective state management) ───────────
def generate(prompt: str, max_new_tokens: int = 100, temperature: float = 0.7, top_p: float = 0.8, use_greedy=False):
    """Deterministic generation with simple effective state management"""
    
    tokens = encoder.tokenizer.encode(prompt)
    
    # Initialize states properly outside the loop
    high_h = torch.zeros(1, EMB_DIM, device=DEVICE)
    low_h = torch.zeros(1, EMB_DIM, device=DEVICE)
    feedback = torch.zeros(1, EMB_DIM, device=DEVICE)
    delta_accum = torch.zeros(1, EMB_DIM, device=DEVICE)
    
    # Store recent states for better context (keep last 10)
    max_history = 10
    high_h_history = []
    low_h_history = []
    feedback_history = []
    delta_accum_history = []

    with torch.no_grad():
        for step in range(max_new_tokens):
            if len(tokens) >= MAX_SEQ_LEN:
                break

            # Get embeddings
            id_tensor = torch.tensor(tokens, dtype=torch.long, device=DEVICE)
            positions = torch.arange(len(id_tensor), device=DEVICE)
            embeddings = encoder.norm(encoder.token_emb(id_tensor) + encoder.pos_emb(positions))

            # Use sequential edges for stability
            edge_index = get_edge_index_sequential(len(tokens)).to(DEVICE)
            max_node = embeddings.size(0) - 1
            edge_index = edge_index.clamp(max=max_node)

            # Expand states to current sequence length
            current_seq_len = embeddings.size(0)
            if high_h.size(0) != current_seq_len:
                high_h = high_h[:1].expand(current_seq_len, -1).contiguous()
                low_h = low_h[:1].expand(current_seq_len, -1).contiguous()
                feedback = feedback[:1].expand(current_seq_len, -1).contiguous()
                delta_accum = delta_accum[:1].expand(current_seq_len, -1).contiguous()

            # Run through reasoning layers (reduced iterations for stability)
            for i in range(2):  # Reduced from 3
                if i == 0:
                    features = token_attn(embeddings)
                    high_h, low_h, b_bias, c_bias, delta = reasoner(features, high_h, low_h, feedback)
                    gnn_base = gnn(high_h, edge_index)
                    delta_accum = (delta_accum * 0.8 + delta * 0.2).clamp(-1, 1)  # More conservative
                    logits = decoder(features, b_bias, c_bias)
                else:
                    high_h, low_h, b_bias, c_bias, delta = reasoner(
                        gnn_base + delta_accum, high_h, low_h, feedback
                    )
                    delta_accum = (delta_accum * 0.8 + delta * 0.2).clamp(-1, 1)
                    delta_scale = min(gnn_base.norm() / (delta_accum.norm() + 1e-6), 1.0)  # Clamp scale
                    features = gnn_base + delta_accum * delta_scale * 0.1  # Reduced impact
                    logits = decoder(features, b_bias, c_bias)

                feedback, _ = diagnostic.get_feedback(logits.detach())

            # Get next token deterministically
            last_logits = logits[-1] / temperature
            
            # Prevent immediate repetition
            if len(tokens) > 0:
                last_logits[tokens[-1]] -= 3.0  # Stronger penalty
            
            # Prevent excessive spaces/newlines
            try:
                space_tokens = [encoder.tokenizer.encode(' ')[0], encoder.tokenizer.encode('\n')[0]]
                for space_token in space_tokens:
                    if space_token < len(last_logits):
                        last_logits[space_token] -= 2.0
            except:
                pass  # Fallback if space token encoding fails

            if use_greedy:
                # Pure greedy for maximum determinism
                next_token = last_logits.argmax().item()
            else:
                # Deterministic top-p sampling
                sorted_logits, sorted_indices = torch.sort(last_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = 0
                sorted_logits[sorted_indices_to_remove] = float('-inf')
                probs = F.softmax(sorted_logits, dim=-1)
                
                # Use argmax on filtered distribution for determinism
                if probs.max() > 0.5:  # If we have a confident prediction
                    next_token = sorted_indices[probs.argmax()].item()
                else:
                    next_token = sorted_indices[0].item()  # Fallback to top token

            tokens.append(next_token)

            # SIMPLE EFFECTIVE STATE MANAGEMENT
            # Add current states to history
            high_h_history.append(high_h[-1:].clone())
            low_h_history.append(low_h[-1:].clone())
            feedback_history.append(feedback[-1:].clone())
            delta_accum_history.append(delta_accum[-1:].clone())
            
            # Limit history size
            if len(high_h_history) > max_history:
                high_h_history.pop(0)
                low_h_history.pop(0)
                feedback_history.pop(0)
                delta_accum_history.pop(0)
            
            # Use simple weighted average of recent states
            if len(high_h_history) > 1:
                high_h = compress_recent_states(high_h_history)
                low_h = compress_recent_states(low_h_history)
                feedback = compress_recent_states(feedback_history)
                delta_accum = compress_recent_states(delta_accum_history)
            else:
                # Keep only last token's state (fallback)
                high_h = high_h[-1:].contiguous()
                low_h = low_h[-1:].contiguous()
                feedback = feedback[-1:].contiguous()
                delta_accum = delta_accum[-1:].contiguous()

            if next_token == encoder.tokenizer.eos_token_id:
                break

    return encoder.tokenizer.decode(tokens, skip_special_tokens=True)


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    prompt = "def calculate_sum(a, b):"
    print(f"\nPrompt: {prompt}\n")
    
    print("=== DETERMINISTIC GENERATION TEST ===")
    
    # Test greedy generation (should be identical each time)
    print("\n--- Greedy Generation (100% deterministic) ---")
    result1 = generate(prompt, max_new_tokens=50, temperature=0.1, use_greedy=True)
    print(f"Run 1: {result1}")
    
    result2 = generate(prompt, max_new_tokens=50, temperature=0.1, use_greedy=True)
    print(f"Run 2: {result2}")
    
    print(f"Identical results: {result1 == result2}")
    
    # Test deterministic sampling
    print("\n--- Deterministic Sampling ---")
    result3 = generate(prompt, max_new_tokens=50, temperature=0.7, top_p=0.8)
    print(f"Run 1: {result3}")
    
    result4 = generate(prompt, max_new_tokens=50, temperature=0.7, top_p=0.8)
    print(f"Run 2: {result4}")
    
    print(f"Identical results: {result3 == result4}")
    
    # Show checkpoint info
    try:
        ckpt = torch.load(CKPT_PATH, map_location='cpu')
        print(f"\nCheckpoint info:")
        print(f"  Step: {ckpt['step']}")
        print(f"  Total tokens: {ckpt.get('total_tokens', 0):,}")
    except:
        print(f"\nCould not load checkpoint {CKPT_PATH}")