import torch
from model import (CodeEncoder, SSMDecoder, CodeGNN, TokenAttention, 
                   ReasoningBlock, ASTDiagnosticSystem)

EMB_DIM = 256
MAX_SEQ_LEN = 512
STATE_DIM = 32

# Build model components
encoder = CodeEncoder(EMB_DIM, MAX_SEQ_LEN)
decoder = SSMDecoder(EMB_DIM, vocab_size=encoder.vocab_size, state_dim=STATE_DIM)
gnn = CodeGNN(EMB_DIM)
token_attn = TokenAttention(EMB_DIM, num_heads=8)
reasoner = ReasoningBlock(EMB_DIM)
diagnostic = ASTDiagnosticSystem(encoder, EMB_DIM)

def count_params(model):
    """Count total parameters in a model"""
    return sum(p.numel() for p in model.parameters())

def count_trainable_params(model):
    """Count trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_size_mb(model):
    """Get model size in MB (assuming float32)"""
    return count_params(model) * 4 / (1024 * 1024)

# Loading checkpoint info
try:
    ckpt = torch.load('checkpoint (7).pt', map_location='cpu')
    ckpt_step = ckpt['step']
    total_tokens = ckpt.get('total_tokens', 0)
    ckpt_available = True
except:
    ckpt_step = "N/A"
    total_tokens = "N/A"
    ckpt_available = False

print("=" * 60)
print("COMPREHENSIVE MODEL STATISTICS")
print("=" * 60)

print(f"\nTRAINING STATUS:")
print(f"  Training step: {ckpt_step}")
print(f"  Total tokens processed: {total_tokens:,}")
print(f"  Checkpoint available: {ckpt_available}")

print(f"\nMODEL ARCHITECTURE:")
print(f"  Embedding dimension: {EMB_DIM}")
print(f"  Max sequence length: {MAX_SEQ_LEN}")
print(f"  SSM state dimension: {STATE_DIM}")
print(f"  Vocabulary size: {encoder.vocab_size:,}")

print(f"\nPARAMETER COUNTS:")
components = [
    ("Encoder", encoder),
    ("Decoder", decoder), 
    ("GNN", gnn),
    ("Token Attention", token_attn),
    ("Reasoner", reasoner),
    ("Diagnostic", diagnostic)
]

total_params = 0
total_trainable = 0
total_size_mb = 0

for name, model in components:
    params = count_params(model)
    trainable = count_trainable_params(model)
    size_mb = get_model_size_mb(model)
    
    print(f"  {name}:")
    print(f"    Total params: {params:,}")
    print(f"    Trainable: {trainable:,}")
    print(f"    Size: {size_mb:.2f} MB")
    
    total_params += params
    total_trainable += trainable
    total_size_mb += size_mb

print(f"\nTOTAL MODEL:")
print(f"  Total parameters: {total_params:,}")
print(f"  Trainable parameters: {total_trainable:,}")
print(f"  Model size: {total_size_mb:.2f} MB")

print(f"\nPARAMETER BREAKDOWN:")
print(f"  Encoder: {count_params(encoder)/total_params*100:.1f}% of total")
print(f"  Decoder: {count_params(decoder)/total_params*100:.1f}% of total")
print(f"  GNN: {count_params(gnn)/total_params*100:.1f}% of total")
print(f"  Token Attention: {count_params(token_attn)/total_params*100:.1f}% of total")
print(f"  Reasoner: {count_params(reasoner)/total_params*100:.1f}% of total")
print(f"  Diagnostic: {count_params(diagnostic)/total_params*100:.1f}% of total")

print(f"\nMODEL COMPLEXITY:")
print(f"  Params per vocab token: {total_params/encoder.vocab_size:.1f}")
print(f"  Params per sequence position: {total_params/MAX_SEQ_LEN:.1f}")
print(f"  Model depth (approx): {total_params/(EMB_DIM * EMB_DIM):.1f} layers equivalent")

print(f"\nHARDWARE REQUIREMENTS (estimated):")
print(f"  VRAM for model (FP32): {total_size_mb:.0f} MB")
print(f"  VRAM for model (FP16): {total_size_mb/2:.0f} MB")
print(f"  VRAM for training (gradients+optimizer): ~{total_size_mb*4:.0f} MB")
print(f"  VRAM for inference (batch size 1): ~{total_size_mb*1.5:.0f} MB")

print("=" * 60)
