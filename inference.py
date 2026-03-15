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