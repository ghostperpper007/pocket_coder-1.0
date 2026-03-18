# Small Programming Model

A compact, from-scratch Python code model built around a novel combination of graph-based structure encoding, state space model decoding, and a live differentiable AST feedback loop.

Trained on raw Python source with next-token prediction.

---

## Architecture

```
Source Code
    │
    ▼
CodeGen Tokenizer (Salesforce/codegen-350M-mono vocab)
    │
    ▼
Token Embedding + Positional Embedding
    │
    ▼
TokenAttention  (multi-head self-attention + FFN)
    │
    ▼
ReasoningBlock  (dual-state: high = context, low = planning)
    │
    ▼
CodeGNN         (2-layer edge-attentive graph neural network over AST)
    │
    ▼
SSMDecoder      (parallel-scan Mamba/S4-style with B/C injection)
    │
    ▼
ASTDiagnosticSystem  (live AST parse → differentiable feedback signal)
    │
    └──► fed back into ReasoningBlock on next pass (×3 passes per sample)
```

### Key Components

**CodeGNN** — builds a graph from the parsed AST of the input source and runs 2 rounds of edge-attentive message passing. Falls back to sequential edges if the source doesn't parse.

**ReasoningBlock** — maintains two hidden states across passes. The high-level state tracks broad context; the low-level state tracks planning signals injected from the diagnostic system. Produces B and C injection vectors fed into the decoder.

**SSMDecoder** — a Mamba/S4-style state space model with a parallel prefix-scan (work-efficient up-sweep / down-sweep). B and C matrices are modulated per-pass by the reasoning block, allowing the model to adjust its recurrent dynamics based on syntactic feedback.

**ASTDiagnosticSystem** — after each decoding pass, the output token IDs are decoded to a string and parsed with Python's `ast` module. Eight scalar signals are extracted (syntax validity, error position, AST depth, node diversity, return/funcdef presence, undefined variable ratio, token entropy) and projected into a feedback embedding that feeds back into the next reasoning pass. This keeps the feedback differentiable through `signal_proj` while stopping gradients through the argmax.

---

## Training

- **Dataset:** `bigcode/the-stack` Python subset (raw source files with comments) and `bigcode/starcoderdata`
- **Objective:** next-token prediction (`cross_entropy` on shifted token IDs)
- **Optimizer:** AdamW, lr=3e-4, weight_decay=0.01
- **Gradient clipping:** 1.0
- **Passes per sample:** 3 refinement passes with AST feedback between each
- **Checkpointing:** every 10 minutes to `/kaggle/working/checkpoint.pt`

---

## Why AST Feedback?

Standard language models receive no signal about whether their output is syntactically valid until evaluation. My model closes this loop during training — every forward pass includes a parse attempt, and the resulting structural signals are injected back into the model's hidden state before the next pass. This means the model is trained to condition on its own syntactic quality in real time.

---

## Parameters

| Component | Details |
|---|---|
| Embedding dim | 600 |
| SSM state dim | 64 |
| Max sequence length | 512 |
| GNN layers | 2 |
| Attention heads | 8 |
| Reasoning passes | 3 (training), 6 (inference demo) |
## Roadmap And Future Plans
- train up to 800m tokens to follow chinchilla scaling laws
- incorporate into a larger system.This is a spoiler for a system coming in distant future
and will await development until model is ready
and then will be developed in private until 
its ready for the public to use
- publish on hugging face
## Current State
the model is still undertrained and currently trained on 206m tokens and is starting to learn about relationships between tokens
and is still far from valid code but has higher confidence and more diversity in its outputs alongwith rephrasing tokens in many ways showing understanding of
relationships between tokens.
## Requirements

```
torch
transformers
datasets
```

---

## License

MIT