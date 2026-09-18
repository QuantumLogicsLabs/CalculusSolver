from typing import List, Optional

import torch
import torch.nn as nn
from .tree_encoder import TreeEncoder
from .tree_decoder import TreeDecoder
from .rule_head import RuleHead
from .step_tracer import StepTracer

class CalculusSolverModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_rules: int,
        hidden_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
        position_dim: int = 3,
        rule_labels: Optional[List[str]] = None,
        pad_id: int = 0,
    ):
        super().__init__()
        self.pad_id = pad_id

        self.encoder = TreeEncoder(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ffn_dim=ffn_dim,
            dropout=dropout,
            position_dim=position_dim,
        )
        
        # 1. Validate custom rule_labels count if provided
        if rule_labels is not None and len(rule_labels) != num_rules:
            raise ValueError(
                f"Expected {num_rules} rule labels, but got {len(rule_labels)}."
            )

        # 2. Dynamic fallback with underscore formatting (resolves RULE_i test assertion)
        if rule_labels is None:
            rule_labels = [f"RULE_{i}" for i in range(num_rules)]

        self.rule_head = RuleHead(
            hidden_dim=hidden_dim,
            rule_labels=rule_labels
        )
        
        self.decoder = TreeDecoder(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )
        
        templates = ["is_valid"]
        self.step_tracer = StepTracer(
            hidden_dim=hidden_dim,
            templates=templates
        )

    def forward(self, src_seq, tgt_in_seq, true_rule_ids=None):
        """Run the full model.

        RETURN CONTRACT -- always a 3-tuple, never a single tensor:

            (decoder_logits, rule_logits, verifier_logits)

            decoder_logits   (batch, tgt_len, vocab_size)  next-token scores
            rule_logits      (batch, num_rules)            rule classifier
            verifier_logits  (batch, num_templates)        step tracer; 1 template

        Every caller must unpack explicitly, e.g.
            decoder_logits, rule_logits, _ = model(src_seq, tgt_in_seq)
        Treating the result as a tensor is what caused
        "'tuple' object has no attribute 'reshape'" in train.py and
        "tuple indices must be integers or slices, not tuple" in beam_search.

        check_forward_contract() below enforces this; train.py runs it before
        the first training step.

        Args:
            src_seq: (batch, src_len) source token ids.
            tgt_in_seq: (batch, tgt_len) decoder input token ids.
            true_rule_ids: optional (batch,) rule ids for teacher forcing the
                rule embedding. When None the model uses argmax(rule_logits).
        """
        device = src_seq.device
        batch_size, seq_len = src_seq.size()
        
        src_positions = torch.zeros(
            (batch_size, seq_len, 3), dtype=torch.float32, device=device
        )
        parent_child_pairs = torch.zeros(
            (batch_size, seq_len, seq_len), dtype=torch.float32, device=device
        )
        
        # 1. Encode source tokens
        encoder_output = self.encoder(
            src_seq, src_positions, parent_child_pairs
        )

        # 2. Get rule logits (using non-pad tokens root mask)
        root_mask = (src_seq != self.pad_id)
        rule_logits = self.rule_head(encoder_output, root_mask=root_mask)
        
        # 3. Embed rule IDs for decoder
        if true_rule_ids is not None:
            rule_ids = true_rule_ids
        else:
            rule_ids = torch.argmax(rule_logits, dim=-1)
        rule_embeddings = self.rule_head.embed_rules(rule_ids)
        
        # 4. Decode target tokens
        decoder_logits, decoder_hidden_states = self.decoder(
            tgt_in_seq,
            encoder_output,
            rule_embeddings=rule_embeddings,
        )
        
        # 5. Trace steps (verifier)
        verifier_logits = self.step_tracer(rule_ids, decoder_hidden_states)
        
        return decoder_logits, rule_logits, verifier_logits


def check_forward_contract(
    model: nn.Module,
    vocab_size: int,
    num_rules: int,
    batch_size: int = 2,
    src_len: int = 7,
    tgt_len: int = 5,
) -> None:
    """Run one dummy forward pass and fail loudly if the contract is broken.

    Checks that forward() returns exactly (decoder_logits, rule_logits,
    verifier_logits) with the expected shapes. It costs one forward pass on a
    tiny batch, so train.py runs it before the first step -- this class of
    mismatch previously surfaced only after multi-hour runs.

    Raises:
        TypeError: the output is not a 3-tuple of tensors.
        ValueError: a tensor has the wrong shape.
    """
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            # Ids >= 1 so no row is all padding (pad_id 0), which would give
            # the rule head an empty root mask.
            src = torch.randint(1, vocab_size, (batch_size, src_len), device=device)
            tgt = torch.randint(1, vocab_size, (batch_size, tgt_len), device=device)
            output = model(src, tgt)
    finally:
        model.train(was_training)

    if not isinstance(output, tuple) or len(output) != 3:
        raise TypeError(
            "model.forward() must return a 3-tuple "
            "(decoder_logits, rule_logits, verifier_logits); got "
            f"{type(output).__name__}"
            + (f" of length {len(output)}" if isinstance(output, tuple) else "")
        )

    decoder_logits, rule_logits, verifier_logits = output
    num_templates = getattr(getattr(model, "step_tracer", None), "num_templates", None)
    expected = {
        "decoder_logits": (decoder_logits, (batch_size, tgt_len, vocab_size)),
        "rule_logits": (rule_logits, (batch_size, num_rules)),
        "verifier_logits": (
            verifier_logits,
            (batch_size, num_templates) if num_templates is not None else None,
        ),
    }
    for name, (tensor, shape) in expected.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
        if shape is not None and tuple(tensor.shape) != shape:
            raise ValueError(f"{name} has shape {tuple(tensor.shape)}, expected {shape}")
