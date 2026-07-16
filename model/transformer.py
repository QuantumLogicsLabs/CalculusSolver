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
    ):
        super().__init__()
        self.encoder = TreeEncoder(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ffn_dim=ffn_dim,
            dropout=dropout,
            position_dim=position_dim,
        )
        
        # Instantiate rule labels based on num_rules
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
        
        # In train.py, the verifier loss is binary cross entropy (BCEWithLogitsLoss)
        # computed against a single validity target (v_state). Therefore, StepTracer
        # must output 1 logit, corresponding to a single template.
        templates = ["is_valid"]
        self.step_tracer = StepTracer(
            hidden_dim=hidden_dim,
            templates=templates
        )

    def forward(self, src_seq, tgt_in_seq):
        device = src_seq.device
        batch_size, seq_len = src_seq.size()
        
        # Construct standard empty positions and parent_child_pairs
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
        
        # 2. Get rule logits
        rule_logits = self.rule_head(encoder_output)
        
        # 3. Embed rule IDs for decoder
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
