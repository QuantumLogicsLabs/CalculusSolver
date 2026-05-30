import torch
import torch.nn as nn

from .tree_encoder import TreeEncoder
from .tree_decoder import TreeDecoder
from .rule_head import RuleHead
from .step_tracer import StepTracer


class CalculusModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        rule_labels=None,
        hidden_dim=512,
        num_heads=8,
        num_layers=8,
        ffn_dim=2048,
        dropout=0.1,
        position_dim=3,
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
        self.rule_head = RuleHead(hidden_dim=hidden_dim, rule_labels=rule_labels)
        self.decoder = TreeDecoder(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )
        templates = [f"Describe {label}." for label in self.rule_head.labels()]
        self.step_tracer = StepTracer(hidden_dim=hidden_dim, templates=templates)

    def forward(
        self,
        src_tokens,
        src_positions,
        parent_child_pairs,
        tgt_tokens,
        root_mask=None,
        rule_ids=None,
        validity_mask=None,
        src_padding_mask=None,
        tgt_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        encoder_output = self.encoder(
            src_tokens, src_positions, parent_child_pairs, padding_mask=src_padding_mask
        )
        rule_logits = self.rule_head(encoder_output, root_mask=root_mask)

        if rule_ids is None:
            rule_ids = torch.argmax(rule_logits, dim=-1)

        rule_embeddings = self.rule_head.embed_rules(rule_ids)
        decoder_logits, decoder_hidden_states = self.decoder(
            tgt_tokens,
            encoder_output,
            rule_embeddings=rule_embeddings,
            validity_mask=validity_mask,
            tgt_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

        description_logits = self.step_tracer(rule_ids, decoder_hidden_states)
        return decoder_logits, rule_logits, description_logits
