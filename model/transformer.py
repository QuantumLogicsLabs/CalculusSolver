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
        pad_id: int = 0,
    ):
        super().__init__()
        # FIX 1 (docs/KNOWN_ISSUES.md, "RuleHead only pools from token 0"):
        # RuleHead.forward() falls back to encoder_out[:, 0, :] whenever no
        # root_mask is supplied -- i.e. it bases every rule prediction on the
        # encoder's representation of ONLY the first source token, ignoring
        # the rest of the expression entirely. This was the likely root
        # cause of the persistent ~0.505 Val Rule loss plateau seen across
        # every training configuration tried (learning rate, data coverage,
        # rule-label conflation fix, gradient clipping, hidden_dim increase
        # all failed to break it). pad_id is now stored so forward() can
        # build a real root_mask covering all non-padding tokens.
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

    def forward(self, src_seq, tgt_in_seq, true_rule_ids=None):
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
        # FIX 1: build a root_mask covering every real (non-padding) source
        # token, instead of letting RuleHead silently fall back to pooling
        # only from position 0. This lets the rule classifier actually see
        # the whole expression (operator, operand, coefficients, structure)
        # rather than just the first token (e.g. "OP:diff"), which is
        # identical across many semantically different problems and cannot
        # by itself distinguish them.
        root_mask = (src_seq != self.pad_id)
        rule_logits = self.rule_head(encoder_output, root_mask=root_mask)
        
        # 3. Embed rule IDs for decoder
        # FIX 2 (docs/KNOWN_ISSUES.md, "rule/decoder circular dependency"):
        # model/architecture.py's older CalculusModel already demonstrates
        # this exact pattern -- an optional true_rule_ids parameter that,
        # when supplied (training), is used instead of the model's own
        # (possibly wrong) argmax prediction. Without this, the decoder was
        # always conditioned on the rule head's own guess even during
        # training, meaning a wrong early rule prediction corrupted the
        # decoder's training signal too, and neither component could
        # specialize independently. At inference time (true_rule_ids=None,
        # the default), behavior is unchanged -- the model still falls back
        # to its own prediction, exactly as before.
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