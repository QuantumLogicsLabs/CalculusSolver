import os
import logging

import torch

logger = logging.getLogger(__name__)

RULE_DESCRIPTIONS = {
    "power_rule":          "d/dx[x^n] = n*x^(n-1)",
    "chain_rule":          "d/dx[f(g(x))] = f'(g(x))*g'(x)",
    "product_rule":        "d/dx[u*v] = u'*v + u*v'",
    "quotient_rule":       "d/dx[u/v] = (v*u' - u*v') / v^2",
    "sum_rule":            "d/dx[f+g] = f' + g'",
    "constant_rule":       "d/dx[c] = 0",
    "power_rule_integral": "integral of x^n = x^(n+1)/(n+1) + C",
    "partial_derivative":  "df/dx: treat other variables as constants",
    "lagrange_multiplier": "grad(f) = lambda * grad(g) at constrained extrema",
}

RULE_MAP = {
    "diff":         "power_rule",
    "integrate":    "power_rule_integral",
    "gradient":     "partial_derivative",
    "product_rule": "product_rule",
    "quotient_rule":"quotient_rule",
    "lagrange":     "lagrange_multiplier",
}

_RULE_LIST = list(RULE_MAP.values())


class CalculusInference:
    """
    Loads the trained model once and exposes a solve() method.
    Falls back to placeholder mode if the model is not ready yet.
    """

    MAX_DECODE_STEPS = 128

    def __init__(
        self,
        checkpoint_path: str = "checkpoints/final/best.pt",
        vocab_path: str = "tokenizer/vocab_flat.json",
    ):
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load(checkpoint_path, vocab_path)

    # ------------------------------------------------------------------
    # Internal: model loading
    # ------------------------------------------------------------------

    def _load(self, checkpoint_path: str, vocab_path: str) -> None:
        try:
            if not os.path.exists(checkpoint_path):
                logger.info(
                    "[Inference] No checkpoint at %s — placeholder mode",
                    checkpoint_path,
                )
                return

            checkpoint = torch.load(
                checkpoint_path,
                map_location=self.device,
                weights_only=True,
            )

            note = str(checkpoint.get("note", "")).lower()
            if "placeholder" in note:
                logger.info("[Inference] Placeholder checkpoint — placeholder mode")
                return

            from tokenizer.slang_tokenizer import SLaNgTokenizer
            from model.architecture import CalculusSolverModel

            self.tokenizer = SLaNgTokenizer(vocab_path)
            self.model = CalculusSolverModel(vocab_size=self.tokenizer.vocab_size)
            self.model.load_state_dict(checkpoint["model_state"])
            self.model.to(self.device)
            self.model.eval()

            logger.info(
                "[Inference] Real model loaded — step %d",
                checkpoint.get("step", 0),
            )

        except ImportError as exc:
            logger.warning("[Inference] Import error (%s) — placeholder mode", exc)

        except Exception as exc:
            logger.exception("[Inference] Load failed (%s) — placeholder mode", exc)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def solve(self, input_envelope: dict) -> dict:
        """
        Input:
            {"op": "diff", "var": "x", "expr": {...}}

        Output:
            {
                "status":       "solved" | "placeholder" | "error",
                "rule":         str,
                "confidence":   float,
                "steps":        [{"rule": str, "description": str}],
                "output_tokens": list[int],
            }
        """
        if not isinstance(input_envelope, dict) or "op" not in input_envelope:
            return {
                "status": "error",
                "message": 'Invalid input: expected a dict containing an "op" key.',
            }

        if self.model is None:
            return self._placeholder_solve(input_envelope)

        return self._real_solve(input_envelope)

    # ------------------------------------------------------------------
    # Internal: placeholder path
    # ------------------------------------------------------------------

    def _placeholder_solve(self, inp: dict) -> dict:
        op = inp.get("op", "diff")
        rule = RULE_MAP.get(op, "power_rule")
        desc = RULE_DESCRIPTIONS.get(rule, f"Apply {rule}")

        return {
            "status": "placeholder",
            "rule": rule,
            "confidence": 0.0,
            "output_tokens": [],
            "steps": [{"rule": rule, "description": desc}],
            "note": "Placeholder — real model not loaded yet",
        }

    # ------------------------------------------------------------------
    # Internal: real inference (greedy decoding)
    # ------------------------------------------------------------------

    def _real_solve(self, inp: dict) -> dict:
        try:
            src_tokens = self.tokenizer.encode(inp)
        except Exception as exc:
            return {"status": "error", "message": f"Tokenization failed: {exc}"}

        src = torch.tensor([src_tokens], dtype=torch.long, device=self.device)

        with torch.no_grad():
            src_pad = src == self.tokenizer.PAD_ID
            memory = self.model.encoder(src, src_key_padding_mask=src_pad)
            rule_id = self.model.rule_head(memory).argmax(dim=-1).item()

        output_tokens = [self.tokenizer.BOS_ID]
        log_probs = []

        with torch.no_grad():
            for _ in range(self.MAX_DECODE_STEPS):
                tgt = torch.tensor(
                    [output_tokens], dtype=torch.long, device=self.device
                )
                tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(
                    tgt.size(1), device=self.device
                )

                dec_logits, _, _ = self.model.decoder(
                    tgt, memory, tgt_mask=tgt_mask
                )

                next_tok = dec_logits[0, -1, :].argmax().item()
                probs = torch.softmax(dec_logits[0, -1, :], dim=-1)
                log_probs.append(torch.log(probs[next_tok] + 1e-10).item())

                if next_tok == self.tokenizer.EOS_ID:
                    break

                output_tokens.append(next_tok)

        # Confidence: avg log-prob rescaled so -10 nats → 0.0, 0 nats → 1.0
        avg_lp = sum(log_probs) / max(len(log_probs), 1)
        confidence = float(min(1.0, max(0.0, 1.0 + avg_lp / 10.0)))

        rule_name = self._resolve_rule_name(rule_id)
        desc = RULE_DESCRIPTIONS.get(rule_name, f"Apply {rule_name}")

        return {
            "status": "solved",
            "output_tokens": output_tokens[1:],
            "rule": rule_name,
            "confidence": round(confidence, 4),
            "steps": [{"rule": rule_name, "description": desc}],
        }

    # ------------------------------------------------------------------
    # Internal: map rule_id integer → rule name string
    # ------------------------------------------------------------------

    def _resolve_rule_name(self, rule_id: int) -> str:
        try:
            from training.dataset import RULE_LABEL_MAP

            id_to_rule = {v: k for k, v in RULE_LABEL_MAP.items()}
            name = id_to_rule.get(rule_id)
            if name is not None:
                return name
        except ImportError:
            pass

        # Fallback: wrap rule_id into the known rule list
        return _RULE_LIST[rule_id % len(_RULE_LIST)]