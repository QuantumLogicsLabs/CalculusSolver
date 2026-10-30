import json
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional

import joblib
import torch

from model.transformer import CalculusSolverModel
from inference.beam_search import NodeValidityPool, beam_search, load_vocab


class _LegacySLaNgTokenizer:
    """Compatibility placeholder for legacy serialized checkpoints."""


class PklTransformerModel(torch.nn.Module):
    """Transformer wrapper for legacy model.pkl checkpoints."""

    def __init__(self, vocab_size: int, hidden_dim: int = 128):
        super().__init__()
        self.pos_encoder = torch.nn.Parameter(torch.zeros(1, 100, hidden_dim))
        self.embedding = torch.nn.Embedding(vocab_size, hidden_dim)
        self.transformer = torch.nn.Transformer(
            d_model=hidden_dim,
            nhead=8,
            num_encoder_layers=3,
            num_decoder_layers=3,
            dim_feedforward=2048,
            batch_first=True,
        )
        self.fc_out = torch.nn.Linear(hidden_dim, vocab_size)

    def forward(self, src_seq, tgt_in_seq, true_rule_ids=None):
        """Return logits and placeholder rule/tracer outputs."""
        max_id = self.embedding.num_embeddings - 1
        src_seq = torch.clamp(src_seq, 0, max_id)
        tgt_in_seq = torch.clamp(tgt_in_seq, 0, max_id)

        max_pos = self.pos_encoder.size(1)
        src_seq = src_seq[:, :max_pos]
        tgt_in_seq = tgt_in_seq[:, :max_pos]

        src_emb = (
            self.embedding(src_seq)
            + self.pos_encoder[:, :src_seq.size(1), :]
        )
        tgt_emb = (
            self.embedding(tgt_in_seq)
            + self.pos_encoder[:, :tgt_in_seq.size(1), :]
        )

        device = src_seq.device
        tgt_mask = torch.triu(
            torch.ones(
                tgt_in_seq.size(1),
                tgt_in_seq.size(1),
                dtype=torch.bool,
                device=device,
            ),
            diagonal=1,
        )

        output = self.transformer(
            src_emb,
            tgt_emb,
            tgt_mask=tgt_mask,
        )
        logits = self.fc_out(output)

        return logits, None, None


class CalculusSolverInference:
    """Inference interface for calculus expressions and gradient requests."""

    def __init__(
        self,
        model_path: str = os.path.join("model", "model.pkl"),
        vocab_path: str = os.path.join("tokenizer", "vocab.json"),
        beam_size: int = 5,
        max_len: int = 80,
        enable_fallback: bool = True,
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model checkpoint not found: {model_path}"
            )

        if not os.path.exists(vocab_path):
            raise FileNotFoundError(
                f"Vocab file not found: {vocab_path}"
            )

        self.enable_fallback = enable_fallback
        self.fallback_solver = None

        if self.enable_fallback:
            try:
                from inference.fallback_solver import FallbackSolver

                self.fallback_solver = FallbackSolver()
            except Exception:
                self.fallback_solver = None

        self.model_data = self._load_checkpoint(model_path)
        self.vocab_map = load_vocab(vocab_path)

        config = (
            self.model_data.get("config", {})
            if isinstance(self.model_data, dict)
            and "config" in self.model_data
            else {}
        )

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        rule_labels = self._load_rule_labels(vocab_path)

        if isinstance(self.model_data, torch.nn.Module):
            self.model = self.model_data.to(self.device)
        else:
            state_dict = self._resolve_state_dict(self.model_data)

            if (
                "fc_out.weight" in state_dict
                and "pos_encoder" in state_dict
            ):
                vocab_size = state_dict["embedding.weight"].shape[0]
                hidden_dim = state_dict["embedding.weight"].shape[1]

                self.model = PklTransformerModel(
                    vocab_size=vocab_size,
                    hidden_dim=hidden_dim,
                ).to(self.device)
            else:
                hidden_dim = 128

                try:
                    root_dir = os.path.dirname(
                        os.path.dirname(os.path.abspath(__file__))
                    )
                    config_path = os.path.join(root_dir, "config.json")

                    if os.path.exists(config_path):
                        with open(
                            config_path,
                            "r",
                            encoding="utf-8",
                        ) as config_file:
                            config_data = json.load(config_file)
                            hidden_dim = config_data.get(
                                "hidden_dim",
                                128,
                            )
                except (OSError, ValueError, TypeError):
                    pass

                vocab_size = (
                    max(self.vocab_map["token_to_id"].values()) + 1
                )

                self.model = CalculusSolverModel(
                    vocab_size=vocab_size,
                    num_rules=len(rule_labels),
                    hidden_dim=hidden_dim,
                    rule_labels=rule_labels,
                ).to(self.device)

            try:
                self.model.load_state_dict(state_dict, strict=True)
            except RuntimeError:
                print(
                    "Warning: Checkpoint state dict mismatch detected. "
                    "Loading with strict=False..."
                )
                self.model.load_state_dict(
                    state_dict,
                    strict=False,
                )

        self.model.eval()

        self.beam_size = beam_size
        self.max_len = max_len

        self.node_pool = NodeValidityPool(
            os.path.join(
                os.path.dirname(__file__),
                "validity_worker.js",
            ),
            num_workers=max(2, beam_size),
        )

        self.bos_id = self.vocab_map["token_to_id"]["[BOS]"]
        self.eos_id = self.vocab_map["token_to_id"]["[EOS]"]
        self.pad_id = self.vocab_map["token_to_id"]["[PAD]"]

    def close(self) -> None:
        """Release resources used by the validity worker pool."""
        self.node_pool.close()

    def _load_rule_labels(self, vocab_path: str) -> List[str]:
        """Load rule labels from the vocabulary file."""
        with open(vocab_path, "r", encoding="utf-8") as vocab_file:
            vocab_json = json.load(vocab_file)

        rule_labels = []

        for token in vocab_json.get("rule_tokens", {}).keys():
            if token.startswith("RULE:"):
                rule_labels.append(token.split("RULE:", 1)[-1])
            else:
                rule_labels.append(token)

        return rule_labels

    def _load_checkpoint(self, model_path: str) -> Any:
        """Load a PyTorch or legacy joblib checkpoint."""
        if model_path.endswith((".pt", ".pth")):
            return torch.load(model_path, map_location="cpu")

        try:
            if not hasattr(sys.modules["__main__"], "SLaNgTokenizer"):
                setattr(
                    sys.modules["__main__"],
                    "SLaNgTokenizer",
                    _LegacySLaNgTokenizer,
                )

            return joblib.load(model_path)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load checkpoint {model_path}: {exc}"
            ) from exc

    def _resolve_state_dict(self, checkpoint: Any) -> Dict[str, Any]:
        """Extract model parameters from supported checkpoint formats."""
        if isinstance(checkpoint, dict):
            if "model_state" in checkpoint:
                return checkpoint["model_state"]

            if "model_state_dict" in checkpoint:
                return checkpoint["model_state_dict"]

            return checkpoint

        raise ValueError(
            "Unsupported checkpoint format for model state."
        )

    def _normalize_input(
        self,
        input_env: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Normalize shorthand expressions into fraction form."""
        env = dict(input_env)
        expr = env.get("expr")

        if isinstance(expr, dict):
            if (
                "coeff" in expr
                and "numi" not in expr
                and "op" not in expr
            ):
                env["expr"] = {
                    "numi": {"terms": [expr]},
                    "deno": 1,
                }
            elif "terms" in expr and "numi" not in expr:
                env["expr"] = {
                    "numi": expr,
                    "deno": 1,
                }

        return env

    def _infer_rule_token(
        self,
        input_env: Dict[str, Any],
    ) -> Optional[str]:
        """Infer an appropriate rule token from the operation and expression."""
        op = input_env.get("op")
        expr = input_env.get("expr", {})

        if op == "integrate":
            return "RULE:power_rule_integral"

        if op == "partial":
            return "RULE:partial_derivative"

        if op == "gradient":
            return "RULE:gradient"

        if op == "tangent_line":
            return "RULE:tangent_line"

        if op == "diff":
            if isinstance(expr, dict) and "op" in expr:
                sub_op = expr["op"]

                if sub_op in ("sin", "cos", "tan"):
                    return "RULE:trig_rule"

                if sub_op == "exp":
                    return "RULE:exp_rule"

                if sub_op == "ln":
                    return "RULE:log_rule"

            terms = (
                expr.get("numi", {}).get("terms", [])
                if isinstance(expr, dict)
                else []
            )

            if len(terms) > 1:
                return "RULE:sum_rule"

            if len(terms) == 1:
                term = terms[0]

                if (
                    not term.get("var")
                    or all(
                        value == 0
                        for value in term.get("var", {}).values()
                    )
                ):
                    return "RULE:constant_rule"

                return "RULE:power_rule"

        return None

    def _serialize_input(
        self,
        input_env: Dict[str, Any],
    ) -> List[str]:
        """Serialize an expression into SLaNg tokens."""
        from tokenizer.slang_serializer import serialize_slang_math

        return serialize_slang_math(input_env)

    def _verify_output(
        self,
        input_env: Dict[str, Any],
        output_tokens: List[str],
    ) -> Dict[str, Any]:
        """Verify generated output tokens."""
        from inference.verifier import verify

        return verify(input_env, output_tokens)

    @staticmethod
    def _extract_gradient_variables(expr: Any) -> List[str]:
        """Extract unique variable names from an expression structure."""
        variables: List[str] = []
        seen = set()

        def add_variable(name: Any) -> None:
            if (
                isinstance(name, str)
                and name
                and name not in seen
            ):
                seen.add(name)
                variables.append(name)

        def visit(value: Any) -> None:
            if isinstance(value, dict):
                for key, child in value.items():
                    if key == "vars" and isinstance(child, (list, tuple)):
                        for name in child:
                            add_variable(name)

                    elif key == "var":
                        if isinstance(child, dict):
                            for name in child:
                                add_variable(name)
                        elif isinstance(child, str):
                            add_variable(child)

                    visit(child)

            elif isinstance(value, (list, tuple)):
                for child in value:
                    visit(child)

        visit(expr)
        return variables

    def _solve_gradient(
        self,
        input_env: Dict[str, Any],
        variables: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Compute a gradient by solving one partial derivative per variable."""
        if not isinstance(input_env, dict):
            raise TypeError("input_env must be a dict")

        if "expr" in input_env:
            expr = input_env["expr"]
        else:
            expr = input_env
            input_env = {"op": "gradient", "expr": expr}

        if variables is None:
            explicit_variables = input_env.get(
                "vars",
                input_env.get("variables"),
            )

            if isinstance(explicit_variables, str):
                variables = [explicit_variables]
            elif isinstance(explicit_variables, (list, tuple)):
                variables = [
                    variable
                    for variable in explicit_variables
                    if isinstance(variable, str) and variable
                ]
            else:
                variables = []

            # Extract all expression variables before using the single-var fallback.
            if not variables:
                variables = self._extract_gradient_variables(expr)

            if not variables and isinstance(input_env.get("var"), str):
                variables = [input_env["var"]]

        if not variables:
            return {
        gradient_result = {}
        return {"input": input_env, "output": gradient_result}
                {"gradient": gradient_result}
                if gradient_result
                else None
            ),
            "warning": warning,
        }

    def solve(self, input_env: Dict[str, Any]) -> Dict[str, Any]:
        """Solve a calculus request, dispatching gradients into partial solves."""
        normalized_env = self._normalize_input(input_env)

        # Task 1: Decompose a gradient into individual partial derivatives.
        if normalized_env.get("op") == "gradient":
            return self._solve_gradient(normalized_env)

        rule_token = self._infer_rule_token(normalized_env)
        token_strings = self._serialize_input(normalized_env)

        token_ids = [
            self.vocab_map["token_to_id"].get(token, self.pad_id)
            for token in token_strings
        ]
        token_ids = token_ids[: self.max_len]

        padded_tokens = token_ids + [
            self.pad_id
        ] * (self.max_len - len(token_ids))

        src_tokens = torch.tensor(
            [padded_tokens],
            dtype=torch.long,
            device=self.device,
        )

        allowed_opvars = None
        if normalized_env.get("op") == "gradient":
            from inference.verifier import get_variables
            input_vars = get_variables(normalized_env)
            if input_vars:
                allowed_opvars = {f"OPVAR:{v}" for v in input_vars}

        result = beam_search(
            model=self.model,
            src_tokens=src_tokens,
            vocab_map=self.vocab_map,
            beam_size=self.beam_size,
            max_len=self.max_len,
            node_pool=self.node_pool,
            seed_rule_token=rule_token,
            allowed_opvars=allowed_opvars,
        )

        output_token_strings = [
            self.vocab_map["id_to_token"][token_id]
            for token_id in result["tokens"]
            if token_id in self.vocab_map["id_to_token"]
        ]

        if (
            output_token_strings
            and output_token_strings[0] == "[BOS]"
        ):
            output_token_strings = output_token_strings[1:]

        predicted_rule = None

        if (
            output_token_strings
            and output_token_strings[0].startswith("RULE:")
        ):
            predicted_rule = output_token_strings[0]
            output_token_strings = output_token_strings[1:]

        verifier_result = self._verify_output(
            normalized_env,
            output_token_strings,
        )

        if verifier_result.get("status") in (
            "solved",
            "unverified",
            "unsolvable",
        ):
            result["status"] = verifier_result["status"]

        result["verified"] = verifier_result.get("verified", False)
        result["confidence"] = verifier_result.get("confidence", 0)
        result["output"] = verifier_result.get("output")

        if verifier_result.get("error"):
            result["warning"] = verifier_result["error"]

        # Task 2: Preserve the existing fallback behavior for normal solves.
        if (
            not result["verified"]
            and self.enable_fallback
            and self.fallback_solver is not None
        ):
            try:
                fallback_result = self.fallback_solver.solve(
                    normalized_env
                )

                if fallback_result.get("status") == "solved":
                    fallback_output = (
                        fallback_result.get("output")
                        or fallback_result.get("expr")
                    )

                    if fallback_output is not None:
                        result["output"] = fallback_output
                        result["verified"] = True
                        result["status"] = "solved"
                        result["confidence"] = 1.0
                        result["fallback_used"] = True

                        if (
                            not predicted_rule
                            and fallback_result.get("rule")
                        ):
                            predicted_rule = fallback_result["rule"]

            except Exception:
                pass

        return {
            "input": input_env,
            "output_tokens": output_token_strings,
            "status": result["status"],
            "verified": result["verified"],
            "confidence": result["confidence"],
            "rule": predicted_rule,
            "output": result["output"],
            "warning": result.get("warning"),
        }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage: python inference/solve.py input.json"
        )

    with open(sys.argv[1], "r", encoding="utf-8") as input_file:
        payload = json.load(input_file)

    solver = CalculusSolverInference()

    try:
        print(json.dumps(solver.solve(payload), indent=2))
    finally:
        solver.close()