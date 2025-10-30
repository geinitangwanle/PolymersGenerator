from typing import Optional

try:
    import huggingface_hub  # noqa: F401
except ImportError:
    huggingface_hub = None
else:
    if not hasattr(huggingface_hub, "get_full_repo_name"):
        def _fallback_get_full_repo_name(model_id, organization=None, token=None):
            if organization and not model_id.startswith(f"{organization}/"):
                return f"{organization}/{model_id}"
            return model_id

        huggingface_hub.get_full_repo_name = _fallback_get_full_repo_name  # type: ignore[attr-defined]

try:
    from transformers import AutoTokenizer
except ImportError as exc:  # pragma: no cover
    raise ImportError("transformers must be installed to use PolyBertTokenizer.") from exc


class PolyBertTokenizer:
    """Thin wrapper around the polyBERT tokenizer with a legacy-compatible API."""

    def __init__(self, name: str = "kuelumbus/polyBERT", tokenizer=None, use_fast: bool = False):
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(name, use_fast=use_fast)

        # Ensure pad token exists; fall back to SEP/EOS if needed.
        if self.tokenizer.pad_token is None:
            fallback = self.tokenizer.sep_token or self.tokenizer.eos_token
            if fallback is None:
                raise ValueError("Tokenizer must define pad or sep/eos tokens.")
            pad_id = self.tokenizer.convert_tokens_to_ids(fallback)
            if pad_id == self.tokenizer.unk_token_id:
                self.tokenizer.add_special_tokens({"pad_token": fallback})
            else:
                self.tokenizer.pad_token = fallback

    @property
    def pad_id(self) -> int:
        return self.tokenizer.pad_token_id

    @property
    def bos_id(self) -> Optional[int]:
        # Prefer CLS, otherwise fall back to BOS if available.
        if self.tokenizer.cls_token_id is not None:
            return self.tokenizer.cls_token_id
        return getattr(self.tokenizer, "bos_token_id", None)

    @property
    def eos_id(self) -> Optional[int]:
        if self.tokenizer.sep_token_id is not None:
            return self.tokenizer.sep_token_id
        return getattr(self.tokenizer, "eos_token_id", None)

    @property
    def vocab_size(self) -> int:
        base = self.tokenizer.vocab_size
        if hasattr(self.tokenizer, "get_added_vocab"):
            base += len(self.tokenizer.get_added_vocab())
        return base

    def encode(self, text: str):
        return self.tokenizer.encode(
            text,
            add_special_tokens=True,
            truncation=False,  # 让上层决定截断
        )

    def decode(self, ids, skip_special_tokens: bool = True):
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def get_vocab(self):
        return self.tokenizer.get_vocab()
