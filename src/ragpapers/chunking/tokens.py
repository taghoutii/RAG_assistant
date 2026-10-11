"""Count tokens with the embedding model's own tokenizer (bge-small), for token-based chunking.

The count excludes special tokens ([CLS], [SEP]); `special_tokens` says how many the model adds
to every sequence, so a full embedded sequence costs count_tokens(text) + special_tokens.
Counts are cached, because the chunker asks for the same words and pieces many times.
"""


class TokenCounter:
    """Callable text -> number of tokens, e.g. TokenCounter("BAAI/bge-small-en-v1.5")("Hi")."""

    def __init__(self, tokenizer_name: str):
        from transformers import AutoTokenizer  # slow import, only when token sizing is used

        self.name = tokenizer_name
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.special_tokens = self._tokenizer.num_special_tokens_to_add()
        self._cache: dict[str, int] = {}

    def __call__(self, text: str) -> int:
        n = self._cache.get(text)
        if n is None:
            # verbose=False: no "sequence longer than 512" warning; we only count here.
            ids = self._tokenizer(text, add_special_tokens=False, verbose=False)["input_ids"]
            n = len(ids)
            self._cache[text] = n
        return n
