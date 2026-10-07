class SimpleTokenizer:

  def __init__(self):
    self.word_to_id = {}
    self.id_to_word = {}
    self.vocab_size = 0
    self.pad_token = "<PAD>"
    self.unk_token = "<UNK>"
    self.bos_token = "<BOS>"
    self.eos_token = "<EOS>"

  def build_vocab(self, texts: list[str]) -> None:
    """Xây dựng từ điển (chuyển về lowercase để tránh lệch case)."""
    special_tokens = [
        self.pad_token,
        self.unk_token,
        self.bos_token,
        self.eos_token,
    ]
    for i, token in enumerate(special_tokens):
      self.word_to_id[token] = i
      self.id_to_word[i] = token

    unique_words = set()
    for text in texts:
      # Chuyển về chữ thường trước khi tách từ
      words = text.lower().strip().split()
      unique_words.update(words)

    current_id = len(special_tokens)
    for word in sorted(unique_words):
      if word not in self.word_to_id:
        self.word_to_id[word] = current_id
        self.id_to_word[current_id] = word
        current_id += 1

    self.vocab_size = len(self.word_to_id)

  def encode(self, text: str) -> list[int]:
    """Mã hóa văn bản thành token IDs."""
    # Chuyển về chữ thường để khớp với từ điển
    words = text.lower().strip().split()
    unk_id = self.word_to_id[self.unk_token]

    return [self.word_to_id.get(word, unk_id) for word in words]

  def decode(self, ids: list[int]) -> str:
    """Giải mã token IDs thành văn bản."""
    words = [self.id_to_word.get(i, self.unk_token) for i in ids]
    return " ".join(words)