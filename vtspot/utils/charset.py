"""Character-set management for the CTC / attention recognition heads.

Choosing the charset is a *from-scratch* design decision that the literature
usually glosses over: with random initialisation there is no pretrained
embedding table to fall back on, so every extra symbol is another class that
must be learnt from your own data.  Keep the alphabet as small as the target
benchmark allows.

Index layout (CTC):
    0                      -> CTC blank
    1 .. num_chars         -> real characters
    num_chars + 1          -> [UNK] (characters seen at train time that are not
                              in the alphabet; never emitted at decode time)

Index layout (attention branch, used only during training for GTC guidance):
    0 -> [GO]/[PAD], 1 -> [EOS], 2 .. -> real characters
"""

from __future__ import annotations

import string
from pathlib import Path
from typing import Iterable, List, Sequence

# 36 symbols: what ICDAR video spotting is actually scored on (case-insensitive
# alphanumeric).  This is the right default for ICDAR13/15-video and DSText.
ALNUM = string.digits + string.ascii_lowercase

# 94 printable ASCII symbols minus space; needed if you want to keep punctuation.
ASCII94 = string.digits + string.ascii_letters + string.punctuation

PRESETS = {"alnum": ALNUM, "ascii94": ASCII94}


class Charset:
    """Maps between transcriptions and integer label sequences."""

    def __init__(self, chars: Iterable[str], case_sensitive: bool = False):
        seen: List[str] = []
        for ch in chars:
            if ch not in seen:
                seen.append(ch)
        self.chars = seen
        self.case_sensitive = case_sensitive
        self.blank = 0
        self.unk = len(self.chars) + 1
        self._c2i = {c: i + 1 for i, c in enumerate(self.chars)}

    # -- construction ----------------------------------------------------
    @classmethod
    def from_preset(cls, name: str, case_sensitive: bool = False) -> "Charset":
        if name not in PRESETS:
            raise KeyError(f"unknown charset preset {name!r}; have {sorted(PRESETS)}")
        return cls(PRESETS[name], case_sensitive=case_sensitive)

    @classmethod
    def from_file(cls, path: str | Path, case_sensitive: bool = True) -> "Charset":
        """One character per line -- use this for Chinese (BOVText) alphabets."""
        text = Path(path).read_text(encoding="utf-8")
        chars = [line for line in text.split("\n") if line != ""]
        return cls(chars, case_sensitive=case_sensitive)

    @classmethod
    def build(cls, spec: str, case_sensitive: bool = False) -> "Charset":
        if spec in PRESETS:
            return cls.from_preset(spec, case_sensitive)
        return cls.from_file(spec, case_sensitive=case_sensitive)

    # -- sizes -----------------------------------------------------------
    @property
    def num_chars(self) -> int:
        return len(self.chars)

    @property
    def ctc_num_classes(self) -> int:
        """blank + characters + UNK."""
        return len(self.chars) + 2

    @property
    def attn_num_classes(self) -> int:
        """[PAD]/[GO] + [EOS] + characters."""
        return len(self.chars) + 2

    # -- text <-> labels -------------------------------------------------
    def normalise(self, text: str) -> str:
        return text if self.case_sensitive else text.lower()

    def encode(self, text: str) -> List[int]:
        """Transcription -> CTC target indices.  Unknown chars map to UNK."""
        return [self._c2i.get(ch, self.unk) for ch in self.normalise(text)]

    def encode_attn(self, text: str, max_len: int) -> List[int]:
        """Transcription -> attention targets, EOS-terminated and PAD-filled."""
        ids = [self._c2i.get(ch, 0) + 1 for ch in self.normalise(text)][: max_len - 1]
        ids = [i if i > 1 else 0 for i in ids]  # UNK -> PAD, it is not supervised
        ids.append(1)  # EOS
        ids += [0] * (max_len - len(ids))
        return ids

    def decode(self, indices: Sequence[int]) -> str:
        out = []
        for i in indices:
            if 1 <= i <= len(self.chars):
                out.append(self.chars[i - 1])
        return "".join(out)

    def is_encodable(self, text: str) -> bool:
        """True when every character is in the alphabet.

        Instances that fail this are still useful for detection and tracking but
        must be excluded from the recognition loss, otherwise the CTC head is
        asked to predict a class it can never emit.
        """
        return all(ch in self._c2i for ch in self.normalise(text))


def filter_transcription(text: str, charset: Charset) -> str:
    """Drop characters outside the alphabet -- the ICDAR evaluation convention."""
    norm = charset.normalise(text)
    return "".join(ch for ch in norm if ch in charset._c2i)
