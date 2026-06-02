"""
lemmatizer.py
"""

from functools import lru_cache

from natasha import MorphVocab, NewsEmbedding, NewsMorphTagger, Segmenter, Doc

_segmenter  = Segmenter()
_morph_vocab = MorphVocab()
_emb         = NewsEmbedding()
_morph_tagger = NewsMorphTagger(_emb)


def lemmatize_text(text: str) -> str:
    """Returns space-joined lemmas for all tokens in text"""
    if not text or not text.strip():
        return ""
    doc = Doc(text[:600])          # cap length for speed
    doc.segment(_segmenter)
    doc.tag_morph(_morph_tagger)
    for tok in doc.tokens:
        tok.lemmatize(_morph_vocab)
    return " ".join(tok.lemma or tok.text for tok in doc.tokens)


def lemmatize_word(word: str) -> str:
    """Returns the lemma of a single word"""
    doc = Doc(word.strip())
    doc.segment(_segmenter)
    doc.tag_morph(_morph_tagger)
    for tok in doc.tokens:
        tok.lemmatize(_morph_vocab)
    tokens = doc.tokens
    return tokens[0].lemma if tokens else word.lower()


@lru_cache(maxsize=256)
def lemmatize_lexicon(lexicon: frozenset) -> frozenset:
    """
    Returns a lemmatized copy of a lexicon frozenset.
    Multi-word phrases (with spaces) are lowercased as-is.
    Cached — each unique lexicon is lemmatized only once per process.
    """
    result = set()
    for entry in lexicon:
        if " " in entry:
            result.add(entry.lower())
        else:
            result.add(lemmatize_word(entry))
    return frozenset(result)
