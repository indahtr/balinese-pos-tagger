from .tagger import PosTagger

from .utils.text_utils import tokenize as word_tokenize

__version__ = "0.1.0"

__all__ = ["PosTagger", "word_tokenize"]