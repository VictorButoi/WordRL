from . import module
from . import a2c_play
from . import a2c_train
from . import agent
from . import embeddingchars
from . import experience
from . import play
from . import sumchars

from .sumchars import SumChars
from .embeddingchars import EmbeddingChars

_registry = {}


def register(ctor, name):
    _registry[name] = ctor


def construct(name, **kwargs):
    return _registry[name](**kwargs)


register(SumChars, "SumChars")
register(EmbeddingChars, "EmbeddingChars")