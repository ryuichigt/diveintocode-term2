import numpy as np
from common.util import preprocess

import sys
sys.path.append("..")


text = "You say goodnight amd I say hello"
corpus, word_to_id, id_to_word = preprocess(text)
