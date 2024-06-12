from flair.embeddings import FlairEmbeddings
import math
import io

# get language model
language_model = FlairEmbeddings('news-forward').lm