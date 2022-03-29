import numpy as np
import glob

import spacy
import en_core_web_lg
import os
os.system(f"echo '🎉 All imports OK'")

paper_path = os.environ['PAPER_PATH']
os.system(f"echo '📄 PDF file located here: {paper_path}'")

model = en_core_web_lg.load()

print('SUCCESS!')

#!python -m spacy download en_core_web_lg

#model = spacy.load('en_core_web_lg')


