import numpy as np
import glob

import spacy

import os
os.system(f"echo '🎉 All imports OK'")

paper_path = os.environ['PAPER_PATH']
os.system(f"echo '📄 PDF file located here: {paper_path}'")

os.system('python -m spacy download en_core_web_lg')

#model = en_core_web_lg.load()
model = spacy.load('en_core_web_lg')
print('SUCCESS!')

#!python -m spacy download en_core_web_lg

#model = spacy.load('en_core_web_lg')


