import numpy as np
import glob

import spacy

import os
os.system(f"echo '🎉 All imports OK'")

fold_loc = 'https://github.com/JoshWilde/test-python-action/tree/main/Author_Folders'
print(fold_loc)

folds = glob.glob(fold_loc)
print(folds)

folds = glob.glob(fold_loc+'/*')
print(folds)


#paper_path = os.environ['PAPER_PATH']
#os.system(f"echo '📄 PDF file located here: {paper_path}'")

#os.system('python -m spacy download en_core_web_lg')

#model = en_core_web_lg.load()
#model = spacy.load('en_core_web_lg')
print('SUCCESS!')

#!python -m spacy download en_core_web_lg

#model = spacy.load('en_core_web_lg')


