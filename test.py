import numpy as np
import glob

import spacy

import os
os.system(f"echo '🎉 All imports OK'")

glow = os.environ['GLOB_FOLDERS']
print(glow)
print('/n')
pdfs = os.environ['GLOB_PDFS']
print(pdfs)

def Make_Folder_dict(pdf):
  Master_dict = {}
  for i in range(len(pdfs)):
    J = re.search('/', pdfs[i])
    K = re.search('/',pdfs[i][J.end():])
    Folder_name = pdfs[i][J.end():J.end()+K.start()]
    pdf_name = pdfs[i]#[J.end()+K.end():]
    if Folder_name not in Master_dict:
      Master_dict[Folder_name] = [pdf_name]
    else:
      Master_dict[Folder_name].append(pdf_name)
  return Master_dict



#fold_loc = 'https://github.com/JoshWilde/test-python-action/tree/main/Author_Folders/ctb'
#print(fold_loc)

#folds = glob.glob(fold_loc)
#print(folds)

#folds = glob.glob(fold_loc+'/*')
#print(folds)

#paper_path = os.environ['FOLDER_PATH']
#print(paper_path)

#glo = os.environ[glob.glob(paper_path+'/*')]
#print(glo)

#paper_path = os.environ['PAPER_PATH']
#os.system(f"echo '📄 PDF file located here: {paper_path}'")

#os.system('python -m spacy download en_core_web_lg')

#model = en_core_web_lg.load()
#model = spacy.load('en_core_web_lg')
print('SUCCESS!')

#!python -m spacy download en_core_web_lg

#model = spacy.load('en_core_web_lg')


