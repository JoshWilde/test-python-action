import numpy as np
import glob
#import spacy
import nltk
import pdfminer
import sklearn

import os
os.system(f"echo '🎉 All imports OK'")

paper_path = os.environ['PAPER_PATH']
os.system(f"echo '📄 PDF file located here: {paper_path}'")
