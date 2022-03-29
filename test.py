import numpy as np
import glob
import sklearn
import pdfminer
import nltk
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
install(spacy)
import spacy
import os
os.system(f"echo '🎉 All imports OK'")

paper_path = os.environ['PAPER_PATH']
os.system(f"echo '📄 PDF file located here: {paper_path}'")

#import spacy
#import nltk
#import pdfminer
#import sklearn

