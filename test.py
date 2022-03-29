import numpy as np
import glob
import sklearn
import pdfminer
import nltk
#import spacy
from pdfminer.high_level import extract_text
import os
os.system(f"echo '🎉 All imports OK'")

paper_path = os.environ['PAPER_PATH']
os.system(f"echo '📄 PDF file located here: {paper_path}'")

POI_PDF = [extract_text(paper_path)] # Extracts text from the PDF file
