import numpy as np
import glob
import sklearn
import pdfminer
import pdfplumber
import PyPDF2
import nltk

from nltk import word_tokenize
from nltk import download
from nltk.corpus import stopwords
from pdfminer.high_level import extract_text
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

#import spacy
import os
os.system(f"echo '🎉 All imports OK'")

paper_path = os.environ['PAPER_PATH']
os.system(f"echo '📄 PDF file located here: {paper_path}'")

paper_path = 'paper.pdf'
#POI_PDF = [extract_text(paper_path)] # Extracts text from the PDF file

def Get_Lemma_Words(POI_PDF):
    ''' 
    Parameters
    ----------
        POI_PDF : list
            A list containing a single string which is the contents of the paper
            
    Returns
    ----------
        words : array_like
            An array where each element is a processed word from the text
    ''' 
    text = str(POI_PDF)
    text2 = text.split() # splits the text into words
    words_no_punc = [] # defines an empty list

    for w in text2: # For each word in the text
        if w.isalpha(): # If the word is an alphanumberic value
            words_no_punc.append(w.lower()) # appends a lowercase version of the word to the no punctionation list
    from nltk.corpus import stopwords # Import stop words
    stopwords = stopwords.words('english')  # Defines english stop words
    clean_words = [] # define clean word list
    for w in words_no_punc: # for each word in no punctionation list
        if w not in stopwords: # if the word is not a stopword
            clean_words.append(w) # if the word is not a stopword it is appended to the clean word list
    clean_words_arr = '' # Defines an empty string
    for i in range(len(clean_words)): # For each word in clean words
        clean_words_arr = clean_words_arr + ' ' + str(clean_words[i]) # Appends the clean words to a string

    string_for_lemmatizing = clean_words_arr 
    lemmatizer = WordNetLemmatizer() 
    words_2 = word_tokenize(string_for_lemmatizing)
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words_2]

    lemmatized_words_arr = '' #  Defines an empty string
    for i in range(len(lemmatized_words)):  # For each word iin lemmanised words
        lemmatized_words_arr = lemmatized_words_arr + ' ' + str(lemmatized_words[i]) # Appends the lemmanised words to a string
    words = word_tokenize(lemmatized_words_arr) # Tokenises each word in the text
    return words



def Get_Top_Words_tf(Paper_interest, df=1, num_top20=20):
    ''' 
    Parameters
    ----------
        Paper_interest : string
            File path to the location of the PDF 
            
        df : 
            A
            
        num_top20 : Int
            Number of most frequent words that are used for calculating the vector of the paper
            
    Returns
    ----------
        top20_tf : array_like
            Array of the most frequent words from the paper in order
    ''' 
    POI_PDF = [extract_text(Paper_interest)] # Extracts text from the PDF file
    #text = str(POI_PDF)
    words =  Get_Lemma_Words(POI_PDF) # Lemmanises words from the extracted text
    top20_tf = -2 # If there are no lemmanised words, this function will output this value
    if len(words) > 0: # If there are lemmanised words
        fdist = FreqDist(words) # Calculates the frequency for each lemmanised word in the text
        X = np.array(fdist.most_common()) # Sorts the words in order of frequency
        top20_tf = X[:num_top20,0] # Saves the top N words as a list

    return top20_tf

top20_tf = Get_Top_Words_tf(paper_path)

#words = Get_Lemma_Words(POI_PDF) # Lemmanises words from the extracted text
#top20_tf = -2 # If there are no lemmanised words, this function will output this value
#if len(words) > 0: # If there are lemmanised words
#        fdist = FreqDist(words) # Calculates the frequency for each lemmanised word in the text
#        X = np.array(fdist.most_common()) # Sorts the words in order of frequency
#        top20_tf = X[:num_top20,0] # Saves the top N words as a list
print(top20_tf)    
