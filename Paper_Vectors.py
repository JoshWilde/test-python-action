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

import spacy
import os
os.system(f"echo '🎉 All imports OK'")

paper_path = os.environ['PAPER_PATH']
os.system(f"echo '📄 PDF file located here: {paper_path}'")

os.system('python -m spacy download en_core_web_lg')
model = spacy.load('en_core_web_lg')

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



def Get_Top_Words_tf(Paper_interest, num_top20=20):
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
    #print('Extracted text')
    #text = str(POI_PDF)
    words =  Get_Lemma_Words(POI_PDF) # Lemmanises words from the extracted text
    #print('Get Lemma Words')
    top20_tf = -2 # If there are no lemmanised words, this function will output this value
    #print('Top20 TF')
    if len(words) > 0: # If there are lemmanised words
        fdist = FreqDist(words) # Calculates the frequency for each lemmanised word in the text
        #print('Freq Dist')
        X = np.array(fdist.most_common()) # Sorts the words in order of frequency
        #print('X')
        top20_tf = X[:num_top20,0] # Saves the top N words as a list
        #print('Top20 TF')

    return top20_tf

#top20_tf = Get_Top_Words_tf(paper_path)

#words = Get_Lemma_Words(POI_PDF) # Lemmanises words from the extracted text
#top20_tf = -2 # If there are no lemmanised words, this function will output this value
#if len(words) > 0: # If there are lemmanised words
#        fdist = FreqDist(words) # Calculates the frequency for each lemmanised word in the text
#        X = np.array(fdist.most_common()) # Sorts the words in order of frequency
#        top20_tf = X[:num_top20,0] # Saves the top N words as a list
#print(top20_tf)    


def Generate_Paper_Vector(Paper_interest, model, num_top20=20):
    ''' 
    Parameters
    ----------
        Paper_interest : string
            File path to the location of the PDF 
            
        model :
            A
        
        df : Dictionary
            A
            
        get_word_fun : function
            A
        
        num_top20 : int
            A
            
    Returns
    ----------
        pap_vector : array_like
            An array of shape (300) representing where the given paper lies in the
            model vector space.
            
        doc_top20 : string
            A string containing the 20 words that were 
        
    ''' 
    #average_vector = np.zeros((300)) # Creates an array for 300 zeros
    #print('Starting Top Words TF')
    top20_tf = Get_Top_Words_tf(Paper_interest) # Gets the top N Words
    #print(top20_tf)
    #print(top20_tf)
    doc_top20= '' # Creates empty string
    if top20_tf != -2: # If the paper has lemmanised words
        for i in top20_tf: # For each word in the top N
                doc_top20 = doc_top20 + i +' ' # Appends each top N word to list
    pap_vector = model(doc_top20).vector # generates a vector for the paper
    #average_vector = average_vector + pap_vector 
    
    return pap_vector, doc_top20

# Generate TF Vectors Paper
def Paper_vectors_TF(paper_list, model,num_top20=20):
    ''' 
    Parameters
    ----------
        paper_list : array_like
            Array of file paths to PDF files
            
        gen_pap_vec : function
            A function to generate the vectors for paper that we are trying to find a reviewer for
            
    Returns
    ----------
        Paper_Dict : Dictionary
            All the keys should be the DOI numbers for each paper taken from the file name. The items are vectors
            of shape (300) which is the vector for where this paper lies in the model vector space.
            
        Paper_20_Dict : Dictionary
            All the keys should be the DOI numbers for each paper taken from the file name. The items are the 
            top 20 words from the paper that have been used to generate the vector representation.
    ''' 
    Paper_Dict = {}  # Defines an empty dictionary
    Paper_20_Dict = {}  # Defines an empty dictionary
    #for k in range(len(paper_list)): # For each paper
    #print(paper_list[k]+ ' - ' +str(k))
    #print('Starting Generate Paper Vectors')
    paper_vector, doc_top20 = Generate_Paper_Vector(paper_list, model) # Generates paper vector and shows the top N words
    #print(paper_vector)
   # print(doc_top20)
    Paper_Dict[paper_list] = paper_vector # Adds this vector to the dictionary
    Paper_20_Dict[paper_list] = doc_top20 # Adds the top N words to the dictionary
   # print(Paper_Dict)
    #print(Paper_20_Dict)
    return Paper_Dict, Paper_20_Dict

# Paper Cosine
def Paper_cosine(author_keys, author_vectors, paper_vec, N=5, printer=True):
    ''' 
    Parameters
    ----------
        author_keys : Dictionary Keys
            A
            
        author_vectors : Dictionary
            A
             
        paper_vec : array-like
            A
            
        N : int
            Number of reviewers suggested 
            
        printer : Boolean
            A
            
    Returns
    ----------
        cos_sim_dict : dictionary
            A
    ''' 
    cos_sim_list = [] # Creates an empty list
    for i in range(len(author_keys)): # For each author key
        idx = list(author_keys)[i] # Creates an index
        author_vec = author_vectors[idx] # Loads the vector for the given author key
    
        cos_sim = cosine_similarity(np.array([paper_vec]), np.array([author_vec]))[0,0] # Calculates the cosine similarity 
        # of the paper and the author of the index
        cos_sim_list.append(cos_sim) # appends cosine similarity to a list of all cosine similarities for each author for 
        # this one paper
    cos_sim_list = np.array(cos_sim_list) # Converts list to numpy array

    cos_sim_dict = {} # Creates an empty dictionary
    sorted_idx = np.argsort(cos_sim_list)[-N:] # Sorts list and selects the top N highest scoring authors
    for i in range(N): # for each of the top N authors
        idx = sorted_idx[-i-1] # Creates an index
        doi = list(author_vectors)[idx] #Finds the author key for the high scoring cosine similarties
        if printer == True:
            print(doi + ' - ' + str(cos_sim_list[idx])[:6]) # Prints author key & cosine similarity for that author to the given paper
        cos_sim_dict[doi] = cos_sim_list[idx] # Adds the author key & cosine similarity to a dictionary
    #return cos_sim_dict

#paper_vector, doc_top20 = Generate_Paper_Vector(paper_path , model)
#print(paper_vector)
#print(doc_top20)
#paper_path = ['paper.pdf', 'paper.pdf', 'paper.pdf']

print('Starting Paper Vectors TF')
Paper_Dict, Paper_20_Dict = Paper_vectors_TF(paper_path, model)

author_Dict = np.load('Reviewers_Idea3_TF_Dict.npy', allow_pickle=True).item()
author_keys = author_Dict.keys()
author_vectors = author_Dict
paper_vec = Paper_Dict[list(Paper_Dict)[0]]

Paper_cosine(author_keys, author_vectors, Paper_Dict)

print('SUCCESS!!!')

