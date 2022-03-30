import numpy as np
import glob
import sklearn
import pdfminer
import pdfplumber
import PyPDF2
import nltk
import spacy

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

#paper_path = os.environ['PAPER_PATH']
#os.system(f"echo '📄 PDF file located here: {paper_path}'")


folder_names = os.environ['FOLDER_PATH']
os.system(f"echo '📄 PDF file located here: {folder_names}'")

os.system('python -m spacy download en_core_web_lg')
model = spacy.load('en_core_web_lg')

#paper_path = 'paper.pdf'
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
    #text = str(POI_PDF)
    words =  Get_Lemma_Words(POI_PDF) # Lemmanises words from the extracted text
    top20_tf = -2 # If there are no lemmanised words, this function will output this value
    if len(words) > 0: # If there are lemmanised words
        fdist = FreqDist(words) # Calculates the frequency for each lemmanised word in the text
        X = np.array(fdist.most_common()) # Sorts the words in order of frequency
        top20_tf = X[:num_top20,0] # Saves the top N words as a list

    return top20_tf



def Reviewer_Paper_Vector(paper_list, model, num_top20=20):
    ''' 
    Parameters
    ----------
        Paper_interest : string
            File path to the location of the PDF 
            
        model : 
            A model that can generate a vector representation of the paper.
            
        df : Dictionary
            A 
            
        get_word_fun : 
            A function that will get the top N words, these could be defined as the 
            most frequent words, the highest score words in terms of TF-IDF, or any
            user defined function
            
        num_top20 : int
            A
                 
    Returns
    ----------
        average_vector : array_like
            A
            
    ''' 
    average_vector = np.zeros((300)) # Creates an array for 300 zeros
    mod = 0 # Keeps track of papers that do not add information to the average_vector
    for i in range(len(paper_list)): # For each paper in the list
        Paper_interest = paper_list[i] # Gets a paper path
        top20_tf = Get_Top_Words_tf(Paper_interest, num_top20) # Generates the top N words for a paper
        doc_top20= ''  # Creates empty string
        if top20_tf != -2: # If the paper has lemmanised words
            for i in top20_tf: # For each word in the top N
                doc_top20 = doc_top20 + i +' ' # Append the top N words to a list
            pap_vector = model(doc_top20).vector # Generates the vector for a paper
            average_vector = average_vector + pap_vector # adds the result to the average
        else:
            mod = mod +1 # Adds a value indicating that the paper had no words, hence did not add
            # any information to the average_vector
            
    diver = len(paper_list)-mod # subtracks the modification from the paper list
    if diver ==0: # If no papers added to the average_vector
        diver = 1 # Updates the division to 1, to avoid an error
    average_vector = average_vector/diver # Average vector divided by papers that added information
    
    return average_vector


# Generate TF Vectors Author
def Author_vectors_TF(folder_names, model, num_top20=20, directory_offset=21):
    ''' 
    Parameters
    ----------
        folder_names : array_like
            Array of folder paths, each folder should be the name of the author and should contain their papers in 
            PDF format. 
            
        gen_ave_vec : function
            A function to generate the vectors for paper that we are trying to find a reviewer for
        
        directory_offset : int
            A value that clips the file path to ensure that the keys for the author name will only contain the 
            author name.
                 
    Returns
    ----------
        Author_Dict : Dictionary
            A dictionary of vectors for each author. The keys are the names of the folders. The items are vectors
            of shape (300) which is the average vector for each authors work.
            
    ''' 
    Author_Dict = {} # Defines an empty dictionary
    for k in range(len(folder_names)): # For each author
        #print(folder_names[k][directory_offset:]+ ' - ' +str(k))
        paper_list = glob.glob(folder_names[k]+'/*.pdf') # Finds all PDF files in this folder
        print(paper_list)
        average_vector = Reviewer_Paper_Vector(paper_list, model, num_top20) # Generates the average vector for all the papers in this folder
        Author_Dict[folder_names[k][directory_offset:]] = average_vector # Adds this average vector to the dictionary
    return Author_Dict
  
print('folder names')
print(folder_names)
folds = glob.glob(folder_names+'/*')
print(folds)
Author_Dict = Author_vectors_TF(folds, model)

print(Author_Dict)
np.save('Author_Dict_generated.npy', Author_Dict)
print('SUCCESS!!!')
