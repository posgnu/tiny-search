from bs4 import BeautifulSoup
import re
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import json
from nltk.tokenize import word_tokenize
import tqdm
from collections import defaultdict

filtered_list = ["https://cbcl.ics.uci.edu/public_data/DeepCons/Y_va_float32.npy",\
         "https://cbcl.ics.uci.edu/public_data/DeepCons/Y_te_float32.npy",\
            "https://cbcl.ics.uci.edu/public_data/DANN/data/ClinVar_ESP.y.npy"]
def tfidf_tokenizer(x):
        return stem(word_tokenize(x))

def get_tf_idf(file_list):
    vectorizer = TfidfVectorizer(tokenizer=tfidf_tokenizer)
    docs = []
    url_list = []

    for doc in tqdm.tqdm(file_list, desc='Building tfidf score matrix'):
        with open(doc, "r") as f:
            data = json.load(f)
            url = data["url"]
            if url in filtered_list:
                continue
            content = data["content"]
            encoding = data["encoding"]

            text = html2text(content)

            cleaned_text = clean_document(text)
            
            docs.append(cleaned_text)
            url_list.append(url)
                
    matrix = vectorizer.fit_transform(docs)

    df = pd.DataFrame.sparse.from_spmatrix(matrix, index=url_list, columns=vectorizer.get_feature_names_out())

    return df

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def stem(tokens):
    return [stemmer.stem(token) for token in tokens]

def lemmatize(tokens):
    return [lemmatizer.lemmatize(token) for token in tokens]

def clean_document(document):
    # Remove Unicode
    document_test = re.sub(r'[^\x00-\x7F]+', ' ', document)
    # Remove Mentions
    document_test = re.sub(r'@\w+', '', document_test)
    # Lowercase the document
    document_test = document_test.lower()
    # Remove short tokens
    document_test = re.sub(r'\W*\b\w{1,2}\b', '', document_test)
    # Remove punctuations
    # document_test = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', document_test)
    # Lowercase the numbers
    # document_test = re.sub(r'[0-9]', '', document_test)
    # Remove the doubled space
    # document_test = re.sub(r'\s{2,}', ' ', document_test)
    
    return document_test

def html2text(content):
    # https://stackoverflow.com/questions/328356/extracting-text-from-html-file-using-python

    soup = BeautifulSoup(content, 'html.parser')
    # kill all script and style elements
    for script in soup(["script", "style"]):
        script.extract()    # rip it out
    # get text
    text = soup.get_text()
    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # drop blank lines
    text = '\n'.join(chunk for chunk in chunks if chunk)

    return text

def get_important_word_dict(content):
    important_dict = defaultdict(list)
    soup = BeautifulSoup(content, 'html.parser')

    for title in soup.find_all('title'):
        stemmed_word_tokens = get_tokens(title.get_text())
        important_dict[1.5].append(stemmed_word_tokens)
    for data in soup.find_all("h1"):
        stemmed_word_tokens = get_tokens(data.get_text())
        important_dict[1.4].append(stemmed_word_tokens)
    for data in soup.find_all("h2"):
        stemmed_word_tokens = get_tokens(data.get_text())
        important_dict[1.3].append(stemmed_word_tokens)
    for data in soup.find_all("h3"):
        stemmed_word_tokens = get_tokens(data.get_text())
        important_dict[1.2].append(stemmed_word_tokens)
    
    return important_dict

def get_tokens(text):
    cleand = clean_document(text)
    word_tokens = word_tokenize(cleand)    
    stemmed_word_tokens = stem(word_tokens)
    return stemmed_word_tokens

def get_simhash_features(s):
    width = 3
    s = s.lower()
    s = re.sub(r'[^\w]+', '', s)
    return [s[i:i + width] for i in range(max(len(s) - width + 1, 1))]