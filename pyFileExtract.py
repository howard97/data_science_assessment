from PyPDF2 import PdfFileReader, pdf
import textract

#nlp library
import nltk 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.chunk import tree2conlltags
from nltk.tag import pos_tag
import spacy
import pandas as pd
import collections



#import summarize from gensim
from gensim.summarization.summarizer import summarize
from gensim.summarization import keywords


#download 
nltk.download('maxent_ne_chunker')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('words')
nltk.download('maxent_ne_chunker')

nlp = spacy.load('en_core_web_sm')

#Create a pdf file reader object
pdfReader = PdfFileReader('SSW-IR20.pdf')

# Discerning the number of pages will allow us to parse through all the pages.
num_pages = pdfReader.numPages
count = 0
text = ""

# The while loop will read each page.
while count < num_pages:
    pageObj = pdfReader.getPage(count)
    count +=1
    text += pageObj.extractText()

# This if statement exists to check if the above library returned words. 
# It's done because PyPDF2 cannot read scanned files.
if text != "":
    text = text
# If the above returns as False, we run the OCR library textract to 
# convert scanned/image based PDF files into text.
else:
    text = textract.process('SSW-IR20.pdf', method='tesseract', language='eng')

#Summarize by half
text_summary= summarize(text,ratio=0.5)
#print(text_tokenize)

#Extract key words
print(keywords(text_summary, words=5))

# The word_tokenize() function will break our text phrases into individual words.
tokens = word_tokenize(text_summary)

# We'll create a new list that contains punctuation we wish to clean.
punctuations = ['(',')',';',':','[',']',',','%','.']

# We initialize the stopwords variable
# list of words like "The," "I," "and," etc. that don't hold much value as keywords.
stop_words = stopwords.words('english')

# List of words that are NOT IN stop_words and NOT IN punctuations.
keywords = [word for word in tokens if not word in stop_words and not word in punctuations]

#Word taging
pos_tags = pos_tag(keywords)
pos_tag


chunks = nltk.ne_chunk(pos_tags, binary=True) #either NE or not NE
for chunk in chunks:
    print(chunk)




entities =[]
labels =[]
for chunk in chunks:
    if hasattr(chunk,'label'):
        #print(chunk)
        entities.append(' '.join(c[0] for c in chunk))
        labels.append(chunk.label())
        
entities_labels = list(set(zip(entities, labels)))
entities_df = pd.DataFrame(entities_labels)
entities_df.columns = ["Entities","Labels"]
entities_df
print(entities_df)
#Count The most frequent words
print("20 most frewuent words are ",collections.Counter(keywords).most_common(20))

