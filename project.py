## TOPIC MODELING USING LSI AND LDA MODELS


import pandas as pd
import csv

# Set the CSV field size limit
csv.field_size_limit(1000000000)

# Read the CSV file into a Pandas DataFrame
df = pd.read_csv('state-of-the-union.csv')
df.head
df.columns=['year','text']
df['year']
# Group the DataFrame by year and concatenate all speeches for each year
df = df.groupby('year')['text'].apply(' '.join).reset_index()

# Rename columns
df.columns = ['year_of_the_speech', 'text_of_the_speech']

# Display the DataFrame
df

df['text_of_the_speech'].fillna("",inplace=True)
  
df['text_of_the_speech'][200]   
# Initialize NLTK stopwords 
import nltk
nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')
print(stopwords)
## Tokenize, lemmitize and stem 
import gensim
import re

from nltk.stem.wordnet import WordNetLemmatizer

def lemmatize_stemming(text):
    return (WordNetLemmatizer().lemmatize(text, pos='v'))# Tokenize and lemmatize
def preprocess(text):
    result=[]
    for token in gensim.utils.simple_preprocess(text) :      #tokenize
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
            
    return result
# Create a set of frequent words

stoplist = stopwords
text_corpus=df['text_of_the_speech']

# Lowercase each document, split it by white space and filter out stopwords
texts = [[re.sub(r"[0-9]+","",word) for word in  preprocess(document) if word not in stoplist]
         for document in text_corpus]

# Count word frequencies
from collections import defaultdict
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1

# Only keep words that appear more than once
processed_corpus = [[token for token in text if (frequency[token] > 1 )] for text in texts]
while('' in processed_corpus) : 
    processed_corpus.remove('') 
print(processed_corpus[0])

# Initialize a Gensim dictionary
from gensim import corpora
dictionary = corpora.Dictionary(processed_corpus)
print(dictionary)

dictionary.filter_extremes(no_below= 10, no_above=0.6)
print(dictionary)


# Create a bag-of-words corpus
bow_corpus = [dictionary.doc2bow(text) for text in processed_corpus]
print(bow_corpus[0])


## Covert to TFDF Weighted Vectors
from gensim import models
# Initialize the TF-IDF model
tfidf = models.TfidfModel(bow_corpus)

# Transform the corpus into TF-IDF weighted vectors
tfidf_corpus = tfidf[bow_corpus]
cnt=0
for vec in tfidf_corpus:
  print(vec)
  cnt+=1
  if(cnt>10):
    break

## Finding optimum number of topics using coherence scores
def compute_coherence_values(dictionary, doc_term_matrix, doc_clean, stop, start=2, step=3):
    """
    Input   : dictionary : Gensim dictionary
              corpus : Gensim corpus
              texts : List of input texts
              stop : Max num of topics
    purpose : Compute c_v coherence for various number of topics
    Output  : model_list : List of LSA topic models
              coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, stop, step):
        # generate LSA model
        model = LsiModel(doc_term_matrix, num_topics=num_topics, id2word = dictionary)  # train model
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=doc_clean, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
    return model_list, coherence_values
def plot_graph(doc_clean,start, stop, step):
    #dictionary,doc_term_matrix=prepare_corpus(doc_clean)
    doc_term_matrix=tfidf_corpus
    doc_clean=processed_corpus
    model_list, coherence_values = compute_coherence_values(dictionary, doc_term_matrix,doc_clean,
                                                            stop, start, step)
    # Show graph
    x = range(start, stop, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.show()
from gensim.models import LsiModel
import matplotlib.pyplot as plt

from gensim.models.coherencemodel import CoherenceModel
start,stop,step=10,50,5
plot_graph(processed_corpus,start,stop,step)
## Application of LSI Model
# LSI MODEL

# Specify the number of topics to generate 
num_topics = 25

# Create an LSI model from the TF-IDF vectors
lsi_model = models.LsiModel(tfidf_corpus, id2word=dictionary, num_topics=num_topics)

# create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi
corpus_lsi = lsi_model[tfidf_corpus] 



cnt=0
print("format")
print("topic followed by its proportion in the document -- >  a vector having dimensions equal to number of topics")
print("text of the speech")
for vec,doc in zip(corpus_lsi,text_corpus):
  print(vec,doc)
  cnt+=1
  if(cnt>2):
    break
     

# Get the first 25 topics

print("LSI Model:")
 
for idx in range(25):
    # Print the first 25 most representative topics
    print("Topic #%s:" % idx, lsi_model.print_topic(idx, 20))
    print()
# Sample LSI output for top words in topics
lsi_topics = lsi_model.print_topics(num_topics=25,num_words=25)

# Function to extract topic labels
def get_topic_label(topic_words):
    # Split the topic_words string into individual words
    words = topic_words.split('" + ')
    
    # Extract meaningful words (those with non-zero probabilities)
    meaningful_words = [word.split('*"')[1] for word in words if float(word.split('*"')[0]) > 0]
    
    # Combine meaningful words to create a label or description for the topic
    topic_label = ', '.join(meaningful_words)
    
    return topic_label

# Create a dictionary to store topic labels
topic_labels = {}

# Iterate through the LSI topics and extract labels
for topic_id, topic_words in lsi_topics:
    label = get_topic_label(topic_words)
    topic_labels[topic_id] = label

# Print the topic labels
for topic_id, label in topic_labels.items():
    print(f"Topic {topic_id}: {label}")
## Find document having max proportion of a given topic
# Define the topic of interest 
desired_topic = 8

# Initialize variables to keep track of the maximum proportion and corresponding year
max_proportion = -1  # Initialize to a value lower than any possible proportion
year_with_max_proportion = None

# Iterate through the corpus and find the document with the highest proportion of the desired topic
for idx, doc in enumerate(corpus_lsi):
    for topic in doc:
        topic_id, proportion = topic
        if topic_id == desired_topic and proportion > max_proportion:
            max_proportion = proportion
            year_with_max_proportion = df.iloc[idx]['year_of_the_speech']

# Print the result
print(f"Year with the highest proportion of Topic #{desired_topic}: {year_with_max_proportion} (Proportion: {max_proportion})")
# Print the speech of the particular year
year = 1897
speech = df[df['year_of_the_speech'] == year]['text_of_the_speech'].values[0]
print(speech)
## LSI Word Cloud
import matplotlib.pyplot as plt
from wordcloud import WordCloud 

lsi_topics = lsi_model.print_topics(num_topics = 25, num_words = 100 )
f, axes_arr = plt.subplots(5, 5, sharex='col', sharey='row', figsize=(20,20))

for i in range(25):
    dicti = {}
    for temp in lsi_topics[i][1].split(" ")[1:]:
        if temp != '+':
            dicti[temp.split('*')[1]] = (float)(temp.split('*')[0])
    wordcloud = WordCloud( max_words=5000, contour_width=3, contour_color='steelblue')# Generate a word cloud
    wordcloud.generate_from_frequencies(dicti)# Visualize the word cloud
    axes_arr[i//5][i%5].imshow(wordcloud)
    axes_arr[i//5][i%5].set_title('Topic ' + str(i), fontdict=dict(size=16))
    plt.axis('off')
## LDA Topic Modeling

# LDA Model

# Specify the number of topics to generate (you should experiment with this)
num_topics = 25

# Create an LDA model from the TF-IDF vectors
lda_model = models.LdaModel(corpus=tfidf_corpus, id2word=dictionary, num_topics=num_topics)

corpus_lda = lda_model[tfidf_corpus]

 
# Print the first 25 topics
for idx in range(25):
    # Print the first 10 most representative topics
    print("Topic #%s:" % idx, lda_model.print_topic(idx, 20))
    print()
# Sample Lda output for top words in topics
lda_topics = lda_model.print_topics(num_topics=25,num_words=25)

# Function to extract topic labels
def get_topic_label(topic_words):
    # Split the topic_words string into individual words
    words = topic_words.split('" + ')
    
    # Extract meaningful words (those with non-zero probabilities)
    meaningful_words = [word.split('*"')[1] for word in words if float(word.split('*"')[0]) > 0]
    
    # Combine meaningful words to create a label or description for the topic
    topic_label = ', '.join(meaningful_words)
    
    return topic_label

# Create a dictionary to store topic labels
topic_labels = {}

# Iterate through the LSI topics and extract labels
for topic_id, topic_words in lda_topics:
    label = get_topic_label(topic_words)
    topic_labels[topic_id] = label

# Print the topic labels
for topic_id, label in topic_labels.items():
    print(f"Topic {topic_id}: {label}")
    print("")
    
## Find document having max proportion of a given topic
# Define the topic of interest 
desired_topic = 9

# Initialize variables to keep track of the maximum proportion and corresponding year
max_proportion = -1  # Initialize to a value lower than any possible proportion
year_with_max_proportion = None

# Iterate through the corpus and find the document with the highest proportion of the desired topic
for idx, doc in enumerate(corpus_lda):
    for topic in doc:
        topic_id, proportion = topic
        if topic_id == desired_topic and proportion > max_proportion:
            max_proportion = proportion
            year_with_max_proportion = df.iloc[idx]['year_of_the_speech']

# Print the result
print(f"Year with the highest proportion of Topic #{desired_topic}: {year_with_max_proportion} (Proportion: {max_proportion})")
# Print the speech of the particular year
year = 1916
speech = df[df['year_of_the_speech'] == year]['text_of_the_speech'].values[0]
print(speech)
## Word Cloud for LDA Model
import matplotlib.pyplot as plt
from wordcloud import WordCloud 

lda_topics = lda_model.print_topics(num_topics = 25, num_words = 100 )
f, axes_arr = plt.subplots(5, 5, sharex='col', sharey='row', figsize=(20,20))

for i in range(25):
    dicti = {}
    for temp in lda_topics[i][1].split(" ")[1:]:
        if temp != '+':
            dicti[temp.split('*')[1]] = (float)(temp.split('*')[0])
    wordcloud = WordCloud( max_words=5000, contour_width=3, contour_color='steelblue')# Generate a word cloud
    wordcloud.generate_from_frequencies(dicti)# Visualize the word cloud
    axes_arr[i//5][i%5].imshow(wordcloud)
    axes_arr[i//5][i%5].set_title('Topic ' + str(i), fontdict=dict(size=16))
    plt.axis('off')
# Decade Summarization Algorithm

print(df['text_of_the_speech'][217])
#Decade Summarization 
decade_dict = {}

decades = ['1901-1910', '1911-1920', '1921-1930', '1931-1940', '1941-1950', '1951-1960', '1961-1970', '1971-1980', '1981-1990', '1991-2000', '2001-2010', '2011-2020']
start = 0  # Start from the first row

for cnt in range(12):
    end = start + 10
    if end >= len(df):  # Check if the end index is within the DataFrame's length
        end = len(df) - 1

    decade_data = " "
    for i in range(start, end + 1):
        decade_data = decade_data + df['text_of_the_speech'][i]

    start = end + 1  # Move to the next decade

    print(decades[cnt])
    decade_dict[decades[cnt]] = decade_data
#Prepare text to apply lda on this already trained on this corpus
print(decade_dict.keys())
# Create a set of frequent words

stoplist = stopwords
#text_corpus=df['text_of_the_speech']


# Lowercase each document, split it by white space and filter out stopwords
texts = [[re.sub(r"[0-9]+","",word) for word in  preprocess(document) if word not in stoplist]
         for document in decade_dict.values()]

# Count word frequencies
from collections import defaultdict
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1

# Only keep words that appear more than once
processed_corpus_decade = [[token for token in text if (frequency[token] > 1 )] for text in texts]
while('' in processed_corpus_decade) : 
    processed_corpus_decade.remove('') 
print(processed_corpus_decade[0])

from gensim import corpora
from gensim import models

dictionary_decade = corpora.Dictionary(processed_corpus_decade)
print(dictionary_decade)

from gensim import corpora
from gensim import models

dictionary_decade = corpora.Dictionary(processed_corpus_decade)
print(dictionary_decade)

dictionary_decade.filter_extremes(no_below= 3, no_above=0.9)
print(dictionary_decade)


bow_corpus_decade = [dictionary_decade.doc2bow(text) for text in processed_corpus_decade]
print(bow_corpus_decade[0])
 
tfidf_decade = models.TfidfModel(bow_corpus_decade)

tfidf_corpus_decade=tfidf_decade[bow_corpus_decade]
lda_model_decade = models.LdaModel(tfidf_corpus_decade, id2word=dictionary_decade, num_topics=15,minimum_probability=0.0)
corpus_lda_decade=lda_model_decade[tfidf_corpus_decade]
print(corpus_lda_decade[2])

lda_model_decade.print_topics(num_topics=15,num_words=25)
## Topic Annotations
Topic 0: "International Relations"
Topic 1: "Economic Matters"
Topic 2: "Military Engagements"
Topic 3: "Territorial Disputes"
Topic 4: "Government and Coinage"
Topic 5: "International Diplomacy"
Topic 6: "Colonial Affairs"
Topic 7: "Slavery and Territories"
Topic 8: "Barbary Wars"
Topic 9: "Currency and Economic Issues"
Topic 10: "Colonial Policies"
Topic 11: "Industrial and Economic Development"
Topic 12: "Economic Challenges"
Topic 13: "Corporations and Labor"
Topic 14: "Labor and Corporate Policies"


for i in range(12):
  print(decades[i],corpus_lda_decade[i]) 
  print()
  print()

import matplotlib.pyplot as plt
import numpy as np

# Define the decades and topic proportions 
decades = [
    "1901-1910", "1911-1920", "1921-1930", "1931-1940", "1941-1950",
    "1951-1960", "1961-1970", "1971-1980", "1981-1990", "1991-2000",
    "2001-2010", "2011-2020"
]

topic_proportions = [
    [0.007229684, 0.007229683, 0.0072297016, 0.007229684, 0.89878446, 0.007229684, 0.0072296876, 0.007229683, 0.0072296895, 0.007229683, 0.007229683, 0.0072296937, 0.007229683, 0.0072296862, 0.007229685],
    [0.008877847, 0.008877846, 0.008877874, 0.008877848, 0.008877864, 0.008877847, 0.008877855, 0.008877847, 0.87571007, 0.008877846, 0.008877846, 0.00887787, 0.008877846, 0.008877855, 0.008877852],
    [0.0086809825, 0.008680982, 0.8784658, 0.008681023, 0.008681031, 0.0086809825, 0.008681008, 0.0086809825, 0.008681017, 0.0086809825, 0.0086809825, 0.008681264, 0.008680982, 0.008681013, 0.008680989],
    [0.008081599, 0.008081598, 0.008081631, 0.008081602, 0.008081648, 0.0080816, 0.76513666, 0.008081599, 0.0080816215, 0.008081599, 0.008081599, 0.12980248, 0.008081598, 0.008081615, 0.008081604],
    [0.0070861904, 0.00708619, 0.0070862025, 0.007086198, 0.0070862286, 0.0070861904, 0.0070862295, 0.0070861904, 0.007086204, 0.0070861904, 0.0070861904, 0.90079325, 0.00708619, 0.0070862016, 0.007086196],
    [0.0066170143, 0.006617014, 0.006617038, 0.006617015, 0.0066170366, 0.0066170143, 0.0066170213, 0.0066170143, 0.006617111, 0.006617014, 0.006617014, 0.9073616, 0.006617014, 0.006617032, 0.006617019],
    [0.0073264027, 0.0073264022, 0.0073264153, 0.0073264036, 0.0073264632, 0.0073264036, 0.0073264088, 0.0073264027, 0.0073264157, 0.0073264027, 0.0073264027, 0.3719266, 0.0073264022, 0.53283006, 0.0073264036],
    [0.0054461546, 0.005446154, 0.0054461597, 0.0054461546, 0.0054461793, 0.0054461556, 0.0054461607, 0.0054461546, 0.005446158, 0.0054461546, 0.005446154, 0.92375374, 0.005446154, 0.005446187, 0.0054461556],
    [0.005237913, 0.0052379114, 0.0052379156, 0.005237912, 0.0052379374, 0.0052379146, 0.005237914, 0.0052379114, 0.0052379156, 0.0052379114, 0.0052379114, 0.9266692, 0.0052379114, 0.005237918, 0.005237913],
    [0.0065567265, 0.006556725, 0.006556731, 0.006556726, 0.13585724, 0.006556731, 0.006556731, 0.006556726, 0.00655673, 0.006556726, 0.006556725, 0.7789053, 0.006556725, 0.006556744, 0.0065567275],
    [0.008212665, 0.008212664, 0.008212666, 0.008212665, 0.0082126865, 0.008212667, 0.008212668, 0.008212665, 0.00821267, 0.008212664, 0.008212664, 0.8850225, 0.008212664, 0.008212751, 0.008212715],
    [0.0054179714, 0.0054179714, 0.005417972, 0.0054179714, 0.0054180245, 0.005417972, 0.005417972, 0.0054179714, 0.005417972, 0.0054179714, 0.0054179714, 0.9241483, 0.0054179714, 0.005417978, 0.005417974]
]

# Define the number of topics
num_topics = len(topic_proportions[0])

# Transpose the data for plotting
topic_proportions = np.array(topic_proportions).T.tolist()

# Create a color map for topics
color_map = plt.cm.get_cmap("tab20", num_topics)

# Plot the data
fig, ax = plt.subplots(figsize=(12, 8))
bottom = np.zeros(len(decades))
for i, topic_proportion in enumerate(topic_proportions):
    ax.bar(decades, topic_proportion, label=f"Topic {i}", color=color_map(i), bottom=bottom)
    bottom += np.array(topic_proportion)

# Add labels and legend
ax.set_xlabel("Decade")
ax.set_ylabel("Proportion")
ax.set_title("Proportion of Topics in Each Decade")
ax.legend(title="Topics", bbox_to_anchor=(1.05, 1), loc='upper left')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha="right")

# Show the plot
plt.tight_layout()
plt.show()
jupyter nbconvert --to pdf notebook.ipynb
!pip install amsmath
