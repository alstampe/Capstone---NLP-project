Machine Learning Engineer Nanodegree
## Capstone Project
Anne Line Stampe DNB
December 8th, 2018

## I. Definition

## Project Overview
Natural Language Processing (NLP) is a higly relevant topic in many business areas, and specifically so within DNB (where I work), Norways largest financial institution. The need for rapidly increasing levels of automation and digitalization spurs a demand for ways to reengineer processes based on manual handling of unstructured data. In addition to reduce costs and risk there is a clearly stated ambition to gain and use data-driven insight from these initiatives. 

With a background from participation in OCR-project piloting recently I chose a NLP-case as my Capstone project, uniting freshy aquired skills and an interest in real-life document analysis. 

NLP was briefly introduced in the Nanodegree course

My focus is to build skills and insights on possible solutions for analysis of text in form of full-scale documents rather than emails, tweets or other short, informal text objects. The data used in the project is text from 41 full books from the GUtenmebrg project. For test purposes an additional unknown book will be used at the end of the project.  

NLP covers a wide variety of methods, tools and theories.

As described in the project proposal my project addresses three challenges;

- Analysis of words in a full corpus; vocabulary, density, vector representation etc
- Methods for comparing individual text objects (books) for similarity based on vector representation
- Methods for topic extraction from text objects, on a book level

In addition to these three specific issues, I want to gain more insight into the NLP field in general from the project.

The project has a high relevance for my job; we want to be able to understand documents by assessing vocabulary, comparing documents and extracting topics with algorithms as an alterative to humans reading and assessing the texts. There is also a need to be able to isolate specific data elements from documents (names, account numbers, values etc), but this is not a part of the Capstone scope. 
I plan to try out the project code on internal documents later. 


In this section, look to provide a high-level overview of the project in layman’s terms. Questions to ask yourself when writing this section:
Has an overview of the project been provided, such as the problem domain, project origin, and related datasets or input data?
Has enough background information been given so that an uninformed reader would understand the problem domain and following problem statement?

### Problem Statement
The underlying problem I want to addreess is the challenge we face when automating our prorcesses in the bank; some tasks are by nature based on non-numerical data and the understanding of 'real languiage'. We still have to maintain staff for digesting large volumes of information manually with related cost and risk of human errors. There is a need to speed up the process and digitize results to be able to make data a part of a digital process. Harvest reusable insight from extracted information as a by-product. Fundamentally, the initiative can also add traction to the collaborative effort of transforming the bank to a more tecnho-driven institution and illustrate opportunities in the field of machine learning and AI.     

Exploring nlp opportunities does not imply that all hunam handling of text is expandable, there are numerous situations where we must have people rading and talking, but it is a start on a journey to a more digital bank.  

#### Compare and analyse documents, ectract essence. 
I go into the project with a moderate ambition of precision from the document analysis, but expect to be able to confirm the hypothesis stated in the three questions. For such a complicated field, a modest proof of concept is a good basis for believing in results from a possible full-scale project   

### Strategy for the work

All text analysis, as other ML tasks, starts with understanding and preprocessing the data.

#### Building more relevant skills

Before starting to code, I needed some more nlp-specific knowledge. I have used the Kindle book 'Text Analysis -- * which I spent some time reading before coding.  

![screenshot 7](https://user-images.githubusercontent.com/42574791/49142891-17369900-f2fa-11e8-9f87-83cfd2cf201f.png)


#### Gathering and loading input data.
Assessing data - this is all about finding, loading and understanding the input and from this, the need of preprocessing.
Data source here is solely books from project Gutenberg.  

#### Preprocessing
Preprocessing text data is partly similar to celaning numerical and categorical data, partly very different. The options and choices are many, hence this step is impirtant and rquires some time.  

#### Transformation, training and produce results
Transforming the text data by vectorization, training the model and producing results will be more straightforward, but steps and methos must be appropriate to the case. 

#### Inspect and assess
Results from transformation of text can be visualized in addition to list-based output.


#### Test of an 'unknown' book
Finally I will do a test on an ebook which is unknown and new. The test will be of the Topic extraction. 

#### Evaluation
Results will be discussed. 

#### Deliverables
Project deliverables will be in form of results described in the repost and code imbedded in the project Notebook. 

The Notebook is quite extensive as it includes all code used for testing different ways to preprocess the text, not all are used for the final solution but has been important to learn more about the many ways to handle text data. 

template q:
In this section, you will want to clearly define the problem that you are trying to solve, including the strategy (outline of tasks) you will use to achieve the desired solution. You should also thoroughly discuss what the intended solution will be for this problem. Questions to ask yourself when writing this section:
Is the problem statement clearly defined? Will the reader understand what you are expecting to solve?
Have you thoroughly discussed how you will attempt to solve the problem?
Is an anticipated solution clearly defined? Will the reader understand what results you are looking for?



### Metrics

As described in the capstone project proposal, the three parts of my project are all of a nature where sharp metrics are not always present. The individual tasks may produce some comparable values, but more important is the subjective opinion ie: 

1) Similarity - words
 - 'Does the word pairing examples on the vectorized data mase sense?'

2) Similarity - books
 - 'Does the book similarity look sensible, given my knowledge of the books?'.    

3) Topic extraction
 - 'Is this a good topic extraction from the given book?'

In this section, you will need to clearly define the metrics or calculations you will use to measure performance of a model or result in your project. These calculations and metrics should be justified based on the characteristics of the problem and problem domain. Questions to ask yourself when writing this section:
Are the metrics you’ve chosen to measure the performance of your models clearly discussed and defined?
Have you provided reasonable justification for the metrics chosen based on the problem and solution?

## II. Analysis

### Data Exploration

#### Data description
Data for the project is in the form of a mini-library of 41 books collected from the project Gutenberg.
Project Gutenberg is a collection of freely available ebooks on various formats, forming a huge historical library of literature. 

'Project Gutenberg offers over 57,000 free eBooks. Choose among free epub books, free kindle books, download them or read them online. You will find the world's great literature here, with focus on older works for which copyright has expired. Thousands of volunteers digitized and diligently proofread the eBooks, for enjoyment and education'. (From the website www.gutenberg.org)

As a balance between corpus size and variety - and volume manageable for computation, I decided to use 41 books for my mimi-library. The choice of books is based on :

- The books are all in english
- 6 books have Norwegian origin (Ibsens pays and 2 fairytale collections)
- Some of the items are plays (Ibsen, Shakespeare)
- I have myself read all books - primarily wholly, but for some only partially,  (Shakespeares complete works, the Iliad)
- Some authors are represented by several works (Verne, Ibsen, Dumas, Burroughs)
- Nearly half of the items can be described as 'journey tales'.
- Two of the authors (Shakespeare and Ibsen) are known for their very rich vocabulary, said to be 29.000 and 27.000 respectively.  

My list, as named in the file folder:

 'A_Dolls_house_Ibsen.rtf',
 'Alice_in_Wonderland.rtf',
 'All_around_the_moon_Verne.rtf',
 'An_archtartic_mystery_verne.rtf',
 'Anthem_Rand.rtf',
 'Around_the_world_in_80_days_Verne.rtf',
 'Don_Quixote.rtf',
 'Fairytales_H_C_Andersen.rtf',
 'Ghosts_Ibsen.rtf',
 'Great_Expectations_by_Charles_Dickens.rtf',
 'Gullivers_Travels_Defoe.rtf',
 'Hedda_Gabler_Ibsen .rtf',
 'Iliad_Homer.rtf',
 'Jungle_tales_of_Tarzan.rtf',
 'LIttle_Eyolf_Ibsen.rtf',
 'Martin_Eden_Jack_London.rtf',
 'Meditations_Aurelius.rtf',
 'Metamorphosis _ Kafka.rtf',
 'Norwegian_tales_Asbjornsen_Moe.rtf',
 'Peter_Pan.rtf',
 'Pride_and_Prejudices .rtf',
 'Robinson_Crusoe _ Defoe.rtf',
 'Shakespeare.rtf',
 'Siddharta _ Hesse.rtf',
 'Swanns_Way_Proust.rtf',
 'Tale_of_two_cities_dickens.rtf',
 'Tarzan_of_the_apes.rtf',
 'Tarzan_the_terrible.rtf',
 'The_3_musketeers_Dumas.rtf',
 'The_Iron_Heel_Jack_london.rtf',
 'The_beasts_of_Tarzan.rtf',
 'The_black_tulip_Dumas.rtf',
 'The_brothers_karamazov_Dostoyevsky.rtf',
 'The_importance_of_being_earnest_Wilde.rtf',
 'The_jungle_book_Kipling.rtf',
 'The_man_in_the_iron_mask_Dumas.rtf',
 'The_return_of_tarzan.rtf',
 'The_secret_of_the_island_verne.rtf',
 'The_trial_Kafka.rtf',
 'War_and_peace_tolstoy.rtf',
 'Wuthering_Heights_bronte.rtf'

Although the filenames indicate .rtf (Rich Text Format), they are in straightforward text format. I started with a few rft format files, bud decided to download the .txt format which i realized is available for all my books. The file naming was kept for the sake of convenience as the code reading the files was setup for the .rtf extension. All books are present in a folder in GitHub. 

#### Comments on the book data

- After a few assessments of the data on a word level I noticed some parts of the vocabulary that I did not expect from older works, such as 'email'. I had screened through the first part of most of the books to verify the content and format, which all seemed correct. At closer inspection I realized that all ebooks had a standard, quite long, section at the end, containing legal description of the ebook handling. In addition to the introduction of unwanted irrelevant vocabulary to the corpus this part is the same in all books and would therefore reduce the uniqueness of the items. The legal section was for this reason removed from each of the books.   

- The complete volume of words from the books, the 'corpus' was analyzed thouroughly. 
  There are many levels (and combination of levels) to look at the corpus;  

- Corpus level
- Book level
- Sentence level
- Vocabulary level (unique words)
- Word level (real words, or 'tokens')
- Level of 'words' before removing non-meaningful characters.  
- Level of 'reduced tokens'; ie words cleaned and/or transformed to a netted minimum.   

Below is an example of numbers for these levels

Corpus_raw: 30776216
Wordlist: 5572533
Sentences: 309218
Vocab: 96909

As a reflection 96909 comes across as a very rich vocabulary, even with Shakespeare and Ibsen onboard. One reason is probably that names are included in the corpus and are counted as words. Linguists state that if we include all varieties of known english words it will add up to ca 500.000. A normal active vocabulary consists of a modest 5-6000 words in most countries.    

One of the first transforms on the corpus was done using the gensim.utils.simple_preprocess, which performs this : 
'Convert a document into a list of lowercase tokens, ignoring tokens that are too short or too long'.
https://programtalk.com/python-examples/gensim.utils.simple_preprocess/. 

On inspection this returns a compact list of tokens, the number of words reduced significantly, ie for Alice in wonderland, from 53690 raw words to 9401 tokens. Netting down to unique tokens, in effect the 'token vocabulary' shows a modest token vocabulary of 1504.    

Although utilities with options such as the 'simple_preprocess' are useful, I wanted to test effects of the preprocessing in detail, and spent time trying out different functions on sentence and word levels, coding small functions with basic tokenizers and cleaning. 

#### Assessing different word_tokenizers

Tokenizers in common utility libraries offers a variety of processing option. I tested the different outcomes of two often used word tokenizers applied on sentences created from the punct_sentence_tokenizer; 

nltk.tokenize.word_tokenize(sentence) versus nltk.tokenize.wordpunct_tokenize(sentence)
For sentence no:99 in the Alice in Wonderland corpus we can see how the word tokenizers differ, respectively:

('she was up to her chin in salt-water.',
 ['she', 'was', 'up', 'to', 'her', 'chin', 'in', 'salt-water', '.'],
 ['she', 'was', 'up', 'to', 'her', 'chin', 'in', 'salt', '-', 'water', '.'])

The word_tokenizer keeps the hyphenated words as one token, the wordpunct_tokenize splits these words into 3 separate tokens.  When analyzing text in literary works, it makes sense to keep the original word as written by the author, this sentence is a good example as the words 'salt' and 'water' separately is quite different from 'salt-water'.  The first could be a part of a recipe for baking bread, the second indicates seawater - or tears, as is the case in this story. 

I choose to use the word_tokenizer for this book-based project, although another context could be a case for choosing the  wordpunct_tokenizer. 

#### Cleaning the sentences for stopwords

When preparing for topic analysis and other content assessments we want to look at the meaningful elements of the text, not the 'filler words'. There are many lists of 'filler words' or 'stop words' as they are called, for several lanuages. I use a list of english 'stop words' (from the nltk.corpus) and write two small code snippets to remove these words in a sentence and then loop for all sentences in a book. The function 'make-fin-sent' will 'make final sentences' for a given book, to be used for topic extraction: 

def remove_stopwords(words):
       return [word for word in words if word not in stop_words]

def make_fin_sent(sent_in):
    fin_sent=[]
    for sent in sent_in:
        fin_sent.append(remove_stopwords(sent))
    return fin_sent
    
A final sentence wil typically look like this (sentence no 80 in Swanns Way):     
    
['Come', 'stop', 'husband', 'drinking', 'brandy']


#### Looking at frequent words in a text

Zooming on on the corpus the need for data cleaning soon becomes obvious. First try of 'most common words' on a raw (uncleaned) corpus returns a rather uninteresting list over the 10 most frequent 'words', using FreqDist (from nltk.probability import FreqDist)

({',': 460770, 'the': 270580, '.': 258286, 'and': 160660, 'of': 143300, 'to': 142159, 'a': 95923, 'I': 91671, 'in': 78754, 'that': 71929, ...})

Removing punctiations and single characters with a small cleaning function improves a little

cleanlist=[]
def sentence_to_cleanlist(raw):
    cleanlist = re.sub("[^a-zA-Z]"," ", raw)
    return cleanlist

({'the': 270904, 'and': 161240, 'to': 143756, 'of': 143437, 'a': 96535, 'I': 92847, 'in': 79205, 'that': 72439, 'he': 64047, 'his': 58524, ...})

This exercize can be done for the individual books, but across english books the top 20 words pretty much remains the same. 

Next step is be to remove the stop words, as decribed above. 

If I use the result from removing stopwords, the most frequent words for the full corpus are:

('said', 19290),
 ('one', 17840),
 ('would', 17020),
 ('man', 11407),
 ('could', 11068),
 ('upon', 9815),
 ('well', 9252),
 ('time', 9081),
 ('know', 8822),
 ('thou', 8765)

Looking at this list, I can recognize the words of literature objects and dialogue-oriented text, with 'said' on top of the lists.  
The last word is a hint of the book's age and I guess this is a Shakespearian word. 

Below is a figure showing the count of the 30 most common words in the corpus, after prereprocessing as described. 

![image](https://user-images.githubusercontent.com/42574791/49156595-06981a00-f31e-11e8-8e8f-f7007265d8b9.png)


Applying the word frequency on only Alice in Wonderland, without stopwords, we get this list: 
('alice', 173),
 ('said', 144),
 ('little', 59),
 ('rabbit', 37),
 ('one', 35),
 ('like', 34),
 ('queen', 30),
 ('could', 28),
 ('mouse', 27),
 ('illustration', 26)

for 'The junglebook' , the 10  most frequent words when cleaned for filler words are (using FreqDist again):

'said': 430, 'little': 231, 'mowgli': 220, 'man': 177, 'one': 174, 'would': 162, 'jungle': 147, 'head': 137, 'bagheera': 129, 'could': 125, 

#### Recap - data cleaning in nlp

Cleaning the corpus can be done by several small cleaning code-snippets;

- sentence tokenization
- word tokenizations
- removing unwanted characters (often called normalization)
- converting all words to lowercase  
- removing stopwords
- lemmming and stemming
- pos-tagging 

The 5 first cleaning steps are described. The two latter were tested on the corpus, but not used for later transformation purposes. 
Code for these tests are kept in the Notebook.  






In this section, you will be expected to analyze the data you are using for the problem. This data can either be in the form of a dataset (or datasets), input data (or input files), or even an environment. The type of data should be thoroughly described and, if possible, have basic statistics and information presented (such as discussion of input features or defining characteristics about the input or environment). Any abnormalities or interesting qualities about the data that may need to be addressed have been identified (such as features that need to be transformed or the possibility of outliers). Questions to ask yourself when writing this section:
If a dataset is present for this problem, have you thoroughly discussed certain features about the dataset? Has a data sample been provided to the reader?
If a dataset is present for this problem, are statistics about the dataset calculated and reported? Have any relevant results from this calculation been discussed?
If a dataset is not present for this problem, has discussion been made about the input space or input data for your problem?
Are there any abnormalities or characteristics about the input space or dataset that need to be addressed? (categorical variables, missing values, outliers, etc.)

### Exploratory Visualization
In this section, you will need to provide some form of visualization that summarizes or extracts a relevant characteristic or feature about the data. The visualization should adequately support the data being used. Discuss why this visualization was chosen and how it is relevant. Questions to ask yourself when writing this section:
Have you visualized a relevant characteristic or feature about the dataset or input data?
Is the visualization thoroughly analyzed and discussed?
If a plot is provided, are the axes, title, and datum clearly defined?
Algorithms and Techniques
In this section, you will need to discuss the algorithms and techniques you intend to use for solving the problem. You should justify the use of each one based on the characteristics of the problem and the problem domain. Questions to ask yourself when writing this section:
Are the algorithms you will use, including any default variables/parameters in the project clearly defined?
Are the techniques to be used thoroughly discussed and justified?
Is it made clear how the input data or datasets will be handled by the algorithms and techniques chosen?

### Benchmark

A benchmark for performance for this project is somewhat challenging as the 'real-life' benchmark starts with time and cost for a person reading and understanding text. Such time-and cost values not often used as benchmark in these experiments. Nevertheless, these figures will be the basis for a business case if the solution is to be evaluated for a project, measured against the project costs.   
With an average adult reading speed of 250 words per minute a book with 90.000 words (estimated average lenght of a novel) will be read in 360 minutes; 6 hours. For my bank case, a relevant comparison would be a document of perhaps 5000 words; estimated reading time being 20 minutes. The act of analysis and documenting the results will represent additional time and effort. Obviously, a machine-based read, classification and topic extract will represent a major performance increase.    

If we rather see the benchmark as the 'first state of the solution' , to be compared to the final version the numbers will describe the model improvements. These improvments will be discussed.  

In this section, you will need to provide a clearly defined benchmark result or threshold for comparing across performances obtained by your solution. The reasoning behind the benchmark (in the case where it is not an established result) should be discussed. Questions to ask yourself when writing this section:
Has some result or value been provided that acts as a benchmark for measuring performance?
Is it clear how this result or value was obtained (whether by data or by hypothesis)?

## III. Methodology

### Data Preprocessing

NLP requires several steps of preprocessing just as we do for numerical values and categorical data. The methods are different, but there are several rich libraries of code available. In effect, the first challenge is to understand which packages to apply. Data cleaning for nlp differs from cleaning of datasets for numerical analysis - the choice of preparing must be made based on the purpose of the analysis, we can seldom do an all-purpose text cleaning. 

I realized that the three separate questions posed in the project would have to be answered with three separate paths with different data prepping and processing. 

1) Data analysis of all words in all books in the mini-library
- With a focus on the sum of all words in the book collection the preprocessing starts with a 'full' collection of words across the books. The text is divided into sentences and all words are identified separately. The books are structured into sentences, using a trained process. I also use a word-oriented vectorization and eventually look at the matrix of all words, identifying the most frequent words and how the words are related to each other in the matrix.

In the analysis process i keep separate versions of the resukt of different text preparations, to be in control of the results and which versions suitable for the next step of analysis. 

2) Data analysis for comparing books in the mini-library. ,   
When aiming to compare text objects I need to separate the objects(books) and prepare them individually, building structures for mathematical comparisons. For this purpose I use another version of the package, with a document(ie book) focus. Each book is vectorized and the vectors are compared.   

3) Data prepping and analysis for topic extraction 
Finally, the prepping for topic extraction will reuse some of the prep steps, but with a twist; the text is divided into books again, this time on a corpus level. All analysis is done on book level. In addition to constructing sentences and cleaning for unneccesary characters, the text is made into lower characters and all stopwords are removed explictly. Stopwords in the text makes it difficult to extract topics and forms unneccessary 'noise'. This was very clerly visible from the code rund I performed before and afre stopword removal.   

Algorithms
- NLTK
- Gensim
- LDA
- TfID
- SKLearn

In this section, all of your preprocessing steps will need to be clearly documented, if any were necessary. From the previous section, any of the abnormalities or characteristics that you identified about the dataset will be addressed and corrected here. Questions to ask yourself when writing this section:
If the algorithms chosen require preprocessing steps like feature selection or feature transformations, have they been properly documented?
Based on the Data Exploration section, if there were abnormalities or characteristics that needed to be addressed, have they been properly corrected?
If no preprocessing is needed, has it been made clear why?

### Implementation

Transformation of data after preprocessing

Stepping up the analysis after preprocessing is done by vectorizing the corpus.
There are numerous different ways to do this and quite tricky to navigate among the options.
I chose to use the Gensim functionality for vectorizing. Gensim offers vetorization option on document and word level, with word2Doc and Word2Vec. 

 1)Analysing the corpus on a word level

The whole corpus is vectorized with Word2Vec. To be able to visually inspect the results I use TSNE to reduce the dimensionality to 2 dimensions.  

The model (word_model) is built and used :

Setting parameters
Instanciating the model; 'word_model'
Building vocabulary
Train the model on corpus sentences
Creating and looking at the matrix from the model
INstanciating tSNE for dimension reduction of the matrix
Run tsneFitTransform on the matrix to obtain a 2dimensional matrix
Visualizing the 2dim matrix from a dataframe and plots
Using the model for finding similar instances (data points) 

2) Analysing the corpus on a book level - looking for similarities

For looking at corpus on a book level I used Gensims Doc2Vec 

model_doc
The model is trained on the book_corpus. 
Similarity on doc(book) level. 





### 1) Word analysis with Word2Vec 

Gensim has a reknown Vectorization option, the Word2Vec which, as the name implies, vectorizes the corpus on a word level. Each unique word is assigned a numerical value, giving av matrix where one dimension is the size of the vocabulary.  

I start by establishing a word-based Word2Vec model; named 'word_model'. 

The word_model is applied on the whole corpus tokenized on a sentence-level, using the version called 'sentences'. In 'sentences', the full corpus is split into sentences and each word is tokenized. 

Sentence no 5678 in the corpus is a good exapmle: 'when','we','get','to','the','moon','what','shall','we','do','there'

The word_model is set by : word_model = w2V.Word2Vec(parameters)

The parameters : 

word_model.build_vocab(sentences)

print("model vocabulary length:", len(word_model.wv.vocab))
model vocabulary length: 12445

COnfirming the model after building vocabulary:

Word2Vec(vocab=12445, size=200, alpha=0.025)

word_model.train(sentences, epochs=5,
total_examples=12445)

all_word_vectors_matrix = word_model.wv.vectors

the resulting matrix is huge (12445, 200). To be able to visualize and look into the matrix I use TSNE to reduce the dimensions to 2. 

tsne = sklearn.manifold.TSNE(n_components = 2, 
                             early_exaggeration = 6,
                             learning_rate = 500,
                             n_iter = 300,
                             random_state = 2)

 
all_word_vectors_matrix_2dim = tsne.fit_transform(all_word_vectors_matrix)

The matrix values are set into a dataframe for inspection and visuals. The shape is the vocabulary size (12445, 2).

On this dataframe I can 'look at' the word data in a frame with visuals. 

All the 12445 wiords are placed along the x- and y-axis, forming a circle-like cloud. 

I can 'zoom in' on elements in the frame, by navigating along the x- and y-axis.

When picking a word the coorinates are given , whowing where the word resides. If I zoom in on the area around the chosen word I can see which words are in the close vicinity vector-vise. 
A few examples:

![image](https://user-images.githubusercontent.com/42574791/49079447-de86b900-f240-11e8-929c-519de82fb7ac.png)

![image](https://user-images.githubusercontent.com/42574791/49079567-31607080-f241-11e8-86e4-cefdea35dffb.png)



Most similar words - in the vector dimension
Below are two examples, the first on the initial model version, the second on an enhanced model version . The results were not dramatically improved for this task after model tweaking.  

word_model.wv.most_similar('pride')

('envy', 0.7191991806030273),
 ('ambition', 0.7174361348152161),
 ('meekness', 0.7124776244163513),
 ('fortitude', 0.7029587030410767),
 ('chastisement', 0.7014052271842957),
 ('disdain', 0.7011118531227112),
 ('hypocrisy', 0.7005854845046997),
 ('penitence', 0.6966824531555176),
 ('cowardice', 0.696437656879425),
 ('deceitful', 0.6881036758422852)

word_model.wv.most_similar('hunger')

('thirst', 0.8457642197608948),
 ('pangs', 0.7334113121032715),
 ('fatigue', 0.7226671576499939),
 ('gnawing', 0.7220444679260254),
 ('starved', 0.6977795362472534),
 ('exhaustion', 0.6882658004760742),
 ('famine', 0.6871694922447205),
 ('hungry', 0.6860570907592773),
 ('sickness', 0.6858627796173096),
 ('craving', 0.6856997013092041)
 
#### Word pairing 
FInally, I want to try out the classic 'king-to-queen' vector comparison. 
 For this purpose I use the most_similar_cosmul functionality, based on cosine computation. 
 
The base for this is a ' x relates to y as a relates to...' logic, expecting to return the word that has a menaingful relation to a as y has to x. My attempts to do this were not in the first tryouts as clear as the reknown queen-king case. I thought the reason was a limited corpus, but changing the model parameters improved the result, as shown below. 

First model:
'ocean is related to submarine, as jungle is related to forest
swann is related to odette, as nora is related to helmer (OK!)
queen is related to king, as woman is related to girl'

After changing parameters the similarity tests are better; 

'man is related to woman, as queen is related to king
husband is related to wife, as man is related to woman
swann is related to odette, as artagnan is related to porthos
sea is related to boat, as city is related to palaces'


### 2) Book comparisons, using Doc2Vec

Dov2Vec is similar to Word2Vec, but applied on document (here book) level. 

A hypothesis was to use tfidf here, but reading lays out that LDA does not need TF-IDF and can be used with bag_of_words only. There are different opnions on whether TF-idf will improve the results, but without clear conclusions. This is also commented later in the report.

The model is defined and named #### model_doc. Data input is 'book_corpus', the book-level corpus established earlier.   
The model uses 3 parameters. All were varied during the project, the results seemed to stay at fairly the same level. 

model_doc = gensim.models.Doc2Vec(vector_size = 300, 
                              min_count = 3, 
                              epochs = 10)
                              
model_doc.build_vocab(book_corpus) 
model's vocabulary length: 31862

model_doc.train(book_corpus, total_examples=31862, epochs=10)

Finding the most similar book pairs, sorted by hitrate(similarity)
for book in book_filenames:
    most_similar = model_doc.docvecs.most_similar(book)[0][0]
    print("{} - {}".format(book, most_similar))
    
Result list below shows that similarity is found between books from same author, but also of same genre. 
Folklore tales are paired, so are Gullivers travels and Robinson Crusoe.
 
Books\A_Dolls_house_Ibsen.rtf - Books\Hedda_Gabler_Ibsen .rtf
Books\Alice_in_Wonderland.rtf - Books\Norwegian_tales_Asbjornsen_Moe.rtf
Books\All_around_the_moon_Verne.rtf - Books\The_secret_of_the_island_verne.rtf
Books\An_archtartic_mystery_verne.rtf - Books\The_secret_of_the_island_verne.rtf
Books\Anthem_Rand.rtf - Books\The_jungle_book_Kipling.rtf
Books\Around_the_world_in_80_days_Verne.rtf - Books\The_secret_of_the_island_verne.rtf
Books\Don_Quixote.rtf - Books\Iliad_Homer.rtf
Books\Fairytales_H_C_Andersen.rtf - Books\Norwegian_tales_Asbjornsen_Moe.rtf
Books\Ghosts_Ibsen.rtf - Books\LIttle_Eyolf_Ibsen.rtf
Books\Great_Expectations_by_Charles_Dickens.rtf - Books\Tale_of_two_cities_dickens.rtf
Books\Gullivers_Travels_Swift.rtf - Books\Robinson_Crusoe _ Defoe.rtf
    
WHen testing on specific books, I use the 'most_similar:

model_doc.docvecs.most_similar('Books\Tarzan_of_the_apes.rtf')

This returns the other Tarzan books, folowed by adventure books for adolescents; quite a good result.  

('Books\\The_beasts_of_Tarzan.rtf', 0.7585173845291138),
 ('Books\\Jungle_tales_of_Tarzan.rtf', 0.7063543796539307),
 ('Books\\The_return_of_tarzan.rtf', 0.4879530370235443),
 ('Books\\Tarzan_the_terrible.rtf', 0.4041978716850281),
 ('Books\\An_archtartic_mystery_verne.rtf', 0.319263219833374),
 ('Books\\The_secret_of_the_island_verne.rtf', 0.3165384531021118),
 ('Books\\The_jungle_book_Kipling.rtf', 0.31132781505584717),
 ('Books\\Peter_Pan.rtf', 0.30049124360084534),
 ('Books\\Alice_in_Wonderland.rtf', 0.28424006700515747),
 ('Books\\Robinson_Crusoe _ Defoe.rtf', 0.2718788981437683)

 model_doc.docvecs.most_similar(10) (Book 10 = Gullivers Travels by Defoe)
 
 ('Books\\Robinson_Crusoe _ Defoe.rtf', 0.47113150358200073),
 ('Books\\Pride_and_Prejudices .rtf', 0.3537435233592987),
 ('Books\\Meditations_Aurelius.rtf', 0.3536701798439026),
 ('Books\\Fairytales_H_C_Andersen.rtf', 0.25862976908683777),
 ('Books\\The_secret_of_the_island_verne.rtf', 0.25140485167503357)
 
Tsne dimension reduction was done for this vectors as well

all_doc_vectors_matrix = model_doc.wv.vectors
all_doc_vectors_matrix_2 = tsne.fit_transform(all_doc_vectors_matrix)


### 3) Topic extraction with doc2bow  and LDA

For topic extraction I chose to use a third part of Gensim; doc2bow - using a bag-of-word method (bow: bag-of-words)
I also used another algorithm; LDA; often used for nlp. LDA: Latent Dirichlet Allocation. LDA is in the Gensim package, too.  
Create a Dictionary from the data, then a bag of words.  

Creating a corpora Dictionary based on a preprocessed book corpus on sentence level
Creating a matrix with the Doc2Bow
Instanciating lda
Performing lda on the corpus, using the dictionary
Printing the calculated topics from the ldamodel

Data input for Topic Extraction will be, on a book level, the set of preprocessed sentences from the 'make-fin-sent' function. For finding topics, we do not want filler words as they will form a data noise here. ''

Using Alice as my example again I start with creating a corpora Dictionary and use 'alice_fin_sent'.

dictionary_alice = corpora.Dictionary(alice_fin_sent)

doc_term_matrix_alice = [dictionary_alice.doc2bow(doc) for doc in alice_fin_sent]

Lda = gensim.models.ldamodel.LdaModel

ldamodel_alice = Lda(doc_term_matrix_alice, num_topics=3, id2word = dictionary_alice, passes=50)

alice_topics = ldamodel_alice.print_topics(num_words=5)
for topic in alice_topics:
    print(topic)
    
Presenting the proposed 3 topics using 50 passes: 

(0, '0.014*"one" + 0.012*"duchess" + 0.009*"alice" + 0.008*"first" + 0.008*"little"')
(1, '0.036*"alice" + 0.018*"little" + 0.016*"said" + 0.012*"rabbit" + 0.009*"could"')
(2, '0.056*"said" + 0.037*"alice" + 0.012*"king" + 0.012*"know" + 0.009*"like"')

Setting to 250 passes: 

(0, '0.031*"alice" + 0.021*"said" + 0.010*"mouse" + 0.009*"thought" + 0.009*"go"')
(1, '0.041*"alice" + 0.037*"said" + 0.010*"rabbit" + 0.007*"came" + 0.007*"moment"')
(2, '0.017*"one" + 0.017*"little" + 0.013*"said" + 0.013*"alice" + 0.012*"way"')

Setting to 750 passes:

0, '0.066*"said" + 0.042*"alice" + 0.010*"duchess" + 0.009*"oh" + 0.009*"think"')
(1, '0.014*"little" + 0.013*"know" + 0.010*"alice" + 0.009*"said" + 0.009*"queen"')
(2, '0.031*"alice" + 0.015*"rabbit" + 0.011*"little" + 0.009*"white" + 0.008*"one"')



The topic extraction must be performed for each book, and on a sentence level.
As shown, I varied the no: of passes, and also the no of topics and num_words. It is not obvious that the results increase with more passes; is "said+alice+duchess+oh+think' better than 'one+duchess+alice+first+little' ?

I can see that Proposed Topics are similar to the book's frequent words, but not 100% identical.

The same topic extraction steps were done for 5 other books, with similar results. 
As  a pattern, topics seem to include the books protagonist, the second most important character (if any) and some typical words





In this section, the process for which metrics, algorithms, and techniques that you implemented for the given data will need to be clearly documented. It should be abundantly clear how the implementation was carried out, and discussion should be made regarding any complications that occurred during this process. Questions to ask yourself when writing this section:
Is it made clear how the algorithms and techniques were implemented with the given datasets or input data?
Were there any complications with the original metrics or techniques that required changing prior to acquiring a solution?
Was there any part of the coding process (e.g., writing complicated functions) that should be documented?

### Refinement

NLP enjoys a wide variety of resources for prepping and transforming data, Nevertheless, there is room for coding and adjusting before, during and after package usage. 

Several of the improvement measures taken are mentioned above, a recap is as following;

Preprocess : Data gathering and cleaning

 - Legal clausuls at the end of the books were removed comppletely as i saw they introduced 'noise' in the text
 - Some sentences in the book's beginnings were removed for same reasons; Internet Urls, email addresses etc. 
 - Tokenizers on sentence and word level were tested and compared; I chose the one that kept full original words
 
1) Word analysis 
 - Fdist results from raw_corpus were poor, I changed the input to sentences with both cleaning of characters and stopwords removed
 - Word2Vec model parameters:
       num_features = 200 (kept unchanged)
       min_word_count = 20 (inreased from 10 to 20)
       num_workers = multiprocessing.cpu_count() (unchanged)
       context_size = 10 (increased from 5 to 10)
       downsampling = 1e-4 (unchanged)
       seed = 9 (unchanged)
       
 - TSNE parameters were tweaked; aiming a balance between results and runtime
 
 
 2) Book analysis
 
 The book analysis is similar to the word analysis method-wise. I did tryouts with parameters for the model and the tsne dimension reduction. On book-level, the model seems less sensitive for parameter tuning, probably because we operate on a book-level where similarity will be more obvious, the most typical result being sililarity by author, followed by genres, picking out adventure stories for adolescents.  
 
 3) Topic extraction
 
 The final task turned out to be quite straightforward when I had found the recommended method from books and articles.
 
 The method is compact when all data is preprocessed and the steps prepared code-wise. I chose to write small code packages for a 'pipeline' and tested this on several books. The results are of varying quality, but confirm the idea of being able to extract topics automatically from larger texts, such as my books.   
 
 One of the articles about the topic is the following, by Susan Li : 
 https://towardsdatascience.com/topic-modelling-in-python-with-nltk-and-gensim-4ef03213cd21
 
 LDA parameters


LDA parameters 
Vector parameters
Number of epochs

The models were kept, the parameters tweaked

Choices of levels where to operate 



In this section, you will need to discuss the process of improvement you made upon the algorithms and techniques you used in your implementation. For example, adjusting parameters for certain models to acquire improved solutions would fall under the refinement category. Your initial and final solutions should be reported, as well as any significant intermediate results as necessary. Questions to ask yourself when writing this section:
Has an initial solution been found and clearly reported?
Is the process of improvement clearly documented, such as what techniques were used?
Are intermediate and final solutions clearly reported as the process is improved?

## IV. Results

1) Word analysis
- Confirming vocabulary, clear visual of vocab in a plotted wordcloud
- Zooming in on chosen words, see the vicinity words
- identifying most frequent words

2) Book similarity
- Good results after parameter tweaks. Obvious similarity between books by the same author, also similarity caused by plot and genre.
- Visuals in form of a wordcloud as for the word analysis 

3) Topic extraction
- Quite good results. 
- Verification of result, in addition to scoring figures done by book knowledge and a generous interpretation of the topic words.   

Test of unknown book 
Finally I found a completely unknown book and performed Topic extraction on the text

The results from this old book about food and chemistry strikes me by being suprisingly relevant for today's view on food and health. 

 '0.017*"food" + 0.016*"protein" + 0.012*"quantity" + 0.011*"matter"'
 '0.007*"salt" + 0.007*"one" + 0.006*"food" + 0.005*"meat"'


### Extra : Test of a contemporary document

For fun and curiosity, I tested the topic extraction on a random modern document; "Google Chrome Terms of Service"

Spending 5 minutes downloading the file, running it through the code, gave this topic extraction:

(0, '0.031*"terms" + 0.030*"adobe" + 0.020*"software" + 0.020*"use" + 0.020*"sublicensee" + 0.019*"shall" + 0.017*"may"')
(1, '0.030*"sublicensee" + 0.028*"terms" + 0.021*"google" + 0.020*"adobe" + 0.018*"rights" + 0.013*"additional" + 0.012*"may"')
(2, '0.035*"adobe" + 0.028*"software" + 0.018*"video" + 0.018*"code" + 0.017*"content" + 0.014*"may" + 0.013*"avc"')
(3, '0.071*"google" + 0.032*"terms" + 0.027*"services" + 0.020*"agreement" + 0.018*"adobe" + 0.018*"chrome" + 0.014*"content"')
(4, '0.046*"google" + 0.038*"services" + 0.023*"may" + 0.021*"use" + 0.019*"software" + 0.017*"adobe" + 0.017*"sublicensee"')

It is not a trimmed executive summary, but it indicates the content, ie by the last topic suggestion

-"google-services-may-use-software-adobe-sublicensee'

### Model Evaluation and Validation

1) Word analysis

in this Kaggle post below (chapter 2 and 3) -  and in other similar articles, different options for further work on the word vecors, beyond the bag-of_words (bow), are discussed. Is seems to be a common consensus that the 'bow' method is robust and gives consistently good results compared to other methods.  

https://www.kaggle.com/c/word2vec-nlp-tutorial#part-1-for-beginners-bag-of-words


2) Document comparisons

3) Topic extraction

Testing on a new and unknown text confirmed the results for the other 41 books.


In this section, the final model and any supporting qualities should be evaluated in detail. It should be clear how the final model was derived and why this model was chosen. In addition, some type of analysis should be used to validate the robustness of this model and its solution, such as manipulating the input data or environment to see how the model’s solution is affected (this is called sensitivity analysis). Questions to ask yourself when writing this section:
Is the final model reasonable and aligning with solution expectations? Are the final parameters of the model appropriate?
Has the final model been tested with various inputs to evaluate whether the model generalizes well to unseen data?
Is the model robust enough for the problem? Do small perturbations (changes) in training data or the input space greatly affect the results?
Can results found from the model be trusted?

### Justification

Benchmarking these results with manual handling of text is not easily mesured, but I'll provide my reflections.

The first issue can be seen as a prepping for the important questions; can we compare documents and extract essence? Analyzing the document based on the full corpus  and word-focused will give us the big pictore of the text 'mass'. 40 books in the mini-library may sound like a lot of words, but in this dicipline it forms a meagre corpus, yet enough to do the project. 

Comparing documents applied to literan peices of text as the library represents is interesting, but only partly representative for my real-life challenge. The documents in scope for implementation in DNB will be more specific and probably more distingusashable by context vocabulary, aligning Trade Finance documents and separating those from ie Loan Agreements.  

Topic extraction is the problem I am most eager to try out internally. 



In this section, your model’s final solution and its results should be compared to the benchmark you established earlier in the project using some type of statistical analysis. You should also justify whether these results and the solution are significant enough to have solved the problem posed in the project. Questions to ask yourself when writing this section:
Are the final results found stronger than the benchmark result reported earlier?
Have you thoroughly analyzed and discussed the final solution?
Is the final solution significant enough to have solved the problem?

## V. Conclusion



Free-Form Visualization
In this section, you will need to provide some form of visualization that emphasizes an important quality about the project. It is much more free-form, but should reasonably support a significant result or characteristic about the problem that you want to discuss. Questions to ask yourself when writing this section:
Have you visualized a relevant or important quality about the problem, dataset, input data, or results?
Is the visualization thoroughly analyzed and discussed?
If a plot is provided, are the axes, title, and datum clearly defined?


## Reflection

Diving into NLP has been challenging yet satisfying as this is not a redo of projects but new skills and learnings. 

THis is not a straight forward-technique but an area where any project will be a series of crossroads where choices must be made of methods, algorithms and so on. Iterations is neccessary, and as metioned earlier there will be separate lanes for the individual questions. 

As always, there are struggles with formats and technicalities. One of my first learnings was that some opartions must be made on sentence level, other on word/token level. It took a while to grasp that topic extraction had to be done book by book, not on the complete corpus, even if this was document-tagged. 

Topic extraction is sensitive to input and the results before stopword removal were quite useless. I testet several parameters and was suprised by the vareid returns. Book knowledge was important for result assessment. 


In this section, you will summarize the entire end-to-end problem solution and discuss one or two particular aspects of the project you found interesting or difficult. You are expected to reflect on the project as a whole to show that you have a firm understanding of the entire process employed in your work. Questions to ask yourself when writing this section:
Have you thoroughly summarized the entire process you used for this project?
Were there any interesting aspects of the project?
Were there any difficult aspects of the project?
Does the final model and solution fit your expectations for the problem, and should it be used in a general setting to solve these types of problems?


## Improvement

My project did not include establishing a user interface for input of random documents, nor a prepped output in the form of screenbased result presnetation or printable report. For my real-life problem this will have to be made, but not neccesarily a glossy user interfase, as this is for internal use in back office environments, nor for customer usage externally. 

Topic extraction need to be tuned more, to find the optimal parameters for a given corpus.


In some initial assessments of the text I tried out two different tokenizers and was suprised to see the diference in vocabulary size it produced. At furter investigation (I made the diff set by subtracting the smallest vocab from the larger) i could see where the tokenizers diverge. The punkt_tokenizer splits hyphenated words, the standard nltk tokenizer does not. 
A discussion on tokenizers is presented here: 
https://www.researchgate.net/publication/264157595_A_Comparison_of_13_Tokenizers_on_MEDLINE


In this section, you will need to provide discussion as to how one aspect of the implementation you designed could be improved. As an example, consider ways your implementation can be made more general, and what would need to be modified. You do not need to make this improvement, but the potential solutions resulting from these changes are considered and compared/contrasted to your current solution. Questions to ask yourself when writing this section:
Are there further improvements that could be made on the algorithms or techniques you used in this project?
Were there algorithms or techniques you researched that you did not know how to implement, but would consider using if you knew how?
If you used your final solution as the new benchmark, do you think an even better solution exists?

Before submitting, ask yourself. . .
Does the project report you’ve written follow a well-organized structure similar to that of the project template?
Is each section (particularly Analysis and Methodology) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
Would the intended audience of your project be able to understand your analysis, methods, and results?
Have you properly proof-read your project report to assure there are minimal grammatical and spelling mistakes?
Are all the resources used for this project correctly cited and referenced?
Is the code that implements your solution easily readable and properly commented?
Does the code execute without error and produce results similar to those reported?
