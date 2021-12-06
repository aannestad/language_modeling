# %% [markdown]
# # INF368A - Exercise 6 - Vector Embeddings
# ### By Hans Martin Aannestad
# 
# ## Exercise 1
# 
# Draw a spatial visualization of word vectors for cherry and
# strawberry, showing just two of the dimensions, corresponding to
# the words pie and sugar.

# %%
import matplotlib.pyplot as plt

# Read off table

cherry = (442, 25)
strawberry = (60, 19)

plt.figure(1)
ax = plt.axes()
ax.axis(xmin=0, xmax=450, ymin=0, ymax=30)
plt.annotate("", xy=(442, 25), xytext=(0, 0), arrowprops=dict(color="green",headwidth=10, headlength=10, width=0.1))
plt.annotate("", xy=(60, 19), xytext=(0, 0), arrowprops=dict(color="blue",headwidth=10, headlength=10, width=0.1))

ax.annotate("Cherry [442,25]", [442, 25],color="green")
ax.annotate("Strawberry [60,19]", [60, 19], color="blue")

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.xlabel("Pie")
plt.ylabel("Sugar")

plt.show()
plt.savefig('figure1.png')

# %% [markdown]
# ## Exercise 3:
# 
# Consider each paragraph of the next slide (adapted from Wikipedia)
# as a document and calculate TF-IDF of the words
# Shakespeare, poet, English

# %%
# Preprocess data

import nltk
from nltk import word_tokenize

with open("data_e6.txt") as input:
    docs = input.read().split("\n\n") # 1 doc pr. paragraph

#docs = [k.split(' ') for k in docs]
docs = [word_tokenize(s.lower()) for s in docs] # Tokenize
docs = [[word for word in doc if word.isalpha()] for doc in docs]
terms = set([w for doc in docs for w in doc])


# %% [markdown]
# ## Term frequency: $ tf_{t,d} = log_{10}(\text{count}(t,d)+1) $

# %%
import math
from collections import defaultdict

counts = [Counter(doc) for doc in docs]  # get word frequencies pr paragraph (document)

tfs = {} # replace with dict tf_{t,d}

doc_num = 1

for doc in counts:
    tf = {}
    for k in doc:
        tf[k] = math.log10(int(doc[k])+1)
    tfs['doc_' + str(doc_num)] = tf
    doc_num += 1

# %% [markdown]
# ## Inverse document frequency: $\text{idf}_{t} = log_{10}(\frac{N}{df_{t}}) $
# 
# For $N$ documents, the <strong>document frequency</strong> $dt_{t}$, for term $t$, is the number of documents the term occurs in

# %%
N = len(docs)
idf = {}

for t in terms:
    df = 0
    for d in docs:
        if t in d:
            df += 1
    idf[t] = math.log10(N/df) if df else math.log10(N) # safediv

# %% [markdown]
# ## tf-idf: $ w_{t,d} = \text{tf}_{t,d} \times \text{idf}_{t} $
# 
# 

# %%
from collections import defaultdict

tf_idf = {}

for t in terms:
    tf_idf[t] = {}
    for d,tf in tfs.items():
        tf_idf[t][d] = tf[t] * idf[t] if t in tf else 0

# %%
print(tf_idf['shakespeare'])
print(tf_idf['poet'])
print(tf_idf['english'])


# %%
## Exercise 4

'''Consider the paragraphs in slide 65.
Calculate the PPMI of the word "poet" in the context
"Shakespeare" and of the word "works" in the context of
"career". Assume that a word w co-occurs in the context c, with
c also being a word, if w and c occur in the same paragraph.
For building the term-context matrix consider
W={poet, works, English}
C={Shakespeare, career, language}
'''


