#!/usr/bin/env python
# coding: utf-8

# <i>Copyright (c) Microsoft Corporation. All rights reserved.</i>
# 
# <i>Licensed under the MIT License.</i>

# # SAR Single Node on MovieLens (Python, CPU)
# 
# Simple Algorithm for Recommendation (SAR) is a fast and scalable algorithm for personalized recommendations based on user transaction history. It produces easily explainable and interpretable recommendations and handles "cold item" and "semi-cold user" scenarios. SAR is a kind of neighborhood based algorithm (as discussed in [Recommender Systems by Aggarwal](https://dl.acm.org/citation.cfm?id=2931100)) which is intended for ranking top items for each user. More details about SAR can be found in the [deep dive notebook](../02_model_collaborative_filtering/sar_deep_dive.ipynb). 
# 
# SAR recommends items that are most ***similar*** to the ones that the user already has an existing ***affinity*** for. Two items are ***similar*** if the users that interacted with one item are also likely to have interacted with the other. A user has an ***affinity*** to an item if they have interacted with it in the past.
# 
# ### Advantages of SAR:
# - High accuracy for an easy to train and deploy algorithm
# - Fast training, only requiring simple counting to construct matrices used at prediction time. 
# - Fast scoring, only involving multiplication of the similarity matrix with an affinity vector
# 
# ### Notes to use SAR properly:
# - Since it does not use item or user features, it can be at a disadvantage against algorithms that do.
# - It's memory-hungry, requiring the creation of an $mxm$ sparse square matrix (where $m$ is the number of items). This can also be a problem for many matrix factorization algorithms.
# - SAR favors an implicit rating scenario and it does not predict ratings.
# 
# This notebook provides an example of how to utilize and evaluate SAR in Python on a CPU.

# # 0 Global Settings and Imports

# In[1]:


get_ipython().run_cell_magic('bash', '', 'ls -al ../../\nls -al ../../')


# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import sys

sys.path.append( "../../" )

import logging
import numpy as np
import pandas as pd
import scrapbook as sb
from sklearn.preprocessing import minmax_scale

from recommenders.utils.python_utils import binarize
from recommenders.utils.timer import Timer
from recommenders.datasets import movielens
from recommenders.datasets.python_splitters import python_stratified_split
from recommenders.evaluation.python_evaluation import (
    map_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    rmse,
    mae,
    logloss,
    rsquared,
    exp_var
)
from recommenders.models.sar import SAR
import sys

print("System version: {}".format(sys.version))
print("Pandas version: {}".format(pd.__version__))


# In[2]:


get_ipython().run_cell_magic('bash', '', 'which python')


# In[3]:


get_ipython().system('which python')


# # 1 Load Data
# 
# SAR is intended to be used on interactions with the following schema:
# `<User ID>, <Item ID>,<Time>,[<Event Type>], [<Event Weight>]`. 
# 
# Each row represents a single interaction between a user and an item. These interactions might be different types of events on an e-commerce website, such as a user clicking to view an item, adding it to a shopping basket, following a recommendation link, and so on. Each event type can be assigned a different weight, for example, we might assign a “buy” event a weight of 10, while a “view” event might only have a weight of 1.
# 
# The MovieLens dataset is well formatted interactions of Users providing Ratings to Movies (movie ratings are used as the event weight) - we will use it for the rest of the example.

# In[5]:


# top k items to recommend
TOP_K = 10

# Select MovieLens data size: 100k, 1m, 10m, or 20m
MOVIELENS_DATA_SIZE = '100k'


# ### 1.1 Download and use the MovieLens Dataset

# In[6]:


data = movielens.load_pandas_df(
    size=MOVIELENS_DATA_SIZE
)

# Convert the float precision to 32-bit in order to reduce memory consumption 
data['rating'] = data['rating'].astype(np.float32)

data.head()


# In[7]:


data.tail()


# ### 1.2 Split the data using the python random splitter provided in utilities:
# 
# We split the full dataset into a `train` and `test` dataset to evaluate performance of the algorithm against a held-out set not seen during training. Because SAR generates recommendations based on user preferences, all users that are in the test set must also exist in the training set. For this case, we can use the provided `python_stratified_split` function which holds out a percentage (in this case 25%) of items from each user, but ensures all users are in both `train` and `test` datasets. Other options are available in the `dataset.python_splitters` module which provide more control over how the split occurs.

# In[7]:


train, test = python_stratified_split(data, ratio=0.75, col_user='userID', col_item='itemID', seed=42)


# In[8]:


print("""
Train:
Total Ratings: {train_total}
Unique Users: {train_users}
Unique Items: {train_items}

Test:
Total Ratings: {test_total}
Unique Users: {test_users}
Unique Items: {test_items}
""".format(
    train_total=len(train),
    train_users=len(train['userID'].unique()),
    train_items=len(train['itemID'].unique()),
    test_total=len(test),
    test_users=len(test['userID'].unique()),
    test_items=len(test['itemID'].unique()),
))


# # 2 Train the SAR Model

# ### 2.1 Instantiate the SAR algorithm and set the index
# 
# We will use the single node implementation of SAR and specify the column names to match our dataset (timestamp is an optional column that is used and can be removed if your dataset does not contain it).
# 
# Other options are specified to control the behavior of the algorithm as described in the [deep dive notebook](../02_model_collaborative_filtering/sar_deep_dive.ipynb).

# In[9]:


logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s %(levelname)-8s %(message)s')

model = SAR(
    col_user="userID",
    col_item="itemID",
    col_rating="rating",
    col_timestamp="timestamp",
    similarity_type="jaccard", 
    time_decay_coefficient=30, 
    timedecay_formula=True,
    normalize=True
)


# ### 2.2 Train the SAR model on our training data, and get the top-k recommendations for our testing data
# 
# SAR first computes an item-to-item ***co-occurence matrix***. Co-occurence represents the number of times two items appear together for any given user. Once we have the co-occurence matrix, we compute an ***item similarity matrix*** by rescaling the cooccurences by a given metric (Jaccard similarity in this example). 
# 
# We also compute an ***affinity matrix*** to capture the strength of the relationship between each user and each item. Affinity is driven by different types (like *rating* or *viewing* a movie), and by the time of the event. 
# 
# Recommendations are achieved by multiplying the affinity matrix $A$ and the similarity matrix $S$. The result is a ***recommendation score matrix*** $R$. We compute the ***top-k*** results for each user in the `recommend_k_items` function seen below.
# 
# A full walkthrough of the SAR algorithm can be found [here](../02_model_collaborative_filtering/sar_deep_dive.ipynb).

# In[10]:


with Timer() as train_time:
    model.fit(train)

print("Took {} seconds for training.".format(train_time.interval))


# In[11]:


with Timer() as test_time:
    top_k = model.recommend_k_items(test, remove_seen=True)

print("Took {} seconds for prediction.".format(test_time.interval))


# In[12]:


top_k.head()


# ### 2.3. Evaluate how well SAR performs
# 
# We evaluate how well SAR performs for a few common ranking metrics provided in the `python_evaluation` module. We will consider the Mean Average Precision (MAP), Normalized Discounted Cumalative Gain (NDCG), Precision, and Recall for the top-k items per user we computed with SAR. User, item and rating column names are specified in each evaluation method.

# In[13]:


eval_map = map_at_k(test, top_k, col_user='userID', col_item='itemID', col_rating='rating', k=TOP_K)


# In[14]:


eval_ndcg = ndcg_at_k(test, top_k, col_user='userID', col_item='itemID', col_rating='rating', k=TOP_K)


# In[15]:


eval_precision = precision_at_k(test, top_k, col_user='userID', col_item='itemID', col_rating='rating', k=TOP_K)


# In[16]:


eval_recall = recall_at_k(test, top_k, col_user='userID', col_item='itemID', col_rating='rating', k=TOP_K)


# In[17]:


eval_rmse = rmse(test, top_k, col_user='userID', col_item='itemID', col_rating='rating')


# In[18]:


eval_mae = mae(test, top_k, col_user='userID', col_item='itemID', col_rating='rating')


# In[19]:


eval_rsquared = rsquared(test, top_k, col_user='userID', col_item='itemID', col_rating='rating')


# In[20]:


eval_exp_var = exp_var(test, top_k, col_user='userID', col_item='itemID', col_rating='rating')


# In[21]:


positivity_threshold = 2
test_bin = test.copy()
test_bin['rating'] = binarize(test_bin['rating'], positivity_threshold)

top_k_prob = top_k.copy()
top_k_prob['prediction'] = minmax_scale(
    top_k_prob['prediction'].astype(float)
)

eval_logloss = logloss(test_bin, top_k_prob, col_user='userID', col_item='itemID', col_rating='rating')


# In[22]:


print("Model:\t",
      "Top K:\t%d" % TOP_K,
      "MAP:\t%f" % eval_map,
      "NDCG:\t%f" % eval_ndcg,
      "Precision@K:\t%f" % eval_precision,
      "Recall@K:\t%f" % eval_recall,
      "RMSE:\t%f" % eval_rmse,
      "MAE:\t%f" % eval_mae,
      "R2:\t%f" % eval_rsquared,
      "Exp var:\t%f" % eval_exp_var,
      "Logloss:\t%f" % eval_logloss,
      sep='\n')


# In[23]:


# Now let's look at the results for a specific user
user_id = 876

ground_truth = test[test['userID']==user_id].sort_values(by='rating', ascending=False)[:TOP_K]
prediction = model.recommend_k_items(pd.DataFrame(dict(userID=[user_id])), remove_seen=True) 
pd.merge(ground_truth, prediction, on=['userID', 'itemID'], how='left')


# Above, we see that one of the highest rated items from the test set was recovered by the model's top-k recommendations, however the others were not. Offline evaluations are difficult as they can only use what was seen previously in the test set and may not represent the user's actual preferences across the entire set of items. Adjustments to how the data is split, algorithm is used and hyper-parameters can improve the results here. 

# In[24]:


# Record results with papermill for tests - ignore this cell
sb.glue("map", eval_map)
sb.glue("ndcg", eval_ndcg)
sb.glue("precision", eval_precision)
sb.glue("recall", eval_recall)
sb.glue("train_time", train_time.interval)
sb.glue("test_time", test_time.interval)


# In[ ]:





# In[ ]:





# In[ ]:





# $$
# \begin{array}{|l|l|l|}
# \hline 1 & \mathrm{a} & \mathrm{aa} \\
# \hline 2 & \mathrm{~b} & \mathrm{bb} \\
# \hline 3 & \mathrm{c} & \mathrm{cc} \\
# \hline 4 & \mathrm{~d} & \mathrm{dd} \\
# \hline
# \end{array}
# $$

# $$
# \begin{array}{|c|c|c|}
# \hline 1 & \mathrm{a} & \mathrm{aa} \\
# \hline 2 & \mathrm{~b} & \mathrm{bb} \\
# \hline 3 & \mathrm{c} & \mathrm{cc} \\
# \hline 3 & \mathrm{c} & \mathrm{cc} \\
# \hline 4 & \mathrm{~d} & \mathrm{dd} \\
# \hline 3 & \mathrm{c} & \mathrm{cc} \\
# \hline 3 & \mathrm{c} & \mathrm{cc} \\
# \hline 3 & \mathrm{c} & \mathrm{cc} \\
# \hline 3 & \mathrm{c} & \mathrm{cc} \\
# \hline 3 & \mathrm{c} & \mathrm{cc} \\
# \hline 3 & \mathrm{c} & \mathrm{cc} \\
# \hline 3 & \mathrm{c} & \mathrm{cc} \\
# \hline 3 & \mathrm{c} & \mathrm{cc} \\
# \hline 3 & \mathrm{c} & \mathrm{cc} \\
# \hline
# \end{array}
# $$

# In[7]:


get_ipython().run_cell_magic('time', '', '1 + 1')


# In[ ]:




