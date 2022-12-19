<i>Copyright (c) Microsoft Corporation. All rights reserved.</i>

<i>Licensed under the MIT License.</i>

# SAR Single Node on MovieLens (Python, CPU)

Simple Algorithm for Recommendation (SAR) is a fast and scalable algorithm for personalized recommendations based on user transaction history. It produces easily explainable and interpretable recommendations and handles "cold item" and "semi-cold user" scenarios. SAR is a kind of neighborhood based algorithm (as discussed in [Recommender Systems by Aggarwal](https://dl.acm.org/citation.cfm?id=2931100)) which is intended for ranking top items for each user. More details about SAR can be found in the [deep dive notebook](../02_model_collaborative_filtering/sar_deep_dive.ipynb). 

SAR recommends items that are most ***similar*** to the ones that the user already has an existing ***affinity*** for. Two items are ***similar*** if the users that interacted with one item are also likely to have interacted with the other. A user has an ***affinity*** to an item if they have interacted with it in the past.

### Advantages of SAR:
- High accuracy for an easy to train and deploy algorithm
- Fast training, only requiring simple counting to construct matrices used at prediction time. 
- Fast scoring, only involving multiplication of the similarity matrix with an affinity vector

### Notes to use SAR properly:
- Since it does not use item or user features, it can be at a disadvantage against algorithms that do.
- It's memory-hungry, requiring the creation of an $mxm$ sparse square matrix (where $m$ is the number of items). This can also be a problem for many matrix factorization algorithms.
- SAR favors an implicit rating scenario and it does not predict ratings.

This notebook provides an example of how to utilize and evaluate SAR in Python on a CPU.

# 0 Global Settings and Imports


```bash
%%bash
ls -al ../../
ls -al ../../
```

    total 232
    drwxr-xr-x   28 hiroshi.wayama  staff    896  8 24 00:34 .
    drwxr-xr-x+ 102 hiroshi.wayama  staff   3264  8 24 00:26 ..
    drwxr-xr-x   14 hiroshi.wayama  staff    448  8 24 00:02 .git
    drwxr-xr-x    9 hiroshi.wayama  staff    288  8 24 00:00 .github
    -rw-r--r--    1 hiroshi.wayama  staff   1887  8 24 00:00 .gitignore
    drwxr-xr-x   12 hiroshi.wayama  staff    384  6  5 22:22 .idea
    -rw-r--r--    1 hiroshi.wayama  staff   4950  4 29 12:00 AUTHORS.md
    -rw-r--r--    1 hiroshi.wayama  staff    460  8 24 00:00 CODE_OF_CONDUCT.md
    -rw-r--r--    1 hiroshi.wayama  staff   4659  8 24 00:00 CONTRIBUTING.md
    -rw-r--r--    1 hiroshi.wayama  staff   7732  4 29 12:00 GLOSSARY.md
    -rw-r--r--    1 hiroshi.wayama  staff   1162  4 29 12:00 LICENSE
    -rw-r--r--    1 hiroshi.wayama  staff     31  8 24 00:00 MANIFEST.in
    -rw-r--r--    1 hiroshi.wayama  staff   5619  8 24 00:00 NEWS.md
    -rw-r--r--    1 hiroshi.wayama  staff  21000  8 24 00:00 README.md
    -rw-r--r--    1 hiroshi.wayama  staff   2383  4 29 12:00 SECURITY.md
    -rw-r--r--    1 hiroshi.wayama  staff  19853  8 24 00:00 SETUP.md
    -rw-r--r--    1 hiroshi.wayama  staff   1882  8 24 00:00 conda.md
    drwxr-xr-x    5 hiroshi.wayama  staff    160  8 24 00:00 contrib
    drwxr-xr-x    7 hiroshi.wayama  staff    224  8 24 00:00 docs
    drwxr-xr-x   15 hiroshi.wayama  staff    480  8 24 00:02 examples
    -rw-r--r--    1 hiroshi.wayama  staff    110  8 24 00:00 pyproject.toml
    -rw-r--r--    1 hiroshi.wayama  staff   1312  8 24 00:28 reco_base.yaml
    drwxr-xr-x    9 hiroshi.wayama  staff    288  8 24 00:00 recommenders
    drwxr-xr-x    9 hiroshi.wayama  staff    288  4 29 12:00 scenarios
    -rw-r--r--    1 hiroshi.wayama  staff   3926  8 24 00:00 setup.py
    drwxr-xr-x    9 hiroshi.wayama  staff    288  8 24 00:00 tests
    drwxr-xr-x    7 hiroshi.wayama  staff    224  8 24 00:00 tools
    -rw-r--r--    1 hiroshi.wayama  staff   2274  8 24 00:00 tox.ini



```python
%load_ext autoreload
%autoreload 2

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
```

    System version: 3.6.11 | packaged by conda-forge | (default, Nov 27 2020, 18:51:43) 
    [GCC Clang 11.0.0]
    Pandas version: 1.1.5



```bash
%%bash
which python
```

    /Users/hiroshi.wayama/anaconda3/bin/python



```python
!which python
```

    /Users/hiroshi.wayama/anaconda3/bin/python


# 1 Load Data

SAR is intended to be used on interactions with the following schema:
`<User ID>, <Item ID>,<Time>,[<Event Type>], [<Event Weight>]`. 

Each row represents a single interaction between a user and an item. These interactions might be different types of events on an e-commerce website, such as a user clicking to view an item, adding it to a shopping basket, following a recommendation link, and so on. Each event type can be assigned a different weight, for example, we might assign a “buy” event a weight of 10, while a “view” event might only have a weight of 1.

The MovieLens dataset is well formatted interactions of Users providing Ratings to Movies (movie ratings are used as the event weight) - we will use it for the rest of the example.


```python
# top k items to recommend
TOP_K = 10

# Select MovieLens data size: 100k, 1m, 10m, or 20m
MOVIELENS_DATA_SIZE = '100k'
```

### 1.1 Download and use the MovieLens Dataset


```python
data = movielens.load_pandas_df(
    size=MOVIELENS_DATA_SIZE
)

# Convert the float precision to 32-bit in order to reduce memory consumption 
data['rating'] = data['rating'].astype(np.float32)

data.head()
```

    100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4.81k/4.81k [00:02<00:00, 2.28kKB/s]





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>userID</th>
      <th>itemID</th>
      <th>rating</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>196</td>
      <td>242</td>
      <td>3.0</td>
      <td>881250949</td>
    </tr>
    <tr>
      <th>1</th>
      <td>186</td>
      <td>302</td>
      <td>3.0</td>
      <td>891717742</td>
    </tr>
    <tr>
      <th>2</th>
      <td>22</td>
      <td>377</td>
      <td>1.0</td>
      <td>878887116</td>
    </tr>
    <tr>
      <th>3</th>
      <td>244</td>
      <td>51</td>
      <td>2.0</td>
      <td>880606923</td>
    </tr>
    <tr>
      <th>4</th>
      <td>166</td>
      <td>346</td>
      <td>1.0</td>
      <td>886397596</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>userID</th>
      <th>itemID</th>
      <th>rating</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>99995</th>
      <td>880</td>
      <td>476</td>
      <td>3.0</td>
      <td>880175444</td>
    </tr>
    <tr>
      <th>99996</th>
      <td>716</td>
      <td>204</td>
      <td>5.0</td>
      <td>879795543</td>
    </tr>
    <tr>
      <th>99997</th>
      <td>276</td>
      <td>1090</td>
      <td>1.0</td>
      <td>874795795</td>
    </tr>
    <tr>
      <th>99998</th>
      <td>13</td>
      <td>225</td>
      <td>2.0</td>
      <td>882399156</td>
    </tr>
    <tr>
      <th>99999</th>
      <td>12</td>
      <td>203</td>
      <td>3.0</td>
      <td>879959583</td>
    </tr>
  </tbody>
</table>
</div>



### 1.2 Split the data using the python random splitter provided in utilities:

We split the full dataset into a `train` and `test` dataset to evaluate performance of the algorithm against a held-out set not seen during training. Because SAR generates recommendations based on user preferences, all users that are in the test set must also exist in the training set. For this case, we can use the provided `python_stratified_split` function which holds out a percentage (in this case 25%) of items from each user, but ensures all users are in both `train` and `test` datasets. Other options are available in the `dataset.python_splitters` module which provide more control over how the split occurs.


```python
train, test = python_stratified_split(data, ratio=0.75, col_user='userID', col_item='itemID', seed=42)
```


```python
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
```

    
    Train:
    Total Ratings: 74992
    Unique Users: 943
    Unique Items: 1649
    
    Test:
    Total Ratings: 25008
    Unique Users: 943
    Unique Items: 1444
    


# 2 Train the SAR Model

### 2.1 Instantiate the SAR algorithm and set the index

We will use the single node implementation of SAR and specify the column names to match our dataset (timestamp is an optional column that is used and can be removed if your dataset does not contain it).

Other options are specified to control the behavior of the algorithm as described in the [deep dive notebook](../02_model_collaborative_filtering/sar_deep_dive.ipynb).


```python
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
```

### 2.2 Train the SAR model on our training data, and get the top-k recommendations for our testing data

SAR first computes an item-to-item ***co-occurence matrix***. Co-occurence represents the number of times two items appear together for any given user. Once we have the co-occurence matrix, we compute an ***item similarity matrix*** by rescaling the cooccurences by a given metric (Jaccard similarity in this example). 

We also compute an ***affinity matrix*** to capture the strength of the relationship between each user and each item. Affinity is driven by different types (like *rating* or *viewing* a movie), and by the time of the event. 

Recommendations are achieved by multiplying the affinity matrix $A$ and the similarity matrix $S$. The result is a ***recommendation score matrix*** $R$. We compute the ***top-k*** results for each user in the `recommend_k_items` function seen below.

A full walkthrough of the SAR algorithm can be found [here](../02_model_collaborative_filtering/sar_deep_dive.ipynb).


```python
with Timer() as train_time:
    model.fit(train)

print("Took {} seconds for training.".format(train_time.interval))
```

    2021-08-24 10:19:27,632 INFO     Collecting user affinity matrix
    2021-08-24 10:19:27,634 INFO     Calculating time-decayed affinities
    2021-08-24 10:19:27,667 INFO     Creating index columns
    2021-08-24 10:19:27,752 INFO     Calculating normalization factors
    2021-08-24 10:19:27,791 INFO     Building user affinity sparse matrix
    2021-08-24 10:19:27,796 INFO     Calculating item co-occurrence
    2021-08-24 10:19:27,959 INFO     Calculating item similarity
    2021-08-24 10:19:27,960 INFO     Using jaccard based similarity
    2021-08-24 10:19:28,041 INFO     Done training


    Took 0.4110744728241116 seconds for training.



```python
with Timer() as test_time:
    top_k = model.recommend_k_items(test, remove_seen=True)

print("Took {} seconds for prediction.".format(test_time.interval))
```

    2021-08-24 10:19:28,077 INFO     Calculating recommendation scores
    2021-08-24 10:19:28,233 INFO     Removing seen items


    Took 0.19810393685474992 seconds for prediction.



```python
top_k.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>userID</th>
      <th>itemID</th>
      <th>prediction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>204</td>
      <td>3.231405</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>89</td>
      <td>3.199445</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>11</td>
      <td>3.154097</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>367</td>
      <td>3.113913</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>423</td>
      <td>3.054493</td>
    </tr>
  </tbody>
</table>
</div>



### 2.3. Evaluate how well SAR performs

We evaluate how well SAR performs for a few common ranking metrics provided in the `python_evaluation` module. We will consider the Mean Average Precision (MAP), Normalized Discounted Cumalative Gain (NDCG), Precision, and Recall for the top-k items per user we computed with SAR. User, item and rating column names are specified in each evaluation method.


```python
eval_map = map_at_k(test, top_k, col_user='userID', col_item='itemID', col_rating='rating', k=TOP_K)
```


```python
eval_ndcg = ndcg_at_k(test, top_k, col_user='userID', col_item='itemID', col_rating='rating', k=TOP_K)
```


```python
eval_precision = precision_at_k(test, top_k, col_user='userID', col_item='itemID', col_rating='rating', k=TOP_K)
```


```python
eval_recall = recall_at_k(test, top_k, col_user='userID', col_item='itemID', col_rating='rating', k=TOP_K)
```


```python
eval_rmse = rmse(test, top_k, col_user='userID', col_item='itemID', col_rating='rating')
```


```python
eval_mae = mae(test, top_k, col_user='userID', col_item='itemID', col_rating='rating')
```


```python
eval_rsquared = rsquared(test, top_k, col_user='userID', col_item='itemID', col_rating='rating')
```


```python
eval_exp_var = exp_var(test, top_k, col_user='userID', col_item='itemID', col_rating='rating')
```


```python
positivity_threshold = 2
test_bin = test.copy()
test_bin['rating'] = binarize(test_bin['rating'], positivity_threshold)

top_k_prob = top_k.copy()
top_k_prob['prediction'] = minmax_scale(
    top_k_prob['prediction'].astype(float)
)

eval_logloss = logloss(test_bin, top_k_prob, col_user='userID', col_item='itemID', col_rating='rating')
```


```python
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
```

    Model:	
    Top K:	10
    MAP:	0.110591
    NDCG:	0.382461
    Precision@K:	0.330753
    Recall@K:	0.176385
    RMSE:	1.253805
    MAE:	1.048484
    R2:	-0.569363
    Exp var:	0.030474
    Logloss:	0.542861



```python
# Now let's look at the results for a specific user
user_id = 876

ground_truth = test[test['userID']==user_id].sort_values(by='rating', ascending=False)[:TOP_K]
prediction = model.recommend_k_items(pd.DataFrame(dict(userID=[user_id])), remove_seen=True) 
pd.merge(ground_truth, prediction, on=['userID', 'itemID'], how='left')
```

    2021-08-24 10:19:30,053 INFO     Calculating recommendation scores
    2021-08-24 10:19:30,068 INFO     Removing seen items





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>userID</th>
      <th>itemID</th>
      <th>rating</th>
      <th>timestamp</th>
      <th>prediction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>876</td>
      <td>523</td>
      <td>5.0</td>
      <td>879428378</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>876</td>
      <td>529</td>
      <td>4.0</td>
      <td>879428451</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>876</td>
      <td>174</td>
      <td>4.0</td>
      <td>879428378</td>
      <td>3.702239</td>
    </tr>
    <tr>
      <th>3</th>
      <td>876</td>
      <td>276</td>
      <td>4.0</td>
      <td>879428354</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>876</td>
      <td>288</td>
      <td>3.0</td>
      <td>879428101</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



Above, we see that one of the highest rated items from the test set was recovered by the model's top-k recommendations, however the others were not. Offline evaluations are difficult as they can only use what was seen previously in the test set and may not represent the user's actual preferences across the entire set of items. Adjustments to how the data is split, algorithm is used and hyper-parameters can improve the results here. 


```python
# Record results with papermill for tests - ignore this cell
sb.glue("map", eval_map)
sb.glue("ndcg", eval_ndcg)
sb.glue("precision", eval_precision)
sb.glue("recall", eval_recall)
sb.glue("train_time", train_time.interval)
sb.glue("test_time", test_time.interval)
```














```python

```


```python

```


```python

```

$$
\begin{array}{|l|l|l|}
\hline 1 & \mathrm{a} & \mathrm{aa} \\
\hline 2 & \mathrm{~b} & \mathrm{bb} \\
\hline 3 & \mathrm{c} & \mathrm{cc} \\
\hline 4 & \mathrm{~d} & \mathrm{dd} \\
\hline
\end{array}
$$

$$
\begin{array}{|c|c|c|}
\hline 1 & \mathrm{a} & \mathrm{aa} \\
\hline 2 & \mathrm{~b} & \mathrm{bb} \\
\hline 3 & \mathrm{c} & \mathrm{cc} \\
\hline 3 & \mathrm{c} & \mathrm{cc} \\
\hline 4 & \mathrm{~d} & \mathrm{dd} \\
\hline 3 & \mathrm{c} & \mathrm{cc} \\
\hline 3 & \mathrm{c} & \mathrm{cc} \\
\hline 3 & \mathrm{c} & \mathrm{cc} \\
\hline 3 & \mathrm{c} & \mathrm{cc} \\
\hline 3 & \mathrm{c} & \mathrm{cc} \\
\hline 3 & \mathrm{c} & \mathrm{cc} \\
\hline 3 & \mathrm{c} & \mathrm{cc} \\
\hline 3 & \mathrm{c} & \mathrm{cc} \\
\hline 3 & \mathrm{c} & \mathrm{cc} \\
\hline
\end{array}
$$


```python
%%time
1 + 1
```

    CPU times: user 3 µs, sys: 0 ns, total: 3 µs
    Wall time: 11.9 µs





    2




```python

```
