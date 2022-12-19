```python
# set the environment path to find Recommenders
import sys

sys.path.append('../../../')

import itertools
import logging
import os

import numpy as np
import pandas as pd
import papermill as pm

from recommenders.datasets import movielens
from recommenders.datasets.python_splitters import python_stratified_split
from recommenders.evaluation.python_evaluation import map_at_k, ndcg_at_k, precision_at_k, recall_at_k
from recommenders.models.sar.sar_singlenode import SARSingleNode

print("System version: {}".format(sys.version))
print("Pandas version: {}".format(pd.__version__))
```


```python
# top k items to recommend
TOP_K = 10

# Select MovieLens data size: 100k, 1m, 10m, or 20m
MOVIELENS_DATA_SIZE = '100k'

data = movielens.load_pandas_df(
    size=MOVIELENS_DATA_SIZE,
    header=['UserId', 'MovieId', 'Rating', 'Timestamp'],
    title_col='Title'
)
```


```python
# Convert the float precision to 32-bit in order to reduce memory consumption 
data.loc[:, 'Rating'] = data['Rating'].astype(np.float32)
```


```python
data.head(100)
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
      <th>UserId</th>
      <th>MovieId</th>
      <th>Rating</th>
      <th>Timestamp</th>
      <th>Title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>196</td>
      <td>242</td>
      <td>3.0</td>
      <td>881250949</td>
      <td>Kolya (1996)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>63</td>
      <td>242</td>
      <td>3.0</td>
      <td>875747190</td>
      <td>Kolya (1996)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>226</td>
      <td>242</td>
      <td>5.0</td>
      <td>883888671</td>
      <td>Kolya (1996)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>154</td>
      <td>242</td>
      <td>3.0</td>
      <td>879138235</td>
      <td>Kolya (1996)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>306</td>
      <td>242</td>
      <td>5.0</td>
      <td>876503793</td>
      <td>Kolya (1996)</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>95</th>
      <td>485</td>
      <td>242</td>
      <td>5.0</td>
      <td>891040423</td>
      <td>Kolya (1996)</td>
    </tr>
    <tr>
      <th>96</th>
      <td>574</td>
      <td>242</td>
      <td>5.0</td>
      <td>891278860</td>
      <td>Kolya (1996)</td>
    </tr>
    <tr>
      <th>97</th>
      <td>758</td>
      <td>242</td>
      <td>3.0</td>
      <td>880672230</td>
      <td>Kolya (1996)</td>
    </tr>
    <tr>
      <th>98</th>
      <td>486</td>
      <td>242</td>
      <td>4.0</td>
      <td>879874018</td>
      <td>Kolya (1996)</td>
    </tr>
    <tr>
      <th>99</th>
      <td>13</td>
      <td>242</td>
      <td>2.0</td>
      <td>881515193</td>
      <td>Kolya (1996)</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 5 columns</p>
</div>




```python
data.tail(100)
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
      <th>UserId</th>
      <th>MovieId</th>
      <th>Rating</th>
      <th>Timestamp</th>
      <th>Title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>99900</th>
      <td>839</td>
      <td>1664</td>
      <td>1.0</td>
      <td>875752902</td>
      <td>8 Heads in a Duffel Bag (1997)</td>
    </tr>
    <tr>
      <th>99901</th>
      <td>880</td>
      <td>1664</td>
      <td>4.0</td>
      <td>892958799</td>
      <td>8 Heads in a Duffel Bag (1997)</td>
    </tr>
    <tr>
      <th>99902</th>
      <td>782</td>
      <td>1664</td>
      <td>4.0</td>
      <td>891499699</td>
      <td>8 Heads in a Duffel Bag (1997)</td>
    </tr>
    <tr>
      <th>99903</th>
      <td>870</td>
      <td>1664</td>
      <td>4.0</td>
      <td>890057322</td>
      <td>8 Heads in a Duffel Bag (1997)</td>
    </tr>
    <tr>
      <th>99904</th>
      <td>489</td>
      <td>1613</td>
      <td>4.0</td>
      <td>891449466</td>
      <td>Tokyo Fist (1995)</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>99995</th>
      <td>840</td>
      <td>1674</td>
      <td>4.0</td>
      <td>891211682</td>
      <td>Mamma Roma (1962)</td>
    </tr>
    <tr>
      <th>99996</th>
      <td>655</td>
      <td>1640</td>
      <td>3.0</td>
      <td>888474646</td>
      <td>Eighth Day, The (1996)</td>
    </tr>
    <tr>
      <th>99997</th>
      <td>655</td>
      <td>1637</td>
      <td>3.0</td>
      <td>888984255</td>
      <td>Girls Town (1996)</td>
    </tr>
    <tr>
      <th>99998</th>
      <td>655</td>
      <td>1630</td>
      <td>3.0</td>
      <td>887428735</td>
      <td>Silence of the Palace, The (Saimt el Qusur) (1...</td>
    </tr>
    <tr>
      <th>99999</th>
      <td>655</td>
      <td>1641</td>
      <td>3.0</td>
      <td>887427810</td>
      <td>Dadetown (1995)</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 5 columns</p>
</div>




```python

```


```python

```


```python

```


```python

```

<i>Copyright (c) Microsoft Corporation. All rights reserved.</i>

<i>Licensed under the MIT License.</i>

# SAR Single Node on MovieLens (Python, CPU)

In this example, we will walk through each step of the Simple Algorithm for Recommendation (SAR) algorithm using a Python single-node implementation.

SAR is a fast, scalable, adaptive algorithm for personalized recommendations based on user transaction history. It is powered by understanding the similarity between items, and recommending similar items to those a user has an existing affinity for.

## 1 SAR algorithm

The following figure presents a high-level architecture of SAR. 

At a very high level, two intermediate matrices are created and used to generate a set of recommendation scores:

- An item similarity matrix $S$ estimates item-item relationships.
- An affinity matrix $A$ estimates user-item relationships.

Recommendation scores are then created by computing the matrix multiplication $A\times S$.

Optional steps (e.g. "time decay" and "remove seen items") are described in the details below.

<img src="https://recodatasets.z20.web.core.windows.net/images/sar_schema.svg?sanitize=true">

### 1.1 Compute item co-occurrence and item similarity

SAR defines similarity based on item-to-item co-occurrence data. Co-occurrence is defined as the number of times two items appear together for a given user. We can represent the co-occurrence of all items as a $m\times m$ matrix $C$, where $c_{i,j}$ is the number of times item $i$ occurred with item $j$, and $m$ is the total number of items.

The co-occurence matric $C$ has the following properties:

- It is symmetric, so $c_{i,j} = c_{j,i}$
- It is nonnegative: $c_{i,j} \geq 0$
- The occurrences are at least as large as the co-occurrences. I.e., the largest element for each row (and column) is on the main diagonal: $\forall(i,j) C_{i,i},C_{j,j} \geq C_{i,j}$.

Once we have a co-occurrence matrix, an item similarity matrix $S$ can be obtained by rescaling the co-occurrences according to a given metric. Options for the metric include `Jaccard`, `lift`, and `counts` (meaning no rescaling).


If $c_{ii}$ and $c_{jj}$ are the $i$th and $j$th diagonal elements of $C$, the rescaling options are:

- `Jaccard`: $s_{ij}=\frac{c_{ij}}{(c_{ii}+c_{jj}-c_{ij})}$
- `lift`: $s_{ij}=\frac{c_{ij}}{(c_{ii} \times c_{jj})}$
- `counts`: $s_{ij}=c_{ij}$

In general, using `counts` as a similarity metric favours predictability, meaning that the most popular items will be recommended most of the time. `lift` by contrast favours discoverability/serendipity: an item that is less popular overall but highly favoured by a small subset of users is more likely to be recommended. `Jaccard` is a compromise between the two.


### 1.2 Compute user affinity scores

The affinity matrix in SAR captures the strength of the relationship between each individual user and the items that user has already interacted with. SAR incorporates two factors that can impact users' affinities: 

- It can consider information about the **type** of user-item interaction through differential weighting of different events (e.g. it may weigh events in which a user rated a particular item more heavily than events in which a user viewed the item).
- It can consider information about **when** a user-item event occurred (e.g. it may discount the value of events that take place in the distant past.

Formalizing these factors produces us an expression for user-item affinity:

$$a_{ij}=\sum_k w_k \left(\frac{1}{2}\right)^{\frac{t_0-t_k}{T}} $$

where the affinity $a_{ij}$ for user $i$ and item $j$ is the weighted sum of all $k$ events involving user $i$ and item $j$. $w_k$ represents the weight of a particular event, and the power of 2 term reflects the temporally-discounted event. The $(\frac{1}{2})^n$ scaling factor causes the parameter $T$ to serve as a half-life: events $T$ units before $t_0$ will be given half the weight as those taking place at $t_0$.

Repeating this computation for all $n$ users and $m$ items results in an $n\times m$ matrix $A$. Simplifications of the above expression can be obtained by setting all the weights equal to 1 (effectively ignoring event types), or by setting the half-life parameter $T$ to infinity (ignoring transaction times).

### 1.3 Remove seen item

Optionally we remove items which have already been seen in the training set, i.e. don't recommend items which have been previously bought by the user again.

### 1.4 Top-k item calculation

The personalized recommendations for a set of users can then be obtained by multiplying the affinity matrix ($A$) by the similarity matrix ($S$). The result is a recommendation score matrix, where each row corresponds to a user, each column corresponds to an item, and each entry corresponds to a user / item pair. Higher scores correspond to more strongly recommended items.

It is worth noting that the complexity of recommending operation depends on the data size. SAR algorithm itself has $O(n^3)$ complexity. Therefore the single-node implementation is not supposed to handle large dataset in a scalable manner. Whenever one uses the algorithm, it is recommended to run with sufficiently large memory. 

## 2 SAR single-node implementation

The SAR implementation illustrated in this notebook was developed in Python, primarily with Python packages like `numpy`, `pandas`, and `scipy` which are commonly used in most of the data analytics / machine learning tasks. Details of the implementation can be found in [Recommenders/recommenders/models/sar/sar_singlenode.py](../../recommenders/models/sar/sar_singlenode.py).

## 3 SAR single-node based movie recommender


```python
# set the environment path to find Recommenders
import sys

sys.path.append('../../../')

import itertools
import logging
import os

import numpy as np
import pandas as pd
import papermill as pm

from recommenders.datasets import movielens
from recommenders.datasets.python_splitters import python_stratified_split
from recommenders.evaluation.python_evaluation import map_at_k, ndcg_at_k, precision_at_k, recall_at_k
from recommenders.models.sar.sar_singlenode import SARSingleNode

print("System version: {}".format(sys.version))
print("Pandas version: {}".format(pd.__version__))
```

    System version: 3.6.11 | packaged by conda-forge | (default, Nov 27 2020, 18:51:43) 
    [GCC Clang 11.0.0]
    Pandas version: 1.1.5



```python
# top k items to recommend
TOP_K = 10

# Select MovieLens data size: 100k, 1m, 10m, or 20m
MOVIELENS_DATA_SIZE = '100k'
```

### 3.1 Load Data

SAR is intended to be used on interactions with the following schema:
`<User ID>, <Item ID>, <Time>`. 

Each row represents a single interaction between a user and an item. These interactions might be different types of events on an e-commerce website, such as a user clicking to view an item, adding it to a shopping basket, following a recommendation link, and so on. 

The MovieLens dataset is well formatted interactions of Users providing Ratings to Movies (movie ratings are used as the event weight) - we will use it for the rest of the example.


```python
data = movielens.load_pandas_df(
    size=MOVIELENS_DATA_SIZE,
    header=['UserId', 'MovieId', 'Rating', 'Timestamp'],
    title_col='Title'
)

# Convert the float precision to 32-bit in order to reduce memory consumption 
data.loc[:, 'Rating'] = data['Rating'].astype(np.float32)

data.head()
```

    100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4.81k/4.81k [00:03<00:00, 1.28kKB/s]





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
      <th>UserId</th>
      <th>MovieId</th>
      <th>Rating</th>
      <th>Timestamp</th>
      <th>Title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>196</td>
      <td>242</td>
      <td>3.0</td>
      <td>881250949</td>
      <td>Kolya (1996)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>63</td>
      <td>242</td>
      <td>3.0</td>
      <td>875747190</td>
      <td>Kolya (1996)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>226</td>
      <td>242</td>
      <td>5.0</td>
      <td>883888671</td>
      <td>Kolya (1996)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>154</td>
      <td>242</td>
      <td>3.0</td>
      <td>879138235</td>
      <td>Kolya (1996)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>306</td>
      <td>242</td>
      <td>5.0</td>
      <td>876503793</td>
      <td>Kolya (1996)</td>
    </tr>
  </tbody>
</table>
</div>



### 3.2 Split the data using the python random splitter provided in utilities:

We split the full dataset into a `train` and `test` dataset to evaluate performance of the algorithm against a held-out set not seen during training. Because SAR generates recommendations based on user preferences, all users that are in the test set must also exist in the training set. For this case, we can use the provided `python_stratified_split` function which holds out a percentage (in this case 25%) of items from each user, but ensures all users are in both `train` and `test` datasets. Other options are available in the `dataset.python_splitters` module which provide more control over how the split occurs.



```python
header = {
    "col_user": "UserId",
    "col_item": "MovieId",
    "col_rating": "Rating",
    "col_timestamp": "Timestamp",
    "col_prediction": "Prediction",
}
```


```python
train, test = python_stratified_split(data, ratio=0.75, col_user=header["col_user"], col_item=header["col_item"], seed=42)
```

In this case, for the illustration purpose, the following parameter values are used:

|Parameter|Value|Description|
|---------|---------|-------------|
|`similarity_type`|`jaccard`|Method used to calculate item similarity.|
|`time_decay_coefficient`|30|Period in days (term of $T$ shown in the formula of Section 1.2)|
|`time_now`|`None`|Time decay reference.|
|`timedecay_formula`|`True`|Whether time decay formula is used.|


```python
# set log level to INFO
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s %(levelname)-8s %(message)s')

model = SARSingleNode(
    similarity_type="jaccard", 
    time_decay_coefficient=30, 
    time_now=None, 
    timedecay_formula=True, 
    **header
)
```


```python
model.fit(train)
```

    2021-09-13 14:50:58,475 INFO     Collecting user affinity matrix
    2021-09-13 14:50:58,477 INFO     Calculating time-decayed affinities
    2021-09-13 14:50:58,506 INFO     Creating index columns
    2021-09-13 14:50:58,589 INFO     Building user affinity sparse matrix
    2021-09-13 14:50:58,600 INFO     Calculating item co-occurrence
    2021-09-13 14:50:58,747 INFO     Calculating item similarity
    2021-09-13 14:50:58,748 INFO     Using jaccard based similarity
    2021-09-13 14:50:58,821 INFO     Done training



```python
top_k = model.recommend_k_items(test, remove_seen=True)
```

    2021-09-13 14:50:58,827 INFO     Calculating recommendation scores
    2021-09-13 14:50:58,888 INFO     Removing seen items


The final output from the `recommend_k_items` method generates recommendation scores for each user-item pair, which are shown as follows.


```python
top_k_with_titles = (top_k.join(data[['MovieId', 'Title']].drop_duplicates().set_index('MovieId'), 
                                on='MovieId', 
                                how='inner').sort_values(by=['UserId', 'Prediction'], ascending=False))
display(top_k_with_titles.head(10))
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
      <th>UserId</th>
      <th>MovieId</th>
      <th>Prediction</th>
      <th>Title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>9420</th>
      <td>943</td>
      <td>82</td>
      <td>21.313228</td>
      <td>Jurassic Park (1993)</td>
    </tr>
    <tr>
      <th>9421</th>
      <td>943</td>
      <td>403</td>
      <td>21.158839</td>
      <td>Batman (1989)</td>
    </tr>
    <tr>
      <th>9422</th>
      <td>943</td>
      <td>568</td>
      <td>20.962922</td>
      <td>Speed (1994)</td>
    </tr>
    <tr>
      <th>9423</th>
      <td>943</td>
      <td>423</td>
      <td>20.162170</td>
      <td>E.T. the Extra-Terrestrial (1982)</td>
    </tr>
    <tr>
      <th>9424</th>
      <td>943</td>
      <td>89</td>
      <td>19.890513</td>
      <td>Blade Runner (1982)</td>
    </tr>
    <tr>
      <th>9425</th>
      <td>943</td>
      <td>393</td>
      <td>19.832944</td>
      <td>Mrs. Doubtfire (1993)</td>
    </tr>
    <tr>
      <th>9426</th>
      <td>943</td>
      <td>11</td>
      <td>19.570244</td>
      <td>Seven (Se7en) (1995)</td>
    </tr>
    <tr>
      <th>9427</th>
      <td>943</td>
      <td>71</td>
      <td>19.553877</td>
      <td>Lion King, The (1994)</td>
    </tr>
    <tr>
      <th>9428</th>
      <td>943</td>
      <td>202</td>
      <td>19.422129</td>
      <td>Groundhog Day (1993)</td>
    </tr>
    <tr>
      <th>9429</th>
      <td>943</td>
      <td>238</td>
      <td>19.115604</td>
      <td>Raising Arizona (1987)</td>
    </tr>
  </tbody>
</table>
</div>


### 3.3 Evaluate the results

It should be known that the recommendation scores generated by multiplying the item similarity matrix $S$ and the user affinity matrix $A$ **DOES NOT** have the same scale with the original explicit ratings in the movielens dataset. That is to say, SAR algorithm is meant for the task of *recommending relevent items to users* rather than *predicting explicit ratings for user-item pairs*. 

To this end, ranking metrics like precision@k, recall@k, etc., are more applicable to evaluate SAR algorithm. The following illustrates how to evaluate SAR model by using the evaluation functions provided in the `recommenders`.


```python
# all ranking metrics have the same arguments
args = [test, top_k]
kwargs = dict(col_user='UserId', 
              col_item='MovieId', 
              col_rating='Rating', 
              col_prediction='Prediction', 
              relevancy_method='top_k', 
              k=TOP_K)

eval_map = map_at_k(*args, **kwargs)
eval_ndcg = ndcg_at_k(*args, **kwargs)
eval_precision = precision_at_k(*args, **kwargs)
eval_recall = recall_at_k(*args, **kwargs)
```


```python
print(f"Model:",
      f"Top K:\t\t {TOP_K}",
      f"MAP:\t\t {eval_map:f}",
      f"NDCG:\t\t {eval_ndcg:f}",
      f"Precision@K:\t {eval_precision:f}",
      f"Recall@K:\t {eval_recall:f}", sep='\n')
```

    Model:
    Top K:		 10
    MAP:		 0.095544
    NDCG:		 0.350232
    Precision@K:	 0.305726
    Recall@K:	 0.164690


## References
Note SAR is a combinational algorithm that implements different industry heuristics. The followings are references that may be helpful in understanding the SAR logic and implementation. 

1. Badrul Sarwar, *et al*, "Item-based collaborative filtering recommendation algorithms", WWW, 2001.
2. Scipy (sparse matrix), url: https://docs.scipy.org/doc/scipy/reference/sparse.html
3. Asela Gunawardana and Guy Shani, "A survey of accuracy evaluation metrics of recommendation tasks", The Journal of Machine Learning Research, vol. 10, pp 2935-2962, 2009.	


```python

```


```python

```
