#!/usr/bin/env python
# coding: utf-8

# ## LightGBMのテンプレート
# 
# テーブルデータの解析でよく利用されるLightGBMですが、最近よく利用することになったので、一度テンプレートとしてまとめおきます。
# 
# (2022/2/5更新: とある優秀なDataScientistの方からShapによる特徴量の解析方法を教えてもらったので、テンプレートに追加しておきます)
# 
# - データの取得
# - モデルの作成
# - Cross Validation
# - TESTの評価
# - Shapによる解析
# 
# ### github
# - githubのjupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/article/library/python/014/014_nb.ipynb)
# 
# ### google colaboratory
# - google colaboratory で実行する場合は[こちら](https://colab.research.google.com/github/hiroshi0530/wa-src/blob/master/article/library/python/014/014_nb.ipynb)
# 
# ### 筆者の環境

# In[1]:


get_ipython().system('sw_vers')


# In[2]:


get_ipython().system('python -V')


# In[3]:


import sys
sys.executable


# ## ライブラリの読み込み

# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")

import time
import json
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import japanize_matplotlib
import snap

import lightgbm as lgb

from sklearn import datasets
from sklearn import metrics

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score

from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

from sklearn.datasets import fetch_california_housing
import pandas as pd

import seaborn as sns
sns.set_style('darkgrid')
sns.set(font='IPAexGothic')

import warnings
warnings.filterwarnings('ignore')


# ### データの読み込み
# 
# データ分析でよく利用されるボストンのデータセットを利用します。目的変数は「CRIM」で人口一人あたりの犯罪発生件数です。

# In[5]:


california_housing = fetch_california_housing(as_frame=True)
df = pd.DataFrame(california_housing['data'], columns=california_housing.feature_names)
df_target = pd.Series(california_housing['target'])


# In[6]:


df


# In[7]:


df.shape


# In[8]:


df.head()


# ### 定数
# 
# ランダム値を固定するためのシードの値を設定します。

# In[9]:


# const
seed = 123
random_state = 123
n_splits=5
test_size=0.2


# ### シードの固定関数

# In[10]:


def set_seed(seed=seed):
  os.environ["PYTHONHASHSEED"] = str(seed) 
  np.random.seed(seed)
  random.seed(seed)


# ### データの分割
# TrainとTest用のデータに分割します。

# In[11]:


# 目的変数の設定
X_data = df
X_data.head()


# In[12]:


# 説明変数の設定
Y_data = df_target
Y_data.head()


# ### データの準備 

# In[13]:


X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=test_size, random_state=random_state)


# In[14]:


X_train.head()


# In[15]:


X_test.head()


# In[16]:


Y_train.head()


# In[17]:


Y_test.head()


# ## モデルの作成
# 
# ### Cross Validation
# 
# 交差検証を行う。

# In[44]:


get_ipython().run_cell_magic('time', '', "\nparams = {\n  'random_state': random_state,\n  'objective': 'regression',\n  'boosting_type': 'gbdt',\n  'metric': {'rmse'},\n  'verbosity': -1,\n  'bagging_freq': 1,\n  'feature_fraction': 0.8,\n  'max_depth': 8,\n  'min_data_in_leaf': 25,\n  'num_leaves': 256,\n  'learning_rate': 0.07,\n  'lambda_l1': 0.2,\n  'lambda_l2': 0.5,\n}\n\nmodel_list = []\nkf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)\n\nfor _index, (_train_index, _val_index) in enumerate(kf.split(X_train, Y_train)):\n \n   _X_train = X_train.iloc[_train_index]\n   _Y_train = Y_train.iloc[_train_index]\n \n   _X_val = X_train.iloc[_val_index]\n   _Y_val = Y_train.iloc[_val_index]\n \n   lgb_train = lgb.Dataset(_X_train, _Y_train)\n   lgb_val = lgb.Dataset(_X_val, _Y_val, reference=lgb_train)\n  \n   lgb_results = {}   \n\n   model = lgb.train(\n     params, \n     train_set=lgb_train, \n     valid_sets=[lgb_train, lgb_val],\n     verbose_eval=-1,\n     num_boost_round=1000,\n     early_stopping_rounds=100,\n     valid_names=['Train', 'Val'],\n     evals_result=lgb_results\n   )\n \n  # CVの各モデルの保存\n   model_list.append(model)\n  \n   loss_train = lgb_results['Train']['rmse']\n   loss_val = lgb_results['Val']['rmse']\n   best_iteration = model.best_iteration\n     \n   # グラフ\n   plt.rcParams['font.size'] = 14\n   fig = plt.figure()\n   ax1 = fig.add_subplot(111)\n   ax1.grid()\n   ax1.grid(axis='both')\n   ax1.set_xlabel('Iteration')\n   ax1.set_ylabel('rmse')\n   ax1.plot(loss_train, label='train loss')\n   ax1.plot(loss_val, label='valid loss')\n   plt.show()\n  \n   # 真値と予測値の表示\n  # df_pred = pd.DataFrame({'CRIM':y_test,'CRIM_pred':y_pred})\n  # display(df_pred)\n  \n  # 散布図を描画(真値 vs 予測値)\n   y_val = model.predict(_X_val, num_iteration=model.best_iteration)\n   plt.plot(y_val, y_val, color = 'red', label = 'x=y') # 直線y = x (真値と予測値が同じ場合は直線状に点がプロットされる)\n   plt.scatter(y_val,_Y_val, s=2)\n   plt.xlabel('y_val')\n   plt.ylabel('_Y_val')\n   plt.title('y_val vs _Y_val')\n   plt.show()")


# In[33]:


lgb_results.keys()


# In[35]:


y_val.shape


# ## 特徴量の重要度を可視化

# In[19]:


# 特徴量重要度を保管する dataframe を用意
# https://www.sairablog.com/article/lightgbm-sklearn-kaggle-classification.html から抜粋
top_x = 50
feature_importances = pd.DataFrame()

for fold, model in enumerate(model_list):

    tmp = pd.DataFrame()
    tmp['feature'] = model.feature_name()
    tmp['importance'] = model.feature_importance(importance_type='gain')
    tmp['fold'] = fold

    feature_importances = feature_importances.append(tmp)

order = list(feature_importances.groupby("feature")["importance"].mean().sort_values(ascending=False).index)[:top_x]

plt.figure(figsize=(4, 4))
sns.barplot(x='importance', y='feature', data=feature_importances, order=order)
plt.title('LGBM importance')
plt.tight_layout()
plt.show()


# ## TESTデータを用いた推論

# In[20]:


_test_score_array = np.zeros(len(X_test))

for model in model_list:
  y_pred = model.predict(X_test, num_iteration=model.best_iteration)
  
  # testのscoreを格納
  _test_score_array += y_pred / n_splits

  # ソート
  _zip = zip(Y_test, y_pred)
  _zip = sorted(_zip, key=lambda x: x[1], reverse=True)
  Y_test_array, y_pred_array = zip(*_zip)

  Y_test_array = np.array(Y_test_array)
  y_pred_array = np.array(y_pred_array)


# ### TEST結果の特徴量の重要度を可視化

# In[21]:


# 特徴量重要度を保存する dataframe を用意
# https://www.sairablog.com/article/lightgbm-sklearn-kaggle-classification.html から抜粋
top_feature_num = 10
feature_importances = pd.DataFrame({
  'feature' : model.feature_name(),
  'importance': model.feature_importance(importance_type='gain'),
})

order = list(feature_importances.groupby("feature")["importance"].mean().sort_values(ascending=False).index)[:top_feature_num]

plt.figure(figsize=(4, 6))
sns.barplot(x='importance', y='feature', data=feature_importances, order=order)
plt.title('LGBM importance')
plt.tight_layout()
plt.show()


# ## 参考記事

# - https://www.sairablog.com/article/lightgbm-sklearn-kaggle-classification.html

# In[ ]:




