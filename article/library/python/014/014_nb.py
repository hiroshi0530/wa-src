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
# 読み込むデータとしてはこれまでボストンの住宅価格のデータセットを利用していましたが、最近depricatedになったので、新しく加わったカリフォルニアの住宅価格を利用する。

# In[5]:


california_housing = fetch_california_housing(as_frame=True)
df = pd.DataFrame(california_housing['data'], columns=california_housing.feature_names)
df_target = pd.Series(california_housing['target'])

# template
# 説明変数の設定
X_data = df
# 目的変数の設定
Y_data = df_target


# In[6]:


X_data.shape, Y_data.shape


# In[7]:


X_data.head()


# In[8]:


Y_data.head()


# ### 定数
# 
# ランダム値を固定するためのシード値や諸々の定数を設定。

# In[9]:


# const
seed = 123
random_state = 123
n_splits=5
test_size=0.2


# ### シードの固定関数
# 
# 一括でシードを設定するための関数。

# In[10]:


def set_seed(seed=seed):
  os.environ["PYTHONHASHSEED"] = str(seed) 
  np.random.seed(seed)
  random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  # GPUパフォーマンスが低下する場合はコメントアウト
  torch.backends.cudnn.deterministic = True
  torch.use_deterministic_algorithms = True


# ### データの分割
# TrainとTest用のデータに分割します。

# In[11]:


X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=test_size, random_state=random_state)


# In[12]:


X_train.head()


# In[13]:


X_test.head()


# In[14]:


Y_train.head()


# In[15]:


Y_test.head()


# ## Trainデータを利用したモデルの作成
# 
# ### Cross Validation
# 
# Trainデータに対して交差検証を行う。交差検証(Cross Validation)は、データ全体を分割し、一部を用いてモデルを作成し、残りのデータでバリデーションを行う方法である。データ全体をK分割し、K-1個を訓練用データ、残りの一つを妥当性検証用のデータに利用する、K-Fold Cross Validationを行う。
# 
# 各Foldデータに対して、損失関数の推移とモデルから計算された予測値と真値の散布図も描画する。

# In[16]:


get_ipython().run_cell_magic('time', '', "\nparams = {\n  'random_state': random_state,\n  'objective': 'regression',\n  'boosting_type': 'gbdt',\n  'metric': {'rmse'},\n  'verbosity': -1,\n  'bagging_freq': 1,\n  'feature_fraction': 0.8,\n  'max_depth': 8,\n  'min_data_in_leaf': 25,\n  'num_leaves': 256,\n  'learning_rate': 0.07,\n  'lambda_l1': 0.2,\n  'lambda_l2': 0.5,\n}\n\nmodel_list = []\nkf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)\n\nfor _index, (_train_index, _val_index) in enumerate(kf.split(X_train, Y_train)):\n \n   _X_train = X_train.iloc[_train_index]\n   _Y_train = Y_train.iloc[_train_index]\n \n   _X_val = X_train.iloc[_val_index]\n   _Y_val = Y_train.iloc[_val_index]\n \n   lgb_train = lgb.Dataset(_X_train, _Y_train)\n   lgb_val = lgb.Dataset(_X_val, _Y_val, reference=lgb_train)\n  \n   lgb_results = {}   \n\n   model = lgb.train(\n     params, \n     train_set=lgb_train, \n     valid_sets=[lgb_train, lgb_val],\n     verbose_eval=-1,\n     num_boost_round=1000,\n     early_stopping_rounds=100,\n     valid_names=['Train', 'Val'],\n     evals_result=lgb_results\n   )\n\n  # CVの各モデルの保存\n   model_list.append(model)\n\n   # 損失関数の推移\n   loss_train = lgb_results['Train']['rmse']\n   loss_val = lgb_results['Val']['rmse']\n   best_iteration = model.best_iteration\n    \n   plt.figure()\n   plt.xlabel('Iteration')\n   plt.ylabel('rmse')\n   plt.plot(loss_train, label='train loss')\n   plt.plot(loss_val, label='valid loss')\n   plt.title('kFold : {} \\n RMSE'.format(_index))\n   plt.legend()\n   plt.show()\n  \n  # 散布図\n   plt.figure(figsize=(4,4))\n   y_val = model.predict(_X_val, num_iteration=model.best_iteration)\n   plt.plot(y_val, y_val, color = 'red', label = '$y=x$')\n   plt.scatter(y_val,_Y_val, s=1)\n   plt.xlabel('予測値')\n   plt.ylabel('真値')\n   plt.title('kFold : {} \\n 予測値 vs 真値'.format(_index))\n   plt.legend()\n   plt.show()")


# ### 特徴量の重要度を可視化
# 
# modelに保存されているimportanceを可視化。こちらのサイトを参考。
# 
# - https://www.sairablog.com/article/lightgbm-sklearn-kaggle-classification.html 

# In[17]:


feature_importances = pd.DataFrame()

for fold, model in enumerate(model_list):

    tmp = pd.DataFrame()
    tmp['feature'] = model.feature_name()
    tmp['importance'] = model.feature_importance(importance_type='gain')
    tmp['fold'] = fold

    feature_importances = feature_importances.append(tmp)

order = list(feature_importances.groupby("feature")["importance"].mean().sort_values(ascending=False).index)

plt.figure(figsize=(4, 4))
sns.barplot(x='importance', y='feature', data=feature_importances, order=order)
plt.title('LGBM importance')
plt.tight_layout()
plt.show()


# ## TESTデータを用いた推論
# 
# 各Foldで生成されたモデルに対して、TESTデータを利用して、予測する。予測値は各Foldの平均とする。

# In[18]:


_test_score_array = np.zeros(len(X_test))

for model in model_list:
  y_pred = model.predict(X_test, num_iteration=model.best_iteration)
  
  _test_score_array += y_pred / n_splits

_test_score_array


# ### TEST結果の特徴量の重要度を可視化

# In[19]:


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


# ## Shapによる解析
# 
# 最近まで知らなかったのだが、Shapという解析方法を教えてもらったので早速テンプレートに追加する。
# 
# Shapを利用して、特徴量が結果に対してどのような影響を与えたか定量的に求める事ができる。
# モデルの生成理由などを説明するためによく利用される。

# In[20]:


import shap

shap.initjs()
explainer = shap.Explainer(model, _X_train)
shap_values = explainer(_X_train, check_additivity=False)


# ### summary plot
# 
# summaryで、特徴量毎にどの程度影響を与えたか可視化することができる。
# 
# 目的変数に対して、青が正の影響を、赤が負の影響を与えた事を表している。

# In[21]:


shap.summary_plot(shap_values, _X_train)


# また、以下の様にして、正と負のトータルとしてどのような影響を与えたか知ることができる。

# In[22]:


shap.summary_plot(shap_values, _X_train, plot_type="bar")


# ### waterfallによる解析
# 
# 各データ一つ一つに対して、特徴量の影響を可視化することができる。

# In[23]:


shap.plots.waterfall(shap_values[0], max_display=20)


# In[24]:


shap_values.shape


# In[25]:


Y_val = model.predict(_X_train, num_iteration=model.best_iteration)
df_result = pd.DataFrame({
  'index': _X_train.index.tolist(),
  'train_score': Y_val,
})


# 上位のデータと下位のデータを比較し、どの特徴量が効果があるか確認する。

# In[26]:


top = df_result.sort_values(by='train_score', ascending=False).index.tolist()
bottom = df_result.sort_values(by='train_score').index.tolist()


# ### 上位5件を表示

# In[27]:


for i in top[:5]:
  shap.plots.waterfall(shap_values[i], max_display=20)


# ### 下位の5件を表示

# In[28]:


for i in bottom[:5]:
  shap.plots.waterfall(shap_values[i], max_display=20)


# ### 特徴量の依存性確認
# 
# それぞれの特徴量とSHAP値の相関関係を可視化することができる。

# In[29]:


feature_importances = pd.DataFrame({
  'feature' : model.feature_name(),
  'importance': model.feature_importance(importance_type='gain'),
})
feature_list = list(feature_importances.groupby("feature")["importance"].mean().sort_values(ascending=False).index)
for feature in feature_list:
  shap.dependence_plot(feature, shap_values.values, _X_train)


# 回帰問題に対してLightGBMを利用する際のテンプレートを用意した。
# 今後も追記する項目が増えた場合は更新する。

# ## 参考記事

# - https://www.sairablog.com/article/lightgbm-sklearn-kaggle-classification.html
