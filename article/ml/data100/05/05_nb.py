#!/usr/bin/env python
# coding: utf-8

# ## 第5章 顧客の退会を予測する10本ノック
# 
# この記事は[「Python実践データ分析100本ノック」](https://www.amazon.co.jp/dp/B07ZSGSN9S/ref=dp-kindle-redirect?_encoding=UTF8&btkr=1)の演習を実際にやってみたという内容になっています。今まで自己流でやってきましたが、一度他の方々がどのような考え方やコーディングをしているのか勉強してみようと思ってやってみました。本書は実際の業務に活用する上でとても参考になる内容だと思っています。データ分析に関わる仕事をしたい方にお勧めしたいです。
# 
# 本演習で利用しているデータは本サイトからは利用できません。ぜひとも「Python実践データ分析１００本ノック」を購入し、本に沿ってダウンロードして自分の手でコーディングしてみてください。（私は決して回し者ではないので安心してください笑）
# 
# 前章（4章）では、クラスタリングと線形回帰を実行してみました。今回は決定木のようです。データ分析や予測において最初に使われるのがXGBoostやLightGBM、ランダムフォレストであり、それらの手法の基礎となっているのが決定木です。
# 
# ### github
# - jupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/ml/data100/05/05_nb.ipynb)
# 
# ### google colaboratory
# - google colaboratory で実行する場合は[こちら](https://colab.research.google.com/github/hiroshi0530/wa-src/blob/master/ml/data100/05/05_nb.ipynb)
# 
# ### 筆者の環境

# In[1]:


get_ipython().system('sw_vers')


# In[2]:


get_ipython().system('python -V')


# 基本的なライブラリをインポートしそのバージョンを確認しておきます。

# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")

import matplotlib
import matplotlib.pyplot as plt
import scipy
import numpy as np
import pandas as pd

print('matplotlib version :', matplotlib.__version__)
print('scipy version :', scipy.__version__)
print('numpy version :', np.__version__)
print('pandas version :', pd.__version__)


# ## 解答

# ### ノック : 41 データを読み込んで利用で他を整形しよう

# In[4]:


customer = pd.read_csv('customer_join.csv')
uselog_months = pd.read_csv('use_log_months.csv')


# In[5]:


customer.head()


# In[6]:


uselog_months.head()


# In[7]:


year_months = list(uselog_months['年月'].unique())
uselog = pd.DataFrame()

for i in range(1, len(year_months)):
  tmp = uselog_months.loc[uselog_months['年月'] == year_months[i]].copy()
  tmp.rename(columns={'count': 'count_0'}, inplace=True)
  tmp_before = uselog_months.loc[uselog_months['年月'] == year_months[i - 1]].copy()
  del tmp_before['年月']
  tmp_before.rename(columns={'count': 'count_1'}, inplace=True)
  tmp = pd.merge(tmp, tmp_before, on='customer_id', how='left')
  uselog = pd.concat([uselog, tmp], ignore_index=True)


# In[8]:


uselog.head()


# ### ノック : 42 大会前日の大会顧客データを作成しよう

# In[9]:


from dateutil.relativedelta import relativedelta
exit_customer = customer.loc[customer['is_deleted'] == 1].copy()

exit_customer['exit_date'] = None
exit_customer['end_date'] = pd.to_datetime(exit_customer['end_date'].copy())

for i in range(len(exit_customer)):
  col_id_exit_date = exit_customer.columns.get_loc('exit_date')
  col_id_end_date = exit_customer.columns.get_loc('end_date')
  exit_customer.iloc[i, col_id_exit_date] = exit_customer.iloc[i, col_id_end_date] - relativedelta(months=1)

exit_customer['年月'] = exit_customer['exit_date'].dt.strftime('%Y%m')
uselog['年月'] = uselog['年月'].astype(str)
exit_uselog = pd.merge(uselog, exit_customer, on=['customer_id', '年月'], how='left')


# In[10]:


len(uselog)


# In[11]:


exit_uselog.head()


# 欠損値を除去します。

# In[12]:


exit_uselog = exit_uselog.dropna(subset=['name'])
print(len(exit_uselog))
print(len(exit_uselog['customer_id'].unique()))
exit_uselog.head()


# ### ノック : 43 継続顧客のデータを作成しよう

# In[13]:


conti_customer = customer.loc[customer['is_deleted'] == 0]
conti_uselog = pd.merge(uselog, conti_customer, on=['customer_id'], how='left')


# In[14]:


conti_uselog.head()


# In[15]:


print(len(conti_uselog))


# 欠損値を削除します。

# In[16]:


conti_uselog = conti_uselog.dropna(subset=['name'])
print(len(conti_uselog))


# pandasのsampleメソッドで、fracオプションを使うと、任意の割合のデータをサンプリングする事が出来ます。

# In[17]:


conti_uselog = conti_uselog.sample(frac=1).reset_index(drop=True)
conti_uselog = conti_uselog.drop_duplicates(subset='customer_id')
print(len(conti_uselog))
conti_uselog.head()


# 退会顧客と継続顧客のデータの結合を行います。

# In[18]:


predict_data = pd.concat([conti_uselog, exit_uselog], ignore_index=True)
print(len(predict_data))
predict_data.head()


# ### ノック : 44 予測する月の在籍期間を作成しよう

# In[19]:


predict_data['period'] = 0
predict_data['now_date'] = pd.to_datetime(predict_data['年月'], format="%Y%m")

predict_data['start_date'] = pd.to_datetime(predict_data['start_date'])

for i in range(len(predict_data)):
  delta = relativedelta(predict_data['now_date'][i], predict_data['start_date'][i])
  predict_data.iloc[i, predict_data.columns.get_loc('period')] = int(delta.years * 12 + delta.months)

predict_data.head()


# ### ノック : 45 欠損値を除去しよう
# 
# 欠損値の確認をします

# In[20]:


predict_data.isna().sum()


# これを見ると、count_1とend_date、exit_dateに欠損値が存在していることがわかります。end_dateとexit_dateは退会顧客しか値を持っていません。count_1に欠損値を持っているデータだけを削除します。

# In[21]:


predict_data = predict_data.dropna(subset=['count_1'])
predict_data.isna().sum()


# ### ノック : 46 文字列型の変数を処理できるように整形しよう

# In[22]:


target_col = ['campaign_name', 'class_name', 'gender', 'count_1', 'routine_flg', 'period', 'is_deleted']
predict_data = predict_data[target_col]
predict_data.head()


# カテゴリカル変数を利用して、ダミー変数を作成します。get_dummyメソッドは文字列データを読み取って、それをダミー変数化してくれます。データ分析の現場ではしばしば利用されるとても便利な関数です。

# In[23]:


predict_data = pd.get_dummies(predict_data)
predict_data.head()


# In[24]:


predict_data.columns


# 文字列をデータとして持つカラムだけダミー変数化されています。

# 本書では、わざわざ明示的に示されてなくてもわかるデータ（gender_Fの0か1がわかれば、gender_Mがわかるので、gender_Mは必要ない）を削除しています。
# 個人的にはあっても問題ないと思いますが、本書に従って削除します。

# In[25]:


del predict_data['campaign_name_通常']
del predict_data['class_name_ナイト']
del predict_data['gender_M']

predict_data.head()


# ### ノック : 47 決定木を用いて大会予測モデルを作成してみよう
# 
# 実際に決定木のアルゴリズムを実行してみます。

# In[26]:


from sklearn.tree import DecisionTreeClassifier
import sklearn.model_selection

exit = predict_data.loc[predict_data['is_deleted'] == 1]

# 退会人数と同じ数だけサンプリング
conti = predict_data.loc[predict_data['is_deleted'] == 0].sample(len(exit))

X = pd.concat([exit, conti], ignore_index=True)
y = X['is_deleted']

# ターゲット変数を削除
del X['is_deleted']

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X,y)

model = DecisionTreeClassifier(random_state=0)
model.fit(X_train, y_train)

# 予測値
y_test_pred = model.predict(X_test)

print(y_test_pred)


# 正解データと予測されたデータの比較を行います。

# In[27]:


results_test = pd.DataFrame({
  'y_test': y_test,
  'y_pred': y_test_pred,
})
results_test.head()


# ### ノック : 48 予測モデルの評価を行い、モデルのチューニングをしてみよう
# 
# 予測データと実際のデータが一致する割合を計算します。

# In[28]:


correct = len(results_test.loc[results_test['y_test'] == results_test['y_pred']])

data_count = len(results_test)

score_test = correct / data_count
print(score_test)


# In[29]:


model.score(X_test, y_test)


# In[30]:


model.score(X_train, y_train)


# 学習用データを用いた場合が98％の一致率で、テスト用データの場合は89％となっています。学習用データに過剰適合（過学習）しています。過学習を防ぐためによく利用される深さの最大値を5に設定します。

# In[31]:


X = pd.concat([exit, conti], ignore_index=True)
y = X['is_deleted']

# ターゲット変数を削除
del X['is_deleted']

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X,y)

model = DecisionTreeClassifier(random_state=0, max_depth=5)
model.fit(X_train, y_train)

print(model.score(X_test, y_test))
print(model.score(X_train, y_train))


# モデルを作る際に考慮されていないテスト用データに対しても、同じ一致率を得ることが出来ました。

# ### ノック : 49 モデルに寄与している変数を確認しよう
# 
# 決定木を利用している場合はfeature_importances_で変数の寄与度を得ることが出来ます。

# In[32]:


importance = pd.DataFrame({
  'feature_names': X.columns,
  'coefficient': model.feature_importances_
})
print(importance)


# 1ヶ月前の利用回数、定期利用、利用期間のそれぞれの寄与度が高いことがわかります。

# ### ノック : 50 顧客の退会を予測しよう
# 
# 実際に未知のデータに対して予測してみます。予測したい適当なデータを作ります。

# In[33]:


count_1 = 3
routing_flg = 1
period = 10
campaign_name = '入会費無料'
class_name = 'オールタイム'
gender = 'M'


# カテゴリカル変数を利用しているので、与えられた変数をカテゴライズ化します。

# In[34]:


if campaign_name == '入会費半額':
  campaign_name_list = [1,0]
elif campaign_name == '入会費無料':
  campaign_name_list = [0,1]
elif campaign_name == '通常':
  campaign_name_list = [0,0]


# In[35]:


if class_name == 'オールタイム':
  class_name_list = [1,0]
elif class_name == 'デイタイム':
  class_name_list = [0,1]
elif class_name == 'ナイト':
  class_name_list = [0,0]


# In[36]:


if gender == 'F':
  gender_list = [1]
elif gender == 'M':
  gender_list = [0]


# In[37]:


input_data = [count_1, routing_flg, period]
input_data.extend(campaign_name_list)
input_data.extend(class_name_list)
input_data.extend(gender_list)

input_data


# In[38]:


print(model.predict([input_data]))
print(model.predict_proba([input_data]))


# 今回の決めたデータでは1で退会が予想されています。その確率も96％でかなり高い確率です。

# ## 関連記事
# - [第1章 ウェブからの注文数を分析する10本ノック](/ml/data100/01/)
# - [第2章 小売店のデータでデータ加工を行う10本ノック](/ml/data100/02/)
# - [第3章 顧客の全体像を把握する10本ノック](/ml/data100/03/)
# - [第4章 顧客の行動を予測する10本ノック](/ml/data100/04/)
# - [第5章 顧客の退会を予測する10本ノック](/ml/data100/05/)
# - [第6章 物流の最適ルートをコンサルティングする10本ノック](/ml/data100/06/)
# - [第7章 ロジスティクスネットワークの最適設計を行う10本ノック](/ml/data100/07/)
# - [第8章 数値シミュレーションで消費者行動を予測する10本ノック](/ml/data100/08/)
# - [第9章 潜在顧客を把握するための画像認識10本ノック](/ml/data100/09/)
# - [第10章 アンケート分析を行うための自然言語処理10本ノック](/ml/data100/10/)
