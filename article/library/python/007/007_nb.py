
# coding: utf-8

# ## pandasのSettingWithCopyWarningのワーニングについて
# 
# Python Tips
# 
# pythonを利用する上で、便利な表記などの個人的なメモです。基本的な部分は触れていません。対象も自分が便利だなと思ったものに限定されます。
# 
# ### github
# - githubのjupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/article/library/python/07/07_nb.ipynb)
# 
# ### google colaboratory
# - google colaboratory で実行する場合は[こちら](https://colab.research.google.com/github/hiroshi0530/wa-src/blob/master/article/library/python/07/07_nb.ipynb)
# 
# ### 筆者の環境

# In[1]:


get_ipython().system('sw_vers')


# In[2]:


get_ipython().system('python -V')


# In[ ]:


SettingWithCopyWarning

columnsのget_locを利用してアクセすると良い

col_id_exit_date = exit_customer.columns.get_loc('exit_date')
col_id_end_date = exit_customer.columns.get_loc('end_date')


# SettingWithCopyWarningが出てしまう
# SettingWithCopyWarningが出てしまう
# 
# #### エラーが出ない
# ilocからのアクセスの方法で依存性がある
# 
# ```python
# print(type(exit_customer))
# exit_customer.iloc[i, col_id_exit_date] = exit_customer.iloc[i, col_id_end_date] - relativedelta(months=1)
# ```

# #### エラーが出る
# 
# ```python
# print(type(exit_customer['exit_date']))
# exit_customer['exit_date'].iloc[0] = exit_customer['end_date'].iloc[0] - relativedelta(months=1)
# ```

# In[ ]:


from dateutil.relativedelta import relativedelta

exit_customer = customer.loc[customer['is_deleted'] == 1].copy()

exit_customer['exit_date'] = None
exit_customer['end_date'] = pd.to_datetime(exit_customer['end_date'].copy())

for i in range(len(exit_customer)):
  col_id_exit_date = exit_customer.columns.get_loc('exit_date')
  col_id_end_date = exit_customer.columns.get_loc('end_date')
  # print(col_idx)
  # print(exit_customer.iloc[[i], [col_id_end_date]])
  # print(type(exit_customer.iloc[[i], [col_id_end_date]]))
  # print(exit_customer.iloc[i, col_id_end_date])
  # exit_customer.iloc[[i], [col_id_exit_date]] = exit_customer.iloc[[i], [col_id_end_date]] - relativedelta(months=1)
  exit_customer.iloc[i, col_id_exit_date] = exit_customer.iloc[i, col_id_end_date] - relativedelta(months=1)
  # print(type(exit_customer.iloc[[i], [col_id_end_date]]))
  # print(type(exit_customer.iloc[i, col_id_end_date]))
  
# for i in range(len(exit_customer)):
# print(exit_customer['exit_date'])
# print(relativedelta(months=1) + exit_customer['end_date'].iloc[0])
# print(exit_customer['exit_date'])
# print(type(exit_customer['exit_date']))
# exit_customer['exit_date'].loc[0] = 2
# # print(exit_customer['exit_date'].loc[0])
# print(exit_customer['exit_date'].iloc[0])
# print(exit_customer['end_date'].iloc[0])
# # exit_customer['end_date'].iloc[0] = 100
# print(exit_customer.columns)
# print(type(exit_customer))
# print()
# print()
# print()
# # exit_customer.iloc[[1, 2], ['exit_date']] = 40
# print(exit_customer.iloc[[1, 2], [0]])
# 
# 
# col_idx = exit_customer.columns.get_loc('exit_date')
# 
# print(col_idx)
# print(col_idx)
# print(col_idx)
# 
# exit_customer.iloc[[1, 2], [col_idx]] = 100

# exit_customer['exit_date'][0] = 1


# ## 参考記事
