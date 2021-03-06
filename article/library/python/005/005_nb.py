
# coding: utf-8

# ## Python Tips
# 
# pythonを利用する上で、便利な表記などの個人的なメモです。基本的な部分は触れていません。対象も自分が便利だなと思ったものに限定されます。
# 
# 自動botなどを作って、自動的にツイートする場合に利用します。
# 
# ### github
# - githubのjupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/article/library/python/005/005_nb.ipynb)
# 
# ### google colaboratory
# - google colaboratory で実行する場合は[こちら](https://colab.research.google.com/github/hiroshi0530/wa-src/blob/master/article/library/python/005/005_nb.ipynb)
# 
# ### 筆者の環境

# In[1]:


get_ipython().system('sw_vers')


# In[2]:


get_ipython().system('python -V')


# ## tweepyのインストール
# 
# TwitterのAPIを利用するには、tweepyというモジュールをインストールする必要があります。

# In[ ]:


get_ipython().system('pip install tweepy')


# ## コード例
# 
# 以下のコードはあるプロジェクトで実際にあるトリガーが引かれたときに実行される自動ツイート用のコードです。
# configファイルとして`./config/login_info.json`が必要です。これらの情報はTwitterの開発者アカウントを作り、取得する必要があります。
# 

# In[ ]:


import datetime
import sys

import tweepy
import json

import os
import time

import json
import shutil

def tweet(text, image_file_path):

  json_file = "./conf/login_info.json"
  with open(json_file) as file:
    payload = json.loads(file.read())

  consumer_key  = payload["Consumer_key"]
  consumer_secret = payload["Consumer_secret"]
  access_token  = payload["Access_token"]
  access_secret   = payload["Access_secret"]

  auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
  auth.set_access_token(access_token, access_secret)

  api = tweepy.API(auth, wait_on_rate_limit = True)

  #画像付きツイート
  # api.update_with_media(status = text, filename = image_file_path)
    
  # 通常のテキストツイート
  api.update_status(text)


if __name__ == '__main__':

  args = sys.argv
  
  today = datetime.date.today()
  yyyy = today.strftime('%Y')
  mm = today.strftime('%m')
  dd = today.strftime('%d')
  yyyymmdd = today.strftime('%Y%m%d')
  
  text = """
【{0}年 {1}月{2}日】

XXXXXXXXXXXXXX

""".format(yyyy, mm, dd)

  tweet(text=text, image_file_path="")


# ### configファイル
# 
# configファイルは以下の様に作っています。

# In[ ]:


{
    "COMMENT": "login_info.json",
    "Consumer_key": "",
    "Consumer_secret": "",
    "Access_token": "",
    "Access_secret": ""
}


# これらのファイルを作ることで、あるイベントトリガーで自動的にツイートする事が可能です。
# 私は、これで大体一日に10個程度自動ツイートしています。
