#!/usr/bin/env python
# coding: utf-8

# ## Pythonからワードプレスのブログに投稿する
# 
# ワードプレスでコーポレイトサイトや個人のブログを運用していると、ブラウザ上からではなく、プログラム上から記事をアップロードしたり、アップデートしたりしたくなります。そのために、`python-wordpress-xmlrpc`をいうライブラリを利用します。
# 
# ### github
# - githubのjupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/article/library/python/006/006_nb.ipynb)
# 
# ### google colaboratory
# - google colaboratory で実行する場合は[こちら](https://colab.research.google.com/github/hiroshi0530/wa-src/blob/master/article/library/python/006/006_nb.ipynb)
# 
# ### 筆者の環境

# In[1]:


get_ipython().system('sw_vers')


# In[2]:


get_ipython().system('python -V')


# ### python-wordpress-xmlrpcのインストール

# In[ ]:


get_ipython().system('pip install python-wordpress-xmlrpc')


# ### 記事の投稿

# In[ ]:


from wordpress_xmlrpc import Client, WordPressPost
from wordpress_xmlrpc.methods.posts import NewPost

wpSiteXMLRPC = 'https://xxxxxx.com/xmlrpc.php'
loginId = 'xxxx'
password = 'yyyy'

wp = Client(wpSiteXMLRPC, loginId, password)
post = WordPressPost()

title = 'title'
body = 'body body '

post.title = title
post.content = body
 
post.terms_names = {
  'post_tag': ['tag'],
  'category': ['category']
}

# post.post_status = 'draft'
post.post_status = 'publish'

# set timezone to JST
post.date = datetime.datetime.now() - datetime.timedelta(hours=9)

# custom field
customFields = []
customFields.append({
  'key': 'aaa',
  'value': '***'
})

post.custom_fields = customFields

new_id = int(wp.call(NewPost(post)))
if new_id > 0:
  print('wordpress update success.')
else:
  print('wordpress update failure.')


# ### 画像のアップロード
# 
# 画像付きの記事をアップロードするときは、最初に記事単体をアップロードし、そのID（メディアID）を取得し、記事をアップロードする際、そのIDを指定することで実行します。画像をアップロードするサンプルコードは以下の通りです。

# In[ ]:


from wordpress_xmlrpc import Client, WordPressPost
from wordpress_xmlrpc.methods import media

wpSiteXMLRPC = 'https://xxxxxx.com/xmlrpc.php'
loginId = 'xxxx'
password = 'yyyy'

wp = Client(wpSiteXMLRPC, loginId, password)
post = WordPressPost()

def upload_image(in_image_file_name, out_image_file_name):
  if os.path.exists(in_image_file_name):
    with open(in_image_file_name, 'rb') as f:
      binary = f.read()

    data = {
      "name": out_image_file_name,
      "type": 'image/jpeg',
      "overwrite": True,
      "bits": binary
    }

    media_id = wp.call(media.UploadFile(data))['id']
    print(in_image_file_name.split('/')[-1], 'upload success')
    return
  else:
    print(in_image_file_name.split('/')[-1], 'NO IMAGE!!')


# ### サムネイルの指定
# 以下の様に、画像をアップロードし`post.thumbbail`とすることで、記事のサムネイルの指定をする事が出来ます。

# In[ ]:


post = WordPressPost()
media_id = wp.call(media.UploadFile(data))['id']
post.thumbnail = media_id


# 記事の投稿や、更新しがいにも記事の一覧を取得したり、固定ページのアップロード、アップデートが可能です。詳しくはpython-wordpress-xmlrpcの[サイト](https://python-wordpress-xmlrpc.readthedocs.io/en/latest/index.html)を確認してみてください。
