#!/usr/bin/env python
# coding: utf-8

# ## [第3章 正規表現](https://nlp100.github.io/ja/ch03.html)
# 結果だけ載せました。正解かどうかは保障しません笑
# 
# ### github
# - githubのjupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa/blob/master/src/ml/nlp100/02/02_nb.ipynb)
# 
# ### 筆者の環境

# In[1]:


get_ipython().system('sw_vers')


# In[2]:


get_ipython().system('python -V')


# In[3]:


get_ipython().system('bash --version')


# ### ダウンロードと解凍

# In[4]:


get_ipython().system('wget https://nlp100.github.io/data/jawiki-country.json.gz -O ./wiki.json.gz')


# In[5]:


get_ipython().system('gzip -f -d wiki.json.gz')


# ## 解答

# ### 共通部分

# In[6]:


import re

file_name = './wiki.json'


# ### 20問 JSONデータの読み込み

# In[34]:


import json

with open(file_name, mode='r') as f:
  for line in f.readlines():
    info = json.loads(line)
    if info['title'] == 'イギリス':
      print('### 先頭500文字 ###')
      print(info['text'][:4000])
      print()
      print('### 最後500文字 ###')
      print(info['text'][-500:])
      
      # 以後 b_infoをイギリスの情報として利用
      b_info = info['text']


# ### 21問 カテゴリ名を含む行を抽出

# In[9]:


for line in b_info.split('\n'):
  s = re.findall('\[\[Category:(.*)\]\]',line)
  if len(s) != 0:
    print(line)


# ### 22問 カテゴリ名の抽出

# In[42]:


s = re.findall('\[\[Category:(.*?)\]\]',b_info.replace('\n',''))
if len(s) != 0:
  for _ in s:
    print(_)


# ### 23問 セクション構造

# In[53]:


s = re.findall('(==+)(.*?[^=])==+',b_info.replace('\n',''))
if len(s) != 0:
  for _ in s:
    print(len(_[0]) - 1, _[1])


# ### 24問 ファイル参照の抽出
# ファイルの定義がわからないのですが、以下を抜き出すものとします。[こちら](https://ja.wikipedia.org/wiki/Help:%E3%83%95%E3%82%A1%E3%82%A4%E3%83%AB%E3%83%9A%E3%83%BC%E3%82%B8)を参考にしています。
# 
# ```text
# [[ファイル:ファイル名]]
# [[:ファイル:ファイル名]]
# [[File:ファイル名]]
# [[Media:ファイル名]] 
# ```

# In[54]:


s1 = re.findall(r'\[\[File:(.+?)\]\]',b_info.replace('\n',''))
s2 = re.findall(r'\[\[Media:(.+?)\]\]',b_info.replace('\n',''))
s3 = re.findall(r'\[\[:?ファイル:(.+?)\]\]',b_info.replace('\n',''))

for s in [s1,s2,s3]:
  if len(s) != 0:
    for _ in s:
      print(_.split('|')[0])


# ### 25問 テンプレートの抽出
# 
# テンプレートが何を意味するか不明ですが、wikiの[ヘルプページ](https://ja.wikipedia.org/wiki/Help:%E3%83%86%E3%83%B3%E3%83%97%E3%83%AC%E3%83%BC%E3%83%88)を参考に
# 
# ```text
# {{テンプレート名}}
# 
# {{テンプレート名|引数1|引数2|.....}}
# ```

# In[66]:


# print(b_info)
s = re.findall(r'\{\{基礎情報(.+)\}\}',b_info.replace('\n',''))

s1 = re.findall(r'\{\{(.+?)\}\}',s[0].replace('\n',''))
print(len(s1))
print(s1)


# ### 26問

# ### 27問

# In[24]:



a = 'a,b,c'
a.split(',')
print(a)
print(a)
print(a)


# ### 28問

# In[39]:


a = '{{red{{ir}}ect|UK}}'

a = """
{{基礎情報 国
|略名  =イギリス
|日本語国名 = グレートブリテン及び北アイルランド連合王国
|公式国名 = {{lang|en|United Kingdom of Great Britain and Northern Ireland}}<ref>英語以外での正式国名:<br />
*{{lang|gd|An Rìoghachd Aonaichte na Breatainn Mhòr agus Eirinn mu Thuath}}（[[スコットランド・ゲール語]]）
*{{lang|cy|Teyrnas Gyfunol Prydain Fawr a Gogledd Iwerddon}}（[[ウェールズ語]]）
*{{lang|ga|Ríocht Aontaithe na Breataine Móire agus Tuaisceart na hÉireann}}（[[アイルランド語]]）
*{{lang|kw|An Rywvaneth Unys a Vreten Veur hag Iwerdhon Glédh}}（[[コーンウォール語]]）
*{{lang|sco|Unitit Kinrick o Great Breetain an Northren Ireland}}（[[スコットランド語]]）
**{{lang|sco|Claught Kängrick o Docht Brätain an Norlin Airlann}}、{{lang|sco|Unitet Kängdom o Great Brittain an Norlin Airlann}}（アルスター・スコットランド語）</ref>
|国旗画像 = Flag of the United Kingdom.svg
|国章画像 = [[ファイル:Royal Coat of Arms of the United Kingdom.svg|85px|イギリスの国章]]
|国章リンク =（[[イギリスの国章|国章]]）
|標語 = {{lang|fr|[[Dieu et mon droit]]}}<br />（[[フランス語]]:[[Dieu et mon droit|神と我が権利]]）
|国歌 = [[女王陛下万歳|{{lang|en|God Save the Queen}}]]{{en icon}}<br />''神よ女王を護り賜え''<br />{{center|[[ファイル:United States Navy Band - God Save the Queen.ogg]]}}
|地図画像 = Europe-UK.svg
|位置画像 = United Kingdom (+overseas territories) in the World (+Antarctica claims).svg
|公用語 = [[英語]]
|首都 = [[ロンドン]]（事実上）
|最大都市 = ロンドン
|元首等肩書 = [[イギリスの君主|女王]]
|元首等氏名 = [[エリザベス2世]]
|首相等肩書 = [[イギリスの首相|首相]]
|首相等氏名 = [[ボリス・ジョンソン]]
|他元首等肩書1 = [[貴族院 (イギリス)|貴族院議長]]
|他元首等氏名1 = [[:en:Norman Fowler, Baron Fowler|ノーマン・ファウラー]]
|他元首等肩書2 = [[庶民院 (イギリス)|庶民院議長]]
|他元首等氏名2 = {{仮リンク|リンゼイ・ホイル|en|Lindsay Hoyle}}
|他元首等肩書3 = [[連合王国最高裁判所|最高裁判所長官]]
|他元首等氏名3 = [[:en:Brenda Hale, Baroness Hale of Richmond|ブレンダ・ヘイル]]
|面積順位 = 76
|面積大きさ = 1 E11
|面積値 = 244,820
|水面積率 = 1.3%
|人口統計年 = 2018
|人口順位 = 22
|人口大きさ = 1 E7
|人口値 = 6643万5600<ref>{{Cite web|url=https://www.ons.gov.uk/peoplepopulationandcommunity/populationandmigration/populationestimates|title=Population estimates - Office for National Statistics|accessdate=2019-06-26|date=2019-06-26}}</ref>
|人口密度値 = 271
|GDP統計年元 = 2012
|GDP値元 = 1兆5478億<ref name="imf-statistics-gdp">[http://www.imf.org/external/pubs/ft/weo/2012/02/weodata/weorept.aspx?pr.x=70&pr.y=13&sy=2010&ey=2012&scsm=1&ssd=1&sort=country&ds=.&br=1&c=112&s=NGDP%2CNGDPD%2CPPPGDP%2CPPPPC&grp=0&a=IMF>Data and Statistics>World Economic Outlook Databases>By Countrise>United Kingdom]</ref>
|GDP統計年MER = 2012
|GDP順位MER = 6
|GDP値MER = 2兆4337億<ref name="imf-statistics-gdp" />
|GDP統計年 = 2012
|GDP順位 = 6
|GDP値 = 2兆3162億<ref name="imf-statistics-gdp" />
|GDP/人 = 36,727<ref name="imf-statistics-gdp" />
|建国形態 = 建国
|確立形態1 = [[イングランド王国]]／[[スコットランド王国]]<br />（両国とも[[合同法 (1707年)|1707年合同法]]まで）
|確立年月日1 = 927年／843年
|確立形態2 = [[グレートブリテン王国]]成立<br />（1707年合同法）
|確立年月日2 = 1707年{{0}}5月{{0}}1日
|確立形態3 = [[グレートブリテン及びアイルランド連合王国]]成立<br />（[[合同法 (1800年)|1800年合同法]]）
|確立年月日3 = 1801年{{0}}1月{{0}}1日
|確立形態4 = 現在の国号「'''グレートブリテン及び北アイルランド連合王国'''」に変更
|確立年月日4 = 1927年{{0}}4月12日
|通貨 = [[スターリング・ポンド|UKポンド]] (£)
|通貨コード = GBP
|時間帯 = ±0
|夏時間 = +1
|ISO 3166-1 = GB / GBR
|ccTLD = [[.uk]] / [[.gb]]<ref>使用は.ukに比べ圧倒的少数。</ref>
|国際電話番号 = 44
|注記 = <references/>
}}
"""

s = re.findall(r'\{\{(.*)\}\}',a.replace('\n',''))
print(a)
for i in s:
  print(i)


# ### 29問

# In[ ]:





# In[ ]:





# ## 関連記事
# - [第1章 準備運動](/ml/nlp100/01/)
# - [第2章 UNIXコマンド](/ml/nlp100/02/)
# - [第3章 正規表現](/ml/nlp100/03/)
# - [第4章 形態素解析](/ml/nlp100/04/)
# - [第5章 係り受け解析](/ml/nlp100/05/)
# - [第6章 機械学習](/ml/nlp100/06/)
# - [第7章 単語ベクトル](/ml/nlp100/07/)
# - [第8章 ニューラルネット](/ml/nlp100/08/)
# - [第9章 RNN,CNN](/ml/nlp100/09/)
# - [第10章 機械翻訳](/ml/nlp100/10/)