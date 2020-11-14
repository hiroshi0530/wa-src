
# coding: utf-8

# ## opencvの使い方
# OpenCVは画像解析や機械学習によく利用される、コンピュータービジョンライブラリの一つです。基本的な画像変換だけでなく、画像のフィルター処理、顔認識、物体認識、オブジェクトトラッキングなど、実務でよく利用される機能が一通りそろっている非常に使いやすいライブラリになっています。実務で画像認識系の仕事をする際は必ず利用するライブラリになっています。
# 
# まずは基本的な使い方からです。
# 
# ### github
# - jupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/article/library/cv2/base/base_nb.ipynb)
# 
# ### google colaboratory
# - google colaboratory で実行する場合は[こちら](https://colab.research.google.com/github/hiroshi0530/wa-src/blob/master/article/library/cv2/base/base_nb.ipynb)
# 
# ### 環境
# 筆者のOSはmacOSです。LinuxやUnixのコマンドとはオプションが異なります。
# 
# ### 筆者の環境

# In[1]:


get_ipython().system('sw_vers')


# In[2]:


get_ipython().system('python -V')


# In[3]:


import cv2

print('opencv version :', cv2.__version__)


# 画像表示用にmatplotlibもimportします。画像はwebでの見栄えを考慮して、svgで保存する事とします。

# In[4]:


get_ipython().magic('matplotlib inline')
get_ipython().magic("config InlineBackend.figure_format = 'svg'")

import matplotlib.pyplot as plt


# 上の階層に lena.jpg というファイルがあるとします。

# In[5]:


get_ipython().run_cell_magic('bash', '', '\nls -al ../ | grep jpg')


# In[6]:


filename = '../lena.jpg'


# ## 画像の読み込み
# 
# 画像を読み込み、表示してみます。jupyter notebookの中で表示させるため、matplotlibを利用しています。

# In[7]:


img = cv2.imread(filename=filename)

# OpenCVではGBRの準備で画像が読み込まれるが、JupyterNotebookではRGBで表示させる
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(rgb_img)
plt.show()


# ### 画像情報の取得
# 
# 画像の高さ、幅、カラーの場合の色の数（通常RGBの3）を確認します。

# In[8]:


def get_image_info(img):
  if len(img.shape) == 3:
    img_height, img_width, img_channels = img.shape[:3]
    print('img_channels :', img_channels)
  else:
    img_height, img_width = img.shape[:2]
    
  print('img_height :', img_height)
  print('img_width :', img_width)
  
get_image_info(img=img)


# ## 画像の保存
# 
# imwriteメソッドを利用します。

# In[9]:


out_filename = '../lena_out.jpg'
cv2.imwrite(out_filename, img)


# ## グレースケール化
# 
# cv2.COLOR_BGR2GRAYを利用します。

# In[10]:


gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

get_image_info(img=gray_img)

plt.imshow(gray_img)
plt.gray()
plt.show()

out_filename = '../gray_out.jpg'
cv2.imwrite(out_filename, gray_img)


# ## 白黒反転
# 
# グレースケールの画像を白黒反転させます。

# In[11]:


bitwise_gray_img = cv2.bitwise_not(gray_img)

get_image_info(img=bitwise_gray_img)

plt.imshow(bitwise_gray_img)
plt.gray()
plt.show()

out_filename = '../bitwise_out.jpg'
cv2.imwrite(out_filename, bitwise_gray_img)


# ## バイナリ化
# 
# グレースケールの画像から二値画像に変換します。

# In[12]:


threshold = 120
ret, binary_img = cv2.threshold(gray_img, threshold, 255, cv2.THRESH_BINARY)

get_image_info(img=binary_img)

plt.imshow(binary_img)
plt.gray()
plt.show()

out_filename = '../binary_out.jpg'
cv2.imwrite(out_filename, binary_img)

