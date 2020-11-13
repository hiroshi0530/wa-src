
# coding: utf-8

# ## opencvの使い方
# scikit-learnは機械学習、
# 
# 
# ### github
# - jupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/article/library/sklearn/datasets/ds_nb.ipynb)
# 
# ### google colaboratory
# - google colaboratory で実行する場合は[こちら](https://colab.research.google.com/github/hiroshi0530/wa-src/blob/master/article/library/sklearn/datasets/ds_nb.ipynb)
# 
# ### 環境
# 筆者のOSはmacOSです。LinuxやUnixのコマンドとはオプションが異なります。
# 
# ### 筆者の環境

# In[1]:


get_ipython().system('sw_vers')


# In[2]:


get_ipython().system('python -V')


# In[7]:


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

# In[14]:


img = cv2.imread(filename=filename)

plt.imshow(img)
plt.show()


# 画像の高さ、幅、カラーの場合の色の数（通常RGBの3）を確認します。

# In[10]:


if len(img.shape) == 3:
  img_height, img_width, img_channels = img.shape[:3]
else:
  img_height, img_width = img.shape[:2]
  
print('img_height :', img_height)
print('img_width :', img_width)
print('img_channels :', img_channels)


# ファイルの保存をします。

# In[15]:


out_filename = '../lena_out.jpg'
cv2.imwrite(out_filename, img)


# In[ ]:


def _set_gray_image():
  ._gray_image = cv2.cvtColor(._rgb_image, cv2.COLOR_BGR2GRAY)


def _set_inverse_gray_image():
  ._inverse_gray_image = cv2.bitwise_not(._gray_image)


def _set_binary_image(, threshold=_binary_image_threshold):
  ret, ._binary_image = cv2.threshold(._gray_image, threshold, 255, cv2.THRESH_BINARY)


def _set_inverse_binary_image():
  ._inverse_binary_image = cv2.bitwise_not(._binary_image)


# In[17]:


gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plt.imshow(gray_img)
plt.show()

out_filename = '../gray_out.jpg'
cv2.imwrite(out_filename, gray_img)

