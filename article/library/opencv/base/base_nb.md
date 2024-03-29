
## opencvの使い方
OpenCVは画像解析や機械学習によく利用される、コンピュータービジョンライブラリの一つです。基本的な画像変換だけでなく、画像のフィルター処理、顔認識、物体認識、オブジェクトトラッキングなど、実務でよく利用される機能が一通りそろっている非常に使いやすいライブラリになっています。実務で画像認識系の仕事をする際は必ず利用するライブラリになっています。

まずは基本的な使い方からです。

### github
- jupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/article/library/cv2/base/base_nb.ipynb)

### google colaboratory
- google colaboratory で実行する場合は[こちら](https://colab.research.google.com/github/hiroshi0530/wa-src/blob/master/article/library/cv2/base/base_nb.ipynb)

### 環境
筆者のOSはmacOSです。LinuxやUnixのコマンドとはオプションが異なります。

### 筆者の環境


```python
!sw_vers
```

    ProductName:	Mac OS X
    ProductVersion:	10.14.6
    BuildVersion:	18G95



```python
!python -V
```

    Python 3.5.5 :: Anaconda, Inc.



```python
import cv2

print('opencv version :', cv2.__version__)
```

    opencv version : 3.4.1


画像表示用にmatplotlibもimportします。画像はwebでの見栄えを考慮して、svgで保存する事とします。


```python
%matplotlib inline
%config InlineBackend.figure_format = 'svg'

import matplotlib.pyplot as plt
```

上の階層に lena.jpg というファイルがあるとします。


```bash
%%bash

ls -al ../ | grep jpg
```

    -rw-r--r--@  1 hiroshi  staff  8211 11 14 22:01 lena.jpg



```python
filename = '../lena.jpg'
```

## 画像の読み込み

画像を読み込み、表示してみます。jupyter notebookの中で表示させるため、matplotlibを利用しています。


```python
img = cv2.imread(filename=filename)

# OpenCVではGBRの準備で画像が読み込まれるが、JupyterNotebookではRGBで表示させる
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(rgb_img)
plt.show()
```


![svg](base_nb_files/base_nb_10_0.svg)


### 画像情報の取得

画像の高さ、幅、カラーの場合の色の数（通常RGBの3）を確認します。


```python
def get_image_info(img):
  if len(img.shape) == 3:
    img_height, img_width, img_channels = img.shape[:3]
    print('img_channels :', img_channels)
  else:
    img_height, img_width = img.shape[:2]
    
  print('img_height :', img_height)
  print('img_width :', img_width)
  
get_image_info(img=img)
```

    img_channels : 3
    img_height : 225
    img_width : 225


## 画像の保存

imwriteメソッドを利用します。


```python
out_filename = '../lena_out.jpg'
cv2.imwrite(out_filename, img)
```




    True



## グレースケール化

cv2.COLOR_BGR2GRAYを利用します。


```python
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

get_image_info(img=gray_img)

plt.imshow(gray_img)
plt.gray()
plt.show()

out_filename = '../gray_out.jpg'
cv2.imwrite(out_filename, gray_img)
```

    img_height : 225
    img_width : 225



![svg](base_nb_files/base_nb_16_1.svg)





    True



## 白黒反転

グレースケールの画像を白黒反転させます。


```python
bitwise_gray_img = cv2.bitwise_not(gray_img)

get_image_info(img=bitwise_gray_img)

plt.imshow(bitwise_gray_img)
plt.gray()
plt.show()

out_filename = '../bitwise_out.jpg'
cv2.imwrite(out_filename, bitwise_gray_img)
```

    img_height : 225
    img_width : 225



![svg](base_nb_files/base_nb_18_1.svg)





    True



## バイナリ化

グレースケールの画像から二値画像に変換します。


```python
threshold = 120
ret, binary_img = cv2.threshold(gray_img, threshold, 255, cv2.THRESH_BINARY)

get_image_info(img=binary_img)

plt.imshow(binary_img)
plt.gray()
plt.show()

out_filename = '../binary_out.jpg'
cv2.imwrite(out_filename, binary_img)
```

    img_height : 225
    img_width : 225



![svg](base_nb_files/base_nb_20_1.svg)





    True


