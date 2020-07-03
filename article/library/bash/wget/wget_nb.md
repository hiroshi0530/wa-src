
## wget
指定されたURLからファイルをダウンロードします。

```bash
WGET(1)                            GNU Wget          

NAME
       Wget  The noninteractive network downloader.
```

### github
- githubのjupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/article/library/bash/wget/wget_nb.ipynb)

### google colaboratory
- google colaboratory で実行する場合は[こちら](https://colab.research.google.com/github/hiroshi0530/wa-src/blob/master/article/library/bash/wget/wget_nb.ipynb)

### 環境
筆者のOSはmacOSです。LinuxやUnixのコマンドとはオプションが異なります。

実際に動かす際は先頭の！や先頭行の%%bashは無視してください。


```python
!sw_vers
```

    ProductName:	Mac OS X
    ProductVersion:	10.14.6
    BuildVersion:	18G2022



```python
!bash --version
```

    GNU bash, version 3.2.57(1)-release (x86_64-apple-darwin18)
    Copyright (C) 2007 Free Software Foundation, Inc.


## 使用例
普段は -O オプションを利用してファイルをローカルに保存します。


```bash
%%bash
wget https://ja.wayama.io/index.html -O test.html
```

    --2020-07-03 20:23:00--  https://ja.wayama.io/index.html
    ja.wayama.io (ja.wayama.io) をDNSに問いあわせています... 99.84.55.105, 99.84.55.81, 99.84.55.46, ...
    ja.wayama.io (ja.wayama.io)|99.84.55.105|:443 に接続しています... 接続しました。
    HTTP による接続要求を送信しました、応答を待っています... 200 OK
    長さ: 9402 (9.2K) [text/html]
    `test.html' に保存中
    
         0K .........                                             100% 54.0M=0s
    
    2020-07-03 20:23:00 (54.0 MB/s) - `test.html' へ保存完了 [9402/9402]
    


ファイルがあるかどうか確認します。ちゃんとダウンロードされています。


```python
!ls | grep test
```

    test.html


## 代表的なオプション
オプションは色々ありますが、とりあえず-Oだけ覚えています。
- O : ファイル名を指定します。
