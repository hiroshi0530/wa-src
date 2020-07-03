
## ls
ファイルやディレクトリを表示します。
以下の様にオプションがたくさんあるので、覚えるのは無理です。
自分がよく使うものを使えこなせれば良いかと思います。

```bash
NAME
     ls -- list directory contents

SYNOPSIS
     ls [-ABCFGHLOPRSTUW@abcdefghiklmnopqrstuwx1] [file ...
```

### github
- githubのjupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/article/library/bash/ls/ls_nb.ipynb)

### google colaboratory
- google colaboratory で実行する場合は[こちら](https://colab.research.google.com/github/hiroshi0530/wa-src/blob/master/article/library/bash/ls/ls_nb.ipynb)

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
オプションなしです。直下のディレクトリにあるファイルを表示します。


```bash
%%bash
ls
```

    ls_nb.ipynb
    ls_nb.md
    ls_nb.py


最もよく使うのが`ls -al`です。エイリアス設定をしています。リスト形式で、ドットから始まる隠しファイルも表示してくれます。


```python
ls -al
```

    total 24
    drwxr-xr-x   6 hiroshi  staff   192  7  3 18:50 [34m.[m[m/
    drwxr-xr-x  24 hiroshi  staff   768  7  3 18:36 [34m..[m[m/
    drwxr-xr-x   3 hiroshi  staff    96  6 24 19:46 [34m.ipynb_checkpoints[m[m/
    -rw-r--r--   1 hiroshi  staff  2967  7  3 18:49 ls_nb.ipynb
    -rw-r--r--   1 hiroshi  staff  1052  7  3 18:49 ls_nb.md
    -rw-r--r--   1 hiroshi  staff   970  7  3 18:49 ls_nb.py


## 代表的なオプション
- a = all
- l == list
