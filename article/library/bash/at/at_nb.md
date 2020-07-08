
## at 
時間を指定してジョブを実行する

```bash
AT(1)                     BSD General Commands Manual  

NAME
     at, batch, atq, atrm -- queue, examine, or delete jobs for later execution
```

### github
- githubのjupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/article/library/bash/at/at_nb.ipynb)

### google colaboratory
- google colaboratory で実行する場合は[こちら](https://colab.research.google.com/github/hiroshi0530/wa-src/blob/master/article/library/bash/at/at_nb.ipynb)

### 環境
筆者のOSはmacOSです。LinuxやUnixのコマンドとはオプションが異なります。

実際に動かす際は先頭の！や先頭行の%%bashは無視してください。


```python
!sw_vers
```

    ProductName:	Mac OS X
    ProductVersion:	10.14.6
    BuildVersion:	18G95



```python
!bash --version
```

    GNU bash, version 3.2.57(1)-release (x86_64-apple-darwin18)
    Copyright (C) 2007 Free Software Foundation, Inc.


## 使用例
通常私が利用するときは、fオプションとtオプションを利用します。

以下の様なbashファイルを用意します。


```bash
%%bash

cat << 'EOF' > temp.sh
#!/bin/bash

echo `date +%Y%m%d`

EOF

chmod +x temp.sh
```


```python
!ls -al
```

    total 32
    drwxr-xr-x   7 hiroshi  staff   224  7  8 17:26 [34m.[m[m
    drwxr-xr-x  25 hiroshi  staff   800  7  8 17:13 [34m..[m[m
    drwxr-xr-x   3 hiroshi  staff    96  7  8 17:14 [34m.ipynb_checkpoints[m[m
    -rw-r--r--   1 hiroshi  staff  4094  7  8 17:26 at_nb.ipynb
    -rw-r--r--   1 hiroshi  staff  2005  7  8 17:26 at_nb.md
    -rw-r--r--   1 hiroshi  staff  1469  7  8 17:26 at_nb.py
    -rwxr-xr-x   1 hiroshi  staff    34  7  8 17:26 [31mtemp.sh[m[m



```python
!./temp.sh
```

    20200708


このファイルを2020年7月8日18:00に実行させるには次のようなコマンドを実行します。タイマー的な使い方が出来るので、一時的に使いたいのであれば、CRONを設定するより簡単です。


```bash
%%bash
at -f temp.sh.sh -t 202007081800
```

## 代表的なオプション
- f : ファイルを指定します
- t : 時間のフォーマットを定義します (YYYYmmddHHMM)
