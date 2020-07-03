
## uniq
重複している行を削除します。

```bash
UNIQ(1)                   BSD General Commands Manual                  UNIQ(1)

NAME
     uniq -- report or filter out repeated lines in a file

SYNOPSIS
     uniq [-c | -d | -u] [-i] [-f num] [-s chars] [input_file [output_file]]
```

### github
- githubのjupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/article/library/bash/uniq/uniq_nb.ipynb)

### google colaboratory
- google colaboratory で実行する場合は[こちら](https://colab.research.google.com/github/hiroshi0530/wa-src/blob/master/article/library/bash/uniq/uniq_nb.ipynb)

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

データ分析の仕事をしているとよく利用します。重複している無駄なものはゴミになる可能性が高いですから。通常は、

```bash
uniq <in file> <out file>
```

で重複している行を削除した結果を別ファイルに出力させます。


```bash
%%bash
echo "ファイルの準備"
echo -e "123\n123\n123" > temp
echo -e "<before>"
cat temp

uniq temp temp2
echo -e "\n<after>"
cat temp2
```

    ファイルの準備
    <before>
    123
    123
    123
    
    <after>
    123

