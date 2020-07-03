
## sed
文字列を置換します。
とても便利なコマンドで、様々な所で利用します。
文字列の置換だけでなく、新たな行を追加したり、削除したり出来ます。

```bash
NAME
     sed -- stream editor

SYNOPSIS
     sed [-Ealn] command [file ...]
     sed [-Ealn] [-e command] [-f command_file] [-i extension] [file ...]
```

### github
- githubのjupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/article/library/bash/sed/sed_nb.ipynb)

### google colaboratory
- google colaboratory で実行する場合は[こちら](https://colab.research.google.com/github/hiroshi0530/wa-src/blob/master/article/library/bash/sed/sed_nb.ipynb)

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
基本的には

```bash
sed s/before/after/g <FILE>
```

でファイル内のbeforeという文字列をafterに置換します。

```bash
sed -ei s/before/after/g <FILE>
```
というeiオプションでファイルを上書きします。MACでなければiオプションで上書きできます。最後のgはファイル内のすべてbeforeに対して置換を行います。なければ最初に一致する最初のbeforeだけ置換します。


```bash
%%bash
echo "ファイルの準備をします。"
echo "=== example : before ===" > temp
cat temp
sed -ie "s/before/after/g" temp
cat temp
```

    ファイルの準備をします。
    === example : before ===
    === example : after ===


## 代表的なオプション
- e
- i
