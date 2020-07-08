
## cat
ファイルを連結します。ファイルの中身を表示するのによく使用します。
ヒアドキュメントなど、複数行にわたるファイルを作成するのに利用します。

```bash
NAME
     cat -- concatenate and print files

SYNOPSIS
     cat [-benstuv] [file ...]
```

### github
- githubのjupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/article/library/bash/cat/cat_nb.ipynb)

### google colaboratory
- google colaboratory で実行する場合は[こちら](https://colab.research.google.com/github/hiroshi0530/wa-src/blob/master/article/library/bash/cat/cat_nb.ipynb)

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

### ファイルの表示、連結

```bash
cat file
cat file1 file2
cat file1 file2 > file3
```

ファイルを作成し、その中身を表示します。


```bash
%%bash
echo "a b c" > temp.txt 
cat temp.txt
```

    a b c


ファイルを二つ作成し連結します。


```bash
%%bash
echo "e f g" > temp1.txt
echo "h i j" > temp2.txt
cat temp1.txt temp2.txt > temp3.txt
```

temp3.txtが作成され、その中でtemp1.txtとtemp2.txtが連結されていることがわかります。


```bash
%%bash 
cat temp3.txt
```

    e f g
    h i j


### ヒアドキュメントの作成

スクリプトの中で複数行にわたるファイルを作成する際によく利用します。
EOFの表記は何でも良いです。ファイルを作成する際にはリダイレクトさせます。


```bash
%%bash

cat << EOF > temp10.txt
a b c
e f g
h i j
EOF
```


```python
cat temp10.txt
```

    a b c
    e f g
    h i j


ただ、これだとコマンドをそのままを入れ込むことが出来ません。コマンドの結果や、変数などが展開されて表記されます。


```bash
%%bash

cat << EOF > temp11.sh
#!/bin/bash

user="test"

echo ${user}

EOF
```
ここであえて変数を展開させたくない場合や、コマンドそのものの表記を残したい場合は、EOFをシングルクオテーションマークで囲みます。

```bash
%%bash
cat << 'EOF' > temp12.sh
#!/bin/bash

user="test"

echo ${user}

EOF

cat temp12.sh
```

    #!/bin/bash
    
    user="test"
    
    echo ${user}
    


となり、ちゃんとファイルの中に`${user}`が展開されずにファイルの中に記載されていることがわかります。

### 代表的なオプション
- t : タブを明示的に表示します(^Iと表示されます)
- e : 改行コードを明示的に表示します（＄と表示されます）


```bash
%%bash
echo -e "a\tb\tc" > temp2.txt 
cat -t temp2.txt
```

    a^Ib^Ic



```bash
%%bash
echo -e "a\tb\tc" > temp3.txt 
cat -e temp3.txt
```

    a	b	c$

