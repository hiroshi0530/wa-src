
## expand
ファイルを読み込み、タブを半角スペースに変換します。

### github
- githubのjupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/article/library/bash/expand/expand_nb.ipynb)


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

```bash
expnad <option> filen_ame
```

## 代表的なオプション
- [-t N] : 変換するタブの数を指定します。 

### tオプション

タブを使ったファイルを作成します。


```bash
%%bash
echo -e "1\t2\t3" > temp.txt  # echo -e でタブをファイルに挿入します。
```


```bash
%%bash
cat -t temp.txt # cat -t でタブを可視化します。
```

    1^I2^I3



```bash
%%bash
expand -t 5 temp.txt
```

    1    2    3


## 参考記事

- [cat](/article/library/bash/cat/)
- [echo](/article/library/bash/echo/)
