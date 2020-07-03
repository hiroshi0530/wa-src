
## git
ソースコードを管理するgitのコマンドです。GUIからでも便利ですが、コマンドを覚えるとさらに便利で、早いです。

gitについてはネット上で使い方がたくさん紹介されているので、ここでは自分が使うコマンドを中心に書いていこうと思います。

```bash
NAME
       git - the stupid content tracker

SYNOPSIS
       git [--version] [--help] [-C <path>] [-c <name>=<value>]
           [--exec-path[=<path>]] [--html-path] [--man-path] [--info-path]
           [-p|--paginate|-P|--no-pager] [--no-replace-objects] [--bare]
           [--git-dir=<path>] [--work-tree=<path>] [--namespace=<name>]
           [--super-prefix=<path>]
           <command> [<args>]
```

### github
- githubのjupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/article/library/bash/git/git_nb.ipynb)

### google colaboratory
- google colaboratory で実行する場合は[こちら](https://colab.research.google.com/github/hiroshi0530/wa-src/blob/master/article/library/bash/git/git_nb.ipynb)

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

よく利用しているコマンドの組み合わせと共に紹介します。

### [git add] ファイルの差分をインデックスに追加する


```python
!git add -h
```

    usage: git add [<options>] [--] <pathspec>...
    
        -n, --dry-run         dry run
        -v, --verbose         be verbose
    
        -i, --interactive     interactive picking
        -p, --patch           select hunks interactively
        -e, --edit            edit current diff and apply
        -f, --force           allow adding otherwise ignored files
        -u, --update          update tracked files
        --renormalize         renormalize EOL of tracked files (implies -u)
        -N, --intent-to-add   record only the fact that the path will be added later
        -A, --all             add changes from all tracked and untracked files
        --ignore-removal      ignore paths removed in the working tree (same as --no-all)
        --refresh             don't add, only refresh the index
        --ignore-errors       just skip files which cannot be added because of errors
        --ignore-missing      check if - even missing - files are ignored in dry run
        --chmod (+|-)x        override the executable bit of the listed files
    


### [git status] 現在の状態を確認する 

```bash
git status
```

## 備忘録
よく使うが、ついつい忘れてしまうコマンドです。コードレビューの時など良く利用します。

### ブランチ間のdiff
これでデフォルト設定しているvimdiffでdiffをチェックできます。.gitconfigでdiffのデフォルトをvimdiffに設定する必要があります。


```python
git difftool branchA branchB
```

### コミット間の差分（ファイル名のみ）
コミットIDを指定して、ファイルの差分をチェック


```python
git diff --stat shaA shaB
```

### コミット間の差分（各ファイル）
vimdiffで差分が見れるのは本当に便利


```python
git difftool -y shaA shaB -- <FILE NAME>
```
