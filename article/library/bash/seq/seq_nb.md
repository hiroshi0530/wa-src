
## seq
連番を作成します。

```bash
NAME
     cat -- concatenate and print files

SYNOPSIS
     cat [-benstuv] [file ...]
```

### github
- githubのjupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/article/library/bash/seq/seq_nb.ipynb)

### 筆者の環境
筆者のOSはmacOSです。LinuxやUnixのコマンドとはオプションが異なります。

実際に動かす際は先頭の！や先頭行の%%bashは無視してください。


```python
!sw_vers

!man seq

```

    ProductName:	Mac OS X
    ProductVersion:	10.14.6
    BuildVersion:	18G2022
    
    SEQ(1)                    BSD General Commands Manual                   SEQ(1)
    
    NAME
         seq -- print sequences of numbers
    
    SYNOPSIS
         seq [-w] [-f format] [-s string] [-t string] [first [incr]] last
    
    DESCRIPTION
         The seq utility prints a sequence of numbers, one per line (default),
         from first (default 1), to near last as possible, in increments of incr
         (default 1).  When first is larger than last the default incr is -1.
    
         All numbers are interpreted as floating point.
    
         Normally integer values are printed as decimal integers.
    
         The seq utility accepts the following options:
    
         -f format     Use a printf(3) style format to print each number.  Only
                       the E, e, f, G, g, and % conversion characters are valid,
                       along with any optional flags and an optional numeric mini-
                       mum field width or precision.  The format can contain char-
                       acter escape sequences in backslash notation as defined in
                       ANSI X3.159-1989 (``ANSI C89'').  The default is %g.
    
         -s string     Use string to separate numbers.  The string can contain
                       character escape sequences in backslash notation as defined
                       in ANSI X3.159-1989 (``ANSI C89'').  The default is \n.
    
         -t string     Use string to terminate sequence of numbers.  The string
                       can contain character escape sequences in backslash nota-
                       tion as defined in ANSI X3.159-1989 (``ANSI C89'').  This
                       option is useful when the default separator does not con-
                       tain a \n.
    
         -w            Equalize the widths of all numbers by padding with zeros as
                       necessary.  This option has no effect with the -f option.
                       If any sequence numbers will be printed in exponential
                       notation, the default conversion is changed to %e.
    
         The seq utility exits 0 on success and non-zero if an error occurs.
    
    EXAMPLES
               # seq 1 3
               1
               2
               3
    
               # seq 3 1
               3
               2
               1
    
               # seq -w 0 .05 .1
               0.00
               0.05
               0.10
    
    SEE ALSO
         jot(1), printf(1), printf(3)
    
    HISTORY
         The seq command first appeared in Plan 9 from Bell Labs.  A seq command
         appeared in NetBSD 3.0, and ported to FreeBSD 9.0.  This command was
         based on the command of the same name in Plan 9 from Bell Labs and the
         GNU core utilities.  The GNU seq command first appeared in the 1.13 shell
         utilities release.
    
    BUGS
         The -w option does not handle the transition from pure floating point to
         exponent representation very well.  The seq command is not bug for bug
         compatible with the Plan 9 from Bell Labs or GNU versions of seq.
    
    BSD                            February 19, 2010                           BSD



```python
!bash --version
```

    GNU bash, version 3.2.57(1)-release (x86_64-apple-darwin18)
    Copyright (C) 2007 Free Software Foundation, Inc.


### 通常の使い方


```python
!for i in `seq 10`; do echo $i; done
```

    1
    2
    3
    4
    5
    6
    7
    8
    9
    10


seqを利用しなくても、{..}でいけます。0埋めなど知る必要がなければこれで十分です。


```python
!for i in {0..10}; do echo $i; done
```

    0
    1
    2
    3
    4
    5
    6
    7
    8
    9
    10


{..}は配列を作る演算子です。


```python
!echo {1..10}
```

    1 2 3 4 5 6 7 8 9 10


### 連番の0埋め


```python
!for i in `seq -w 3 10`; do echo $i; done
```

    03
    04
    05
    06
    07
    08
    09
    10


### 埋める0の量を変更 


```python
!for i in `seq -f %03g 3 10`; do echo $i; done
```

    003
    004
    005
    006
    007
    008
    009
    010

