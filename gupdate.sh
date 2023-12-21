#!/bin/bash -l

WA="/Users/hiroshi/private"

# タイミングによって 余計なファイルが入ってしまうので、一度コピージョブをkill
pkill -kill -f copy_md_to_hugo
pkill -kill -f execute_copy

# 別途手動でコミットしてしまうと、pushしないので分ける
cd $WA/wa/ && git add . && git commit -m "[auto] update"
cd $WA/wa/ && git push origin master

cd $WA/wa_src/ && git add . && git commit -m "[auto] update"
cd $WA/wa_src/ && git push origin master
