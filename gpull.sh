#!/bin/bash -l

#タイミングによって 余計なファイルが入ってしまうので、一度コピージョブをkill
pkill -kill -f copy_md_to_hugo
pkill -kill -f execute_copy

cd ~/wa/ && git pull origin master
cd ~/wa_src/ && git pull origin master
