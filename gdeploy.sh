#!/bin/bash -l

#タイミングによって 余計なファイルが入ってしまうので、一度コピージョブをkill
pkill -kill -f copy_md_to_hugo
pkill -kill -f execute_copy

cd ~/wa/ && git add . && git commit -m "[auto] update"
cd ~/wa/ && git push origin master

cd ~/wa_src/ && git add . && git commit -m "[auto] update"
cd ~/wa_src/ && git push origin master

cd ~/wa/static/ && source .envrc && bash deploy.sh
