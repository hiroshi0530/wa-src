#!/bin/bash

# tmuxのセッションで watch コマンドが生き残っている場合削除するようにする
pkill -kill -f copy_md_to_hugo
pkill -kill -f execute_copy

# 2sごとに実行
watch -n 2 ./copy_md_to_hugo.sh

