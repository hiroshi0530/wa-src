#!/bin/bash -l

cd ~/wa/ && git add . && git commit -m "[auto] update" && git push origin master
cd ~/wa_src/ && git add . && git commit -m "[auto] update" && git push origin master
cd ~/wa/static/ && source .envrc && bash deploy.sh
