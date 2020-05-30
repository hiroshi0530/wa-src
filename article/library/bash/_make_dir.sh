#!/bin/bash

# 001 seq
# 002 expand
# 003 cut
# 004 paste
# 005 split
# 006 cat
# 007 ls
# 008 sort
# 009 tar
# 010 sed
# 011 xarg
# 012 git
# 013 aws
# 014 wget
# 015 head
# 016 tail
# 017 uniq
# 018 sort
# 019 echo
# 020

# for i in `seq -f %03g 1 $1`; do
#   if [ ! -d $i ]; then mkdir $i; fi
#   if [ ! -f $i/${i}_nb.ipynb ]; then cp template.ipynb $i/${i}_nb.ipynb; fi
# done

array=()
array+=("seq") 
array+=("expand") 
array+=("cat") 
array+=("echo") 
array+=("paste") 
array+=("cut") 
array+=("paste") 
array+=("split") 
array+=("sort") 
array+=("unexpand") 
array+=("ls") 
array+=("tr") 

for i in ${array[@]}; do
  if [ ! -d $i ]; then mkdir $i; fi
  if [ ! -f $i/${i}_nb.ipynb ]; then cp ./_template/template.ipynb $i/${i}_nb.ipynb; fi
done
