#!/bin/bash

array=()

# array+=("awk") 
array+=("cat") 
array+=("cut") 
array+=("echo") 
array+=("expand") 
array+=("head") 
array+=("ls") 
array+=("paste") 
array+=("sed") 
array+=("seq") 
array+=("sort") 
array+=("split") 
array+=("tail") 
array+=("tar") 
array+=("tr") 
array+=("unexpand") 
array+=("uniq") 
array+=("wget") 
array+=("xarg") 


for i in ${array[@]}; do
  if [ ! -d $i ]; then mkdir $i; fi
  if [ ! -f $i/${i}_nb.ipynb ]; then cp ./_template/template.ipynb $i/${i}_nb.ipynb; fi
done
