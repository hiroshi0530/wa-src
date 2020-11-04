#!/bin/bash

for i in `seq -f %03g 8 15`; do
  if [ ! -d $i ]; then mkdir $i; fi
  if [ ! -f $i/${i}_nb.ipynb ]; then cp template.ipynb $i/${i}_nb.ipynb; fi
done
