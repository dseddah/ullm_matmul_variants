#!/bin/sh


#for ULLM in `ls ./out/ullm.elf.*`; do 
for SUF in  C neon neon2 neon_max mt_dispatch accelerate; do 
	ULLM=./out/ullm.elf.$SUF
	echo "running $ULLM"
	$ULLM -c out/stories15M.bin -t out/llama2.c/tokenizer.bin -p "The quick brown fox jumped. Where did he go?" 2>&1 | grep "token/s"
done


