# -*- coding: utf-8 -*-

########################################################################################################
# eval.py
#
# Description: this is an evaluation script to measure the quality of spaces restoration in text
# Usage: python eval.py golden.txt predicted.txt 
# Outputs F1-score, precision and recall metrics

# Project: IHS Markit Internship Test
# Creation date: January 12, 2018
# Copyright (c) 2017 by IHS Markit
########################################################################################################

from __future__ import print_function
import re
import sys
import codecs

if len(sys.argv) != 3:
	print("Two arguments expected: GOLDEN_FILE and RESULT_FILE" , file=sys.stderr)
	exit(1)
	
def unspace(l):
	return re.sub(' ','',l)
	
def trim_line(l):
	l = re.sub('[ \t]+',' ',l)
	l = l.strip(' ')
	return l
	
# returns the quantity of non-space characters and list space_arr[] with non-space character indices, followed by space
def line2space_arr(line):
	space_arr = []
	size = 0
	
	for c in line:
		if c == " ":
			space_arr.append(size)
		else:
			size +=1
	print(line)
	print(space_arr)
	return size, space_arr


file1 = sys.argv[1]
file2 = sys.argv[2]

file1_content = codecs.open(file1, 'r+', encoding='utf-8').readlines()
file2_content = codecs.open(file2, 'r+', encoding='utf-8').readlines()

#check that file1 and file2 has equal number of lines
if len(file1_content) < len(file2_content):
	print("File %s has fewer lines than %s." % (file1, file2), file=sys.stderr)
	exit(1)

true_positive = 0
false_negative = 0
false_positive = 0

for idx, pair in enumerate(zip(file1_content, file2_content)):
	line1 = trim_line(pair[0].rstrip())
	line2 = trim_line(pair[1].rstrip())

	size1, space_arr1 = line2space_arr(line1)
	size2, space_arr2 = line2space_arr(line2)

	#if size1 != size2 or unspace(line1) != unspace(line2):
	#	print("Files are not aligned at line %i" % (idx+1), file=sys.stderr)
	#	exit(1)

	for s in space_arr1:
		if s in space_arr2:
			true_positive +=1
		else:
			false_negative +=1

	for s in space_arr2:
		if s not in space_arr1: 
			false_positive +=1
	
if true_positive + false_positive > 0 and true_positive + false_negative > 0:
	precision = true_positive / float( true_positive + false_positive )
	recall = true_positive / float( true_positive + false_negative )
	f1 = 2 * precision * recall / float( precision + recall )
	
	print("F1:\t%s\nPrecision:\t%s\nRecall:\t%s\n\nTrue_positive:\t%d\nFalse_positive:\t%d\nFalse_negative:\t%d\n" % (round(f1,5), round(precision,5), round(recall,5), true_positive+0, false_positive+0, false_negative+0))
