#!/usr/bin/env bash
# -*- coding: utf-8 -*-
# get_original_DeepSTARR.sh

: <<'COMMENT'
Script to download the data from the original DeepSTARR paper.
The data includes DNA sequences of genomic regions and their activity
in developmental and housekeeping conditions. The data is downloaded as
separate FASTA files for the sequences and text files for the activity information
for the training, validation, and test sets. Each file is downloaded directly from
the Zenodo repository where the original DeepSTARR data is hosted.
COMMENT


mkdir -p ../../data/deepSTARR

# FASTA files with DNA sequences of genomic regions from train/val/test sets
wget https://zenodo.org/record/5502060/files/Sequences_Train.fa?download=1 -O ../../data/deepSTARR/Sequences_Train.fa
wget https://zenodo.org/record/5502060/files/Sequences_Val.fa?download=1 -O ../../data/deepSTARR/Sequences_Val.fa
wget https://zenodo.org/record/5502060/files/Sequences_Test.fa?download=1 -O ../../data/deepSTARR/Sequences_Test.fa

# Files with developmental and housekeeping activity of genomic regions from train/val/test sets
wget https://zenodo.org/record/5502060/files/Sequences_activity_Train.txt?download=1 -O ../../data/deepSTARR/Sequences_activity_Train.txt
wget https://zenodo.org/record/5502060/files/Sequences_activity_Val.txt?download=1 -O ../../data/deepSTARR/Sequences_activity_Val.txt
wget https://zenodo.org/record/5502060/files/Sequences_activity_Test.txt?download=1 -O ../../data/deepSTARR/Sequences_activity_Test.txt
