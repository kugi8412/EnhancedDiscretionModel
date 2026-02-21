#!/usr/bin/env bash
# -*- coding: utf-8 -*-
# get_DREAMChallenge.sh

: <<'COMMENT'
Script to download the data from the DREAM challenge.
The data includes DNA sequences, their activity, and fold information.
The data is downloaded as a tar.gz file, which is then extracted into a
directory named 'dream_challenge'. After extraction, the original tar.gz file
is removed to save space.
COMMENT


mkdir -p ../../data/dream_challenge

# Get data from DREAM challenge (Sequence + Activity + Folds)
wget https://zenodo.org/records/10633252/files/drosophila_starrseq_data.tar.gz?download=1 -O ../../data/dream_challenge/drosophila_starrseq_data.tar.gz

cd ../../data/dream_challenge
tar -xzf drosophila_starrseq_data.tar.gz
rm drosophila_starrseq_data.tar.gz
cd ../../src/sh
