#!/usr/bin/env bash
# -*- coding: utf-8 -*-
# get_Drosophila_melanogaster_genome.sh

: <<'COMMENT'
Script to download the reference genome of Drosophila melanogaster (dm6).
The genome is downloaded in 2bit format from the UCSC Genome Browser, which is a
compressed format that allows for efficient storage and retrieval of genomic sequences.
After downloading, the script uses the twoBitToFa tool to convert the 2bit file
into FASTA format, which is more commonly used for genomic analyses. Finally, the
original 2bit file is removed to save space.
COMMENT


mkdir -p ../../data/drosophila_genome
cd ../../data/drosophila_genome

# Reference genome of Drosophila melanogaster
wget https://hgdownload.soe.ucsc.edu/goldenPath/dm6/bigZips/dm6.2bit

# twoBitToFa for conversion 2bit na FASTA
wget https://hgdownload.soe.ucsc.edu/admin/exe/linux.x86_64/twoBitToFa
chmod +x twoBitToFa

# Conversotion to FASTA format
./twoBitToFa dm6.2bit dm6.fa

rm dm6.2bit
cd ../../src/sh
