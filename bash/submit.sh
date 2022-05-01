#!/bin/bash

mkdir -p job_err
mkdir -p job_out

sbatch --requeue -p sablab -t 8:00:00 --mem=16G --job-name=$1 -e ./job_err/$1-%j.err -o ./job_out/$1-%j.out $2
