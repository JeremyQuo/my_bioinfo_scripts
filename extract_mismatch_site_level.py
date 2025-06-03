import pandas as pd
import numpy as np
import subprocess
import os
import argparse
import re
from multiprocessing import Pool
from functools import partial


def run_cmd(cmd):
    print(cmd)
    try:
        output = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, universal_newlines=True)
        print(output)
    except subprocess.CalledProcessError as e:
        print("Command execution failed:", e)
        raise RuntimeError('There are some errors in the cmd as below, please check your env\n' + cmd)

def count_mis_vectorized(df, feature_matrix):
    for item in feature_matrix:
        df[item] = df['Align string'].str.count(item)
    return df

def sum_plus_minus_numbers_vectorized(df):
    df['Ins'] = df['Align string'].apply(lambda x: sum(int(num) for num in re.findall(r'\+([0-9]+)', str(x))))
    df['Del'] = df['Align string'].apply(lambda x: sum(int(num) for num in re.findall(r'-([0-9]+)', str(x))))
    return df


def process_chunk(chunk, feature_matrix):
    # Ensure chunk has the correct columns
    if len(chunk.columns) < 6:
        print(f"Warning: Chunk has {len(chunk.columns)} columns, expected at least 6")
        return pd.DataFrame()  # Return empty DataFrame if invalid
    chunk.columns = ['Chrom', 'Position', 'Base', 'Coverage', 'Align string', 'Q string']
    chunk = count_mis_vectorized(chunk, feature_matrix)
    chunk = sum_plus_minus_numbers_vectorized(chunk)
    chunk['Mis'] = chunk['Coverage'] - chunk['\.'] - chunk[',']
    return chunk


def main(args):
    # Ensure result_path exists
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)

    # Check if temp.txt exists
    temp_file = os.path.join(args.result_path, 'temp.txt')
    if not os.path.exists(temp_file):
        cmds = (
            f"samtools mpileup {args.bam_file} "
            f"--no-output-ins --no-output-del -B -Q 1 -f {args.reference} "
            f"-o {temp_file}"
        )
        run_cmd(cmds)
    else:
        print(f"{temp_file} already exists, skipping samtools mpileup.")

    # Set feature_matrix
    feature_matrix = ['\.',",", 'A', 'T', 'C', 'G', 'a', 't', 'c', 'g']

    # Read and process in chunks
    chunksize = 10000
    num_cores = args.CPU
    print(f"Using {num_cores} CPU cores for parallel processing")

    # Check temp.txt structure
    with open(temp_file, 'r') as f:
        first_line = f.readline().strip().split('\t')
        if len(first_line) < 6:
            raise ValueError(f"temp.txt has {len(first_line)} columns, expected at least 6")

    chunks = pd.read_csv(temp_file, sep='\t', header=None, chunksize=chunksize)
    results = []

    with Pool(num_cores) as pool:
        process_partial = partial(process_chunk, feature_matrix=feature_matrix)
        for chunk in chunks:
            chunk = chunk[chunk[3] >= args.depth_limit]
            if not chunk.empty:
                results.append(pool.apply_async(process_partial, [chunk]))

        # Collect results
        results = [r.get() for r in results if r]

    if results:
        final_feature = pd.concat([r for r in results if not r.empty])
        final_feature.columns = ['Chrom', 'Position', 'Base', 'Coverage', 'Align string', 'Q string',
                                 'Match plus','Match minus', 'A', 'T', 'C', 'G', 'a', 't', 'c', 'g', 'Ins', 'Del', 'Mis']
        final_feature.to_csv(os.path.join(args.result_path, 'alignment_feature.csv'), index=False)
        print("Processing complete, output saved to alignment_feature.csv")
    else:
        print("No data after filtering, no output generated")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run samtools mpileup with specified inputs.")
    parser.add_argument("--bam_file", default='/t1/zhguo/Data/human/an_aligned.bam', help="Path to the input BAM file")
    parser.add_argument("--reference", default='/t1/zhguo/Data/human/gencode_transcript.fa',
                        help="Path to the reference FASTA file")
    parser.add_argument("--result_path", default='tmp_result/', help="Directory to store output temp.txt")
    parser.add_argument("--depth_limit", default=5, type=int, help="Minimum coverage depth")
    parser.add_argument("-t","--CPU", default=32, type=int, help="cpu number")
    args = parser.parse_args()

    if not os.path.exists(args.bam_file):
        raise FileNotFoundError(f"BAM file {args.bam_file} does not exist")
    if not os.path.exists(args.reference):
        raise FileNotFoundError(f"Reference file {args.reference} does not exist")

    main(args)
