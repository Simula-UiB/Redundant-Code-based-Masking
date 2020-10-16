import numpy as np
import scipy.stats
import math
import csv
import sys
import os
from numpy import random as rd

# CLI: SIGMA2 N_TRACES N_EXPERIMENT (optional) OUTPUT_PATH (optional) seed
# ex: 0.2 400 1000 results/output.csv -> 0.2 sigma over 400 traces repeated for 
# 1000 experiments and output in results/output.csv

# value of noise for which we want to run experiments
SIGMA2 = float(sys.argv[1])
# number of trace to process for each value of noise
N_TRACES = int(sys.argv[2])
# number of experiment to perform for the success rate
N_EXPERIMENTS = int(sys.argv[3])

OUTPUT_PATH = "second_order_boolean.csv"

if len(sys.argv) >= 5:
    var_path = sys.argv[4]
    if var_path != "default":
        OUTPUT_PATH = var_path

if len(sys.argv) == 6:
    seed = int(sys.argv[5])
else:
    seed = int.from_bytes(os.urandom(4), sys.byteorder)

# amount of data point taken by experiment, 100 by default
# if the amount of traces is below 100 we take a point every traces
# meaning your amount of traces has to be either a multiple of 100 or < 100
data_step = 100
if N_TRACES < 100:
    data_step = N_TRACES
assert(N_TRACES % data_step == 0)

# Value of the HW for all value between 0 and 256

HAMMING_WEIGTH_TABLE = [0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4,
                        2, 3, 3, 4, 3, 4, 4, 5, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2,
                        3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3,
                        3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3,
                        4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6,
                        5, 6, 6, 7, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3,
                        4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5,
                        5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 2, 3, 3, 4, 3, 4, 4,
                        5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
                        3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 4, 5, 5, 6, 5, 6, 6, 7, 5,
                        6, 6, 7, 6, 7, 7, 8]

# AES SBOX table

SBOX_TABLE = [
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5,	0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0,	0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc,	0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a,	0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0,	0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b,	0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85,	0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5,	0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17,	0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88,	0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c,	0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9,	0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6,	0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e,	0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94,	0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68,	0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
]

# Compute the Hamming-Weigths of the shares made from 2 random mask and a secret value

def make_shares(sbox_out, mask_1, mask_2):
    mask_3 = sbox_out ^ mask_1 ^ mask_2
    return [HAMMING_WEIGTH_TABLE[mask_3], HAMMING_WEIGTH_TABLE[mask_1], HAMMING_WEIGTH_TABLE[mask_2]]

################
#              #
# MAIN PROGRAM #
#              #
################

# precomputation
# 2 tables are built, mean_values_amount, mean_values_flat

# mean_value_amount is the number of time each tuple of HW appears
# for all the possible randomness

# mean_values_flat is a map from randomness to the corresponding noiseless
# leakage (the HW of the shares) to save time

# Finally all pdfs used in processing traces are generated in a "frozen"
# array. Which will help when actually computing the scores

mean_values = [[[make_shares(sbox_out, mask_1, mask_2) for mask_2 in range(
    256)] for mask_1 in range(256)] for sbox_out in range(256)]

mean_values_amount = [
    [0 for mean_012 in range(729)] for sbox_out in range(256)]
mean_values_flat = []

for sbox_out in range(256):
    for mask_1 in range(256):
        for mask_2 in range(256):
            means = mean_values[sbox_out][mask_1][mask_2]
            for mean in means:
                mean_values_flat.append(mean)
            mean_values_amount[sbox_out][81 *
                                         means[0] + 9*means[1] + means[2]] += 1
mean_values_amount = np.array(mean_values_amount)
mean_values_flat = np.array(mean_values_flat)
del mean_values


# value used to determine when to stop an experiment in advance if the secret
# is ranked at the top consecutively N time (where N is corner_cut)
corner_cut = 100
if N_TRACES >= 1000:
    corner_cut = 15

# Research assumption: all shares have the same sigma^2
cov_matrix = cov_matrix = [[0 for x in range(3)]
                           for y in range(3)]
for i in range(3):
    cov_matrix[i][i] = SIGMA2

pdfs = [scipy.stats.multivariate_normal(
                mean=i, cov=SIGMA2) for i in range(0,9)]


# Processing

experiment_results = []
for experiment in range(N_EXPERIMENTS):
    # the PRNG is reseeded with seed + experiment to allow for
    # easy inspection of individual experiment
    rd_state = rd.default_rng(seed+experiment)

    # distinguishing scores for each possible value of the secret
    scores = np.zeros(256)

    # results of the experiment, a 1 will indicate that at step
    # n (where a step is N_TRACES / data_step) the correct secret has
    # the highest distinguishing score
    secret_ranked_top = [1 for i in range(data_step)]

    # fixed secret for the experiment
    secret = rd_state.integers(0, 256)

    successful_last_attempt = 0
    for t in range(N_TRACES):
        plaintext = rd_state.integers(0, 256)
        mask_1 = rd_state.integers(0, 256)
        mask_2 = rd_state.integers(0, 256)
        idx = (SBOX_TABLE[secret ^ plaintext]*65536 + mask_1*256 + mask_2)*3

        # One trace generated randomly from a multivariate normal distribution
        # with means = noiseless leakage (precomputed in mean_values_flat) +
        # noise with a sigma^2 similar for all shares
        trace = rd_state.multivariate_normal(
            mean_values_flat[idx:idx+3], cov_matrix, 1)[0]

        # evaluate each pdf(trace) once, compute every product for the amount of shares
        # to get the value for every tuple of HW, and then multiply it with
        # the amount of occurence of the tuple of HW stored in mean_values_amount
        evaluations = [1]
        for share in trace:
            tmp = [pdf.pdf(share) for pdf in pdfs]
            buff = []
            for x in evaluations:
                for y in tmp:
                    buff.append(x*y)
            evaluations = buff
        summed_scores = np.sum(
            mean_values_amount * evaluations, axis=1)

        # now we have the scores for each possible value of SBOX[secret^plaintext]
        # to be able to sum the scores from one experiment to the next we need to
        # map those score to the actual secret, so we reindex the scores
        indexes = np.array([SBOX_TABLE[s ^ plaintext]
                            for s in range(256)])
        summed_scores = summed_scores[indexes]

        # summing the log distinguisher score
        scores += np.log(summed_scores)

        # every step we take a peak at the scores
        # if the secret is not ranked at the top we replace
        # the 1 by a 0 for this step
        # if the secret has been ranked at the top for corner_cut
        # consecutive time then we shortcircuit the experiment and
        # consider that every data point after will be a 1
        rate_idx = (t+1) % (N_TRACES/data_step)
        if rate_idx == 0:
            if np.argmax(scores) == secret:
                if successful_last_attempt == corner_cut:
                    break
                else:
                    successful_last_attempt += 1
            else:
                successful_last_attempt = 0
                secret_ranked_top[(t) // (N_TRACES//data_step)] = 0

    experiment_results.append(secret_ranked_top)


# output
f = open(OUTPUT_PATH, 'a')
with f:
    writer = csv.writer(f, delimiter=';')
    writer.writerow(["sigma_2", SIGMA2])
    writer.writerow(["number of traces", N_TRACES])
    writer.writerow(["seed", seed])
    for row in experiment_results:
        writer.writerow(row)
