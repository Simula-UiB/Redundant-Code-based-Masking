<h1 align="center">Redundant Code-based Masking Revisited: sources and experimental data</h1>

<p align="center">
    <a href="https://github.com/Simula-UiB/Redundant-Code-based-Masking/blob/master/AUTHORS"><img src="https://img.shields.io/badge/authors-SimulaUIB-orange.svg"></a>
    <a href="https://github.com/Simula-UiB/Redundant-Code-based-Masking/blob/master/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg"></a>
</p>

This repository contains the source code used to run experiments for the paper `Redundant Code-based Masking Revisited` scheduled to appear in [TCHES](https://tches.iacr.org/) 2021, issue 1. All the data generated and used to draw the plots figuring in the paper is also provided along with the seeds used for randomness generation to allow researchers to reproduce and study our results.

## License

This repository is licensed under the MIT License.

* MIT license ([LICENSE](LICENSE) or http://opensource.org/licenses/MIT)


## Overview

3 python files contain the code for the attacks. A simplified CLI allows for specifying any relevant parameter for the experiment that is to be conducted (noise, public points, number of shares, amount of traces/experiment, output, seed). Some example of the CLI are given as comments on top of each file. They all require the packages [numpy](https://numpy.org/) and [scipy](https://www.scipy.org/) to run. All the python script are monothreaded, since every attack is independant the simplest way to run large experiments is to start several of them with the same parameters and combine the resulting output.
In the [data](data) folder are the experiments used in the paper to make all our plots.

## Experimental protocol

We call an experiment the combination of several independent attacks which use the same amount of traces and the same parameters. One experiment gives us the data for approximating the number of traces necessary to reach a 90% success rate of attacks with those parameters. For each attack, we randomly generate a secret which will stay fixed. Then we make traces by generating fresh randomness used by the masking scheme and computing our leakage. For each trace, we get the Hamming-weights of the shares of the first round AES SBOX output with the addition of Gaussian noise. We then use a Maximum Likelihood Distinguisher, as explained in the paper. We output the result of those experiments in a csv format.

## Data format and reproducing experiments

Every file in [data](data) is labelled with the format d_n_with_a_b_..._e.csv except for the attacks on first and second order Boolean Masking. Where `d` is the degree of the masking, `n` is the amount of shares and `a,b,...,e` are the public points chosen for the Polynomial masking. Inside each file, you will find experiments for a given value of sigma2. One experiment always starts with those three rows:
```
sigma_2;X
number of traces;Y
seed;Z
```
`X` will give you the variance of noise (sigma2) used for this experiment, `Y` the total amount of traces used for every single attack and `Z` the seed used for the experiment. All our seed were generated from `/dev/urandom`.
After this header follow all the attacks (in our case always 1000) where one row gives the result for one attack. Each attack is seeded with the initial seed + its index. Meaning that if you want to look at the 400th attack of an experiment, you can run directly with seed + 400 as the starting seed. The result of an attack is an array of 0 and 1 where 1 indicate that the correct key was ranked first at this point and 0 indicated the opposite. If the amount of traces per attack is > 100 all our results are downsampled to a 100 points, which means that the 0 and 1 of an attack are for every 1% of the total amount of traces used for the attack.
/!\ If an attack has ranked the correct key first for 15 consecutive data points we assume that it will continue to do so and stop the attack by fixing all remaining points to 1. We empirically checked that even a lower threshold (10) does not influence the results, but we take 15 for security measure. This is to speed up the computation.

## Why downsampling?

There are several reasons which explain why our data is downsampled and outputted with this type of format, and they are mostly the byproduct of a previous approach that we ended up removing from the paper. We initially wanted to approximate the success rate as a function of the amount of traces by a probability distribution. For that, we needed the raw bins of data to fit the distribution and we needed to store them for post-processing. The fact that our code is mono threaded also called for storing the results of each experiment. For these reasons, we decided to downsample to a 100 data points per attack regardless of the amount of traces. We do not believe it impacted our results since, while it does mean we lose in precision, the overall precision is already bounded by the fact that we only use a 1000 attacks per experiment. Furthermore, our results do tolerate a generous loss of precision since the masking schemes we compare are orders of magnitude away in term of amount of traces necessary to break them. Even 10% of error would be unnoticeable for the differences that the paper consider.
As said previously, we ended up discarding that approach (initially we used a Rayleigh distribution as an approximation) and went for a classical success rate instead. New experiments could definitely be ran without downsampling and the success rate computed online instead of requiring post-processing of the data. We just arrived at that conclusion too late in the reviewing process to rerun the experiments.

## Usage of generated data

To compute the success rate as a function of the number of traces, all the attacks of an experiment are summed and divided by the amount of attacks ran. Then each resulting number gives the success rate for each % of the total amount of traces (again if there are more than 100 traces used). This is an approximation of the function we are looking for and for which we look for the 90% cutoff. Since the resulting function can be jittery and non-monotonous, a signal processing filter can be used as a low pass filter. For the plots in the paper, we used a Savitsky-Golay filter.