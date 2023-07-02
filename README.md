# An informal implemantation of MoveSim
Codes for paper in KDD'20 (AI for COVID-19): Learning to Simulate Human Mobility

## Modification

This code is based on the implementation of the authors (https://github.com/FIBLAB/MoveSim), but there are some modifications.

- Remove p_loss and d_loss because our using data does not satisfy the assumption of them.
- Replace the two self-attention network of the generator into one self-attention network because the network does not work well in our dataset.


## Usage

To run this code using the test data

`./run.sh`

## Citation

`Feng, Jie, et al. "Learning to simulate human mobility." Proceedings of the 26th ACM SIGKDD international conference on knowledge discovery & data mining. 2020.`