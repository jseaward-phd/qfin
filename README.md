# qfin
Implementation of a variational parametrised quantum circuit for time-series analysis from [Quantum Machine Learning in Finance: Time Series Forecasting](http://arxiv.org/abs/2202.00599).

It compares a Bidirectional Long Short-Term Memory (BiLSTM) model with a parameterized quantum circuit (PQC) in predicting the percent change in the price of financial instruments.

The BiLSTM model is implemented in tensorflow and the PQC is simulated using pennylane. A reqirements.txt file is provided which can be used by running
```
pip install -r requirements.txt
```
or the necessary requirements can be installed by running

```
pip install pandas pennylane sklearn tensorflow tqdm
```
for a more flexible installation.
