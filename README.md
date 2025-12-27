# Bandit Algorithms in Julia
Multi-armed bandits capture the essence of tradeoffs between exploration and exploitation while making decision in unknown environments. Some of the popular bandit algorithms have been coded in Julia language

Currently implemented algorithms:

- Thompson Sampling
- $\epsilon-$ Thompson Sampling

Code optimization:
- Used pre-allocated arrays to reduce memory allocation and improve performance.
- Used loops instead of slicing
- Type specification for lists

TODO:
- $\epsilon-$ Greedy
- Upper Confidence Bound (UCB)
- Bayesian UCB
- Information Directed Sampling (IDS)
- Bayesian Information Directed Sampling (BIDS)