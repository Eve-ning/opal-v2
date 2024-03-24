import streamlit as st

st.title("Information")
st.header("How it works ELI5")
st.warning(
    "We say distance, but it's actually displacement, as it can be negative."
)
st.write(
    r"""
As a simple example, lets put 2 maps and a player on a line. 
**By placing any non-numerical object on a line, we call this operation
an embedding.** 

- The **further right** a map is, the **harder** it is, 
- The **further right** a player is, the **better** they are.
```
. ┌────┐  ┌────┐  ┌────┐
──┤ M1 ├──┤ M2 ├──┤ P1 ├──►
  └────┘  └────┘  └────┘
     ◄────── ──────►
 Lower Skill Higher Skill
```

We assume that the distance between a map and player is related to accuracy
in some manner.
```
. P1 - M1 = Higher Acc
   ◄──────────────────►
  ┌────┐  ┌────┐  ┌────┐
──┤ M1 ├──┤ M2 ├──┤ P1 ├──►
  └────┘  └────┘  └────┘
           ◄──────────►
        P1 - M2 = Lower Acc
```

> Larger distances indicates larger skill gap, therefore, higher accuracies.   

However, the distance can't be interpreted directly as accuracy.
Therefore, we train a model to learn the relationship between
distance and accuracy, with the constraint that larger distances should yield
larger accuracies.

As a result, we train a model that
- learns the **best location** to place these maps and players
- figures out **the relationship** between the distance and accuracy

### A 2nd Dimension

In practice, maps and players are much more complex.
As an example, they can be LN or RC-difficulty biased: 
better at one type of difficulty than the other.

With reference to the above example, we can create a scenario where the
distance between $M_1$ and $P_1$, and between $M_2$ and $P_1$ are the same
despite $M_2 > M_1$ on the RC-axis

```
.            ▲ LN
             │
  ┌────┐     │    ┌────┐
  │ M1 ├─────┼────┤ P1 │
  └────┘     │    └────┘
             │     /
             │    /              RC
─────────────┼───/────────────────►
             │  /
             │ /
             │/
          ┌──┴─┐       
          │ M2 │       
          └────┘       
```


As shown, because $M_2$ is LN-easy, while $M_1$ is LN-hard, it compensates
the difference in RC difficulty, resulting in the same distance, 
thus same accuracy.

This argument, *unfortunately*, can be extended to infinitely more dimensions.

However, with more dimensions, it's harder to understand what exactly each
dimension represents.
It's up to the creator to decide if they want an explainable model, 
or an accurate model.

### Difficulties with separating RC and LN

In practice, it's hard to separate RC and LN, as they're often intertwined,
with no unbiased metrics to separate them.

The closest metric is possibly
the ratio of RC to LN notes, but it often runs into edge cases where maps
are predominantly LNs because the mapper uses mini-LNs, which should lean
towards RC difficulty. As mini-LNs are denser than 
LNs in actual LN maps, ratio-based metrics biases the resulting model.

## How it works ELI(a Data Scientist)

Before we start, we define some terms mathematically.

- The one-hot encoded player and map vectors are $P$ and $M$ respectively.
- We have $n_P$ players and $n_M$ maps.

$$
\begin{align*}
P\in\mathbb{R}^{n_P}, P_i\in\{0,1\} \text{for all } i\in\{1,2,...,n_P\}\\
M\in\mathbb{R}^{n_M}, M_i\in\{0,1\} \text{for all } i\in\{1,2,...,n_M\}
\end{align*}
$$

As shown, the above 2 vectors cannot be directly compared, as they are of
different dimensions. Therefore, we need to map them to some common space of
dimension $n_D$.

E.g. If we only had the dimensions $RC$ and $LN$, we would have
$n_D=2$.

$$
\begin{align*}
E_P&\in\mathbb{R}^{n_P\times n_D}\\
E_M&\in\mathbb{R}^{n_M\times n_D}\\
\therefore PE_P=Q\in\mathbb{R}^{n_D}&, ME_M=N\in\mathbb{R}^{n_D}\\
\end{align*}
$$

As a quick illustration, the embedding simply just maps the one-hot encoded
vector to another space.
```
.                  ┌─────┬─────┬─────┐
  ┌───────────────►│ E00 │ ... │ E0D │
  │                ├─────┼─────┼─────┤
  │                │     │     │     │
  │                │  .  │ .   │  .  │
  │     ┌─────────►│  .  │  .  │  .  │
  │     │          │  .  │   . │  .  │
  │     │          │     │     │     │
  │     │          ├─────┼─────┼─────┤
  │     │    ┌────►│ EN0 │ ... │ END │
  │     │    │     └──┬──┴──┬──┴──┬──┘
  │     │    │        │     │     │
  ▼     ▼    ▼        ▼     ▼     ▼
┌────┬─────┬────┐  ┌─────┬─────┬─────┐
│ P0 │ ... │ PN ├─►│ Q0  │ ... │ QD  │
└────┴─────┴────┘  └─────┴─────┴─────┘
```
Now that we have a common space, we can compare the vectors.

We'll denote this difference as $Q - N = \Delta \in\mathbb{R}^{n_D}$.
> We use Q and N as they are the next alphabet after P and M, respectively.

### Mapping $\Delta$ to Accuracy

There are 2 constraints we have:

1) The transform is non-linear.
2) The transform must be monotonically increasing. Which means, if we increase
   the distance, the accuracy must increase.
 
In order to achieve this, we can use a neural network. However, to satisfy
constraint 2, we need to force all weights to be positive, to guarantee a
monotonically increasing function.

To do so, we use the **softplus** weights during a fully connected layer.
This forces the resulting weights to be positive,
while still allowing the model to learn a non-positive weight.

The architecture is simple, but it works:

$$
\Delta
\rightarrow \text{PositiveLinear(D, N)}
\rightarrow \text{PositiveLinear(N, 1)}
\rightarrow \text{Acc}
$$

```
.                 ┌─────┬─────┬─────┐
  ┌──────────────►│ W00 │ ... │ W0N │
  │               ├─────┼─────┼─────┤
  │               │  .  │ .   │  .  │
  │     ┌────────►│  .  │  .  │  .  │
  │     │         │  .  │   . │  .  │
  │     │         ├─────┼─────┼─────┤
  │     │    ┌───►│ WD0 │ ... │ WDN │
  │     │    │    └──┬──┴──┬──┴──┬──┘
  │     │    │       │     │     │
  │     │    │       ▼     ▼     ▼
  │     │    │    ┌─────┬─────┬─────┐
  │     │    │    │ B0  │ ... │ BN  │
  │     │    │    └──┬──┴──┬──┴──┬──┘
  │     │    │       f     f     f
  │     │    │       ▼     ▼     ▼
┌─┴──┬──┴──┬─┴──┐ ┌─────┬─────┬─────┐ ┌─────┐
│ D0 │ ... │ DD │ │ X0  │ ... │ XN  │ │ ACC │
└────┴─────┴────┘ └──┬──┴──┬──┴──┬──┘ └─────┘
                     │     │     │       ▲
                     │     │     │       s
                     │     │     │    ┌──┴──┐
                     │     │     └───►│ W00 │
                     │     │          ├─────┤
                     │     │          │  .  │
                     │     └─────────►│  .  │
                     │                │  .  │
                     │                ├─────┤
                     └───────────────►│ WD0 │
                                      └─────┘
```

Where $f$ is the softplus activation function.

We'll denote this transform as $A(\Delta)=\text{Acc}$.

## Uncertainty

Reference: [Deep Ensembles](https://arxiv.org/abs/1612.01474)

> We didn't use ensembles yet, as we're in early access, it's just 1 model.

The method of estimating uncertainty is through predicting 2 metrics of a
distribution, then minimizing its Negative Log Likelihood Loss (NLLLoss).
We use a Laplace distribution, over the original Normal Distribution as
we will explain later.

The Laplace distribution is defined as:

$$
\text{Laplace}(x|\mu,b)=\frac{1}{2b}\exp\left(-\frac{|x-\mu|}{b}\right)
$$

Where $\mu$ is the mean, and $b$ is the scale parameter.
Therefore, we just need to predict $\mu$ and $b$.

Notice that because had a positive constraint on the weights, the model
is forced to learn a monotonic increasing function between $\Delta$ and
the $\mu$. However, larger $\Delta$ does not necessarily mean larger $b$.
The simple solution is to create a separate branch for $b$ which you can see
in the `delta_model.py` code.

We can be more liberal with the $b$ transformation as we don't have any 
specific constraints on it.

```python
self.delta_to_acc_mean = nn.Sequential(
    PositiveLinear(n_emb, n_delta_mean_neurons),
    nn.Tanh(),
    PositiveLinear(n_delta_mean_neurons, 1),
)
self.delta_to_acc_var = nn.Sequential(
    nn.Linear(n_emb, n_delta_var_neurons),
    nn.ReLU(),
    nn.Linear(n_delta_var_neurons, n_delta_var_neurons),
    nn.ReLU(),
    nn.Linear(n_delta_var_neurons, 1),
)
```

### Laplace Distribution over Normal Distribution

The Laplace distribution is used over the Normal distribution as it has
heavier tails, which means it's kinder to outliers. This is important as
we observe that many scores are outliers, especially for gimmick maps and
niche players.

Mathematically speaking, because the Laplace distribution has a heavier tail,
the NLLLoss is smaller for outliers, which means the model is less likely to
shift drastically because a score is unexpectedly high or low. 

# Appendix

## LN Dimension Weighting 

> This attempt didn't work as explained in
> *Difficulties with separating RC and LN*.

Fortunately, each map embeds information on its LN density. 
That means we can assume that for a LN-heavy map, the LN dimension contributes
more to the accuracy than the RC dimension, and vice versa.

We denote this ratio as a proportion $\rho_{LN}$ and $\rho_{RC}$, such that

$$\rho_{LN},\rho_{RC}\in[0,1]:\rho_{LN}+\rho_{RC}=1$$

This means that if we have $\Delta_{RC}$ and $\Delta_{LN}$, then we have 
$\text{Acc}_{RC}$ and $\text{Acc}_{LN}$. As both $\text{Acc}\in[0,1]$, 
the final accuracy is simply the weighted sum of the 2 accuracies.

$$
\begin{align*}
\text{Acc}&=\rho_{RC}\text{Acc}_{RC}+\rho_{LN}\text{Acc}_{LN}\\
&=\sum_T^{RC,LN} \rho_T\text{Acc}_T
\end{align*}
$$

By doing so, we too train model alignment to the expected
LN, RC embeddings, in turn, improves interpretability.



"""
)
