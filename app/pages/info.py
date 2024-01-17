import streamlit as st

st.title("Information")
st.header("How it works ELI5")
st.write(
    r"""
    
st.warning(
    "We say distance, but it's actually displacement, as it can be negative."
)
As a simple example, lets put 2 maps and a player on a line.

The further right a map is, the harder it is, 
and the further right a player is, the better they are.
```
. ┌────┐  ┌────┐  ┌────┐
──┤ M1 ├──┤ M2 ├──┤ P1 ├──►
  └────┘  └────┘  └────┘
    ◄────── ──────►
Lower Skill Higher Skill
```

Then the distance between a map and player is related to accuracy.
```
. P1 - M1 = Higher Acc
   ◄──────────────────►
  ┌────┐  ┌────┐  ┌────┐
──┤ M1 ├──┤ M2 ├──┤ P1 ├──►
  └────┘  └────┘  └────┘
           ◄──────────►
        P1 - M2 = Lower Acc
```
I believe it's intuitive that larger distances indicates larger skill gap,
therefore, higher accuracies.   

By using a ML model to fit this distance to an accuracy,
we get a model that learns the best location to place these maps and players
such that the constraint is best satisfied.

### A 2nd Dimension

However, maps and players are much more complex.
They can be LN or RC-difficulty biased: 
better at one type of difficulty than the other.

In fact, we can create a scenario where the distance between $M_1$ and $P_1$,
and the distance between $M_2$ and $P_1$ are the same.

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

### Mapping $\Delta$ to Accuracy

There are 2 constraints we have:

1) The transform is non-linear.
2) The transform must be monotonically increasing. Which means, if we increase
   the distance, the accuracy must increase.
 
In order to achieve this, we can use a neural network. However, to satisfy
constraint 2, we need to force all weights to be positive, to guarantee a
monotonically increasing function.

To do so, we use the exponential of
the weights when we apply the linear transformation, this allows the weights
to be positive, while still allowing the model to learn non-linearity

The architecture is simple, but it works:

$$
\Delta
\rightarrow \text{ExpLinear(D, N)}
\rightarrow \text{ReLU}
\rightarrow \text{ExpLinear(N, 1)}
\rightarrow \text{Sigmoid}
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

We'll denote this transform as $A(\Delta)=\text{Acc}$.

### LN Dimension Weighting 

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

### Interpreting Map and Player Embeddings

Unfortunately, the Map and Player embeddings aren't easily interpretable, 
this is because we only made use of their difference, but never their absolute
locations.

For example, if we forced $Q - N = 1$, $Q$ and $N$ can be any number, as long
as their difference is 1. However, if we'd try to fix some values, not only 
would it be complex, it'll interfere with the model's ability to learn.
Therefore, the approach, shouldn't be to force the embeddings to be
interpretable, but to use the embeddings, then preprocess it to make sense.

A simple solution is to concatenate the embeddings, however, it can be a bit 
odd that for a map and player with the same embeddings, which yields 
$\Delta=0$, the accuracy is some arbitrary value. Thus, it's good to set some
constraints on the embeddings, such that $\Delta=0$ yields $\text{Acc}=0.95$, 
which is a common metric to show that the player is able to play the map well.

We want to solve the following problem:

$$
Acc = 0.95 = \sum_T^{RC,LN} A_T(Q_T-N_T + \text{Bias}_T)
$$

This 

Let's say we want to show $Q, N$ (the embeddings), in the range $\in[0,1]$,
while maintaining some sort of meaning with their differences. Before that, we
review 

$$
\begin{align*}
\text{Acc}&=\rho A(Q-N)\\
\frac{\text{Acc}}{\rho}&=A(Q-N)\\
A^{-1}\left(\frac{\text{Acc}}{\rho}\right)&=Q-N\\
N + A^{-1}\left(\frac{\text{Acc}}{\rho}\right)&=Q\\
A^{-1}\left(\frac{\text{Acc}}{\rho}\right) + Q&=N\\
\end{align*} 
$$

With this formula, given a map, and an accuracy, we can find the player 
embedding that is expected to achieve that accuracy.

Furthermore, we can introduce an inequality constraint, such that we find all
players that are expected to achieve an accuracy greater than some threshold.

$$
\begin{align*}
\text{Threshold}&\leq\rho A(Q-N)\\
\frac{\text{Threshold}}{\rho}&\leq A(Q-N)\\
A^{-1}\left(\frac{\text{Threshold}}{\rho}\right)&\leq Q-N\\
N + A^{-1}\left(\frac{\text{Threshold}}{\rho}\right)&\leq Q\\
A^{-1}\left(\frac{\text{Threshold}}{\rho}\right) + Q&\leq N\\
\end{align*}
$$
"""
)
