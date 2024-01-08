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

Then the distance between a map and player is related to the 
score.
```
. P1 - M1 = Higher Score
   ◄──────────────────►
  ┌────┐  ┌────┐  ┌────┐
──┤ M1 ├──┤ M2 ├──┤ P1 ├──►
  └────┘  └────┘  └────┘
           ◄──────────►
        P1 - M2 = Lower Score
```
I believe it's intuitive that larger distances indicates larger skill gap,
therefore, higher scores.   

By using a ML model to fit this distance to a score,
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
the difference in RC difficulty, resulting in the same distance, thus same score.

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

We'll denote this difference as $Q - M = \Delta \in\mathbb{R}^{n_D}$.

### Mapping $\Delta$ to a score

There are 2 constraints we have:

1) The transform is non-linear.
2) The transform must be monotonically increasing. Which means, if we increase
   the distance, the score must increase.
 
In order to achieve this, we can use a neural network. However, to satisfy
constraint 2, we need to force all weights to be positive, to guarantee a
monotonically increasing function.

To do so, we use the exponential of
the weights when we apply the linear transformation, this allows the weights
to be positive, while still allowing the model to learn non-linearity

The architecture is laughably simple, but it works:

$$
\Delta
\rightarrow \text{ExpLinear(D, N)}
\rightarrow \text{ReLU}
\rightarrow \text{ExpLinear(N, 1)}
\rightarrow \text{Sigmoid}
\rightarrow \text{Score}
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
                     │     │     │    ┌──┴─ ─┐
                     │     │     └───►│ W00 │
                     │     │          ├─────┤
                     │     │          │  .  │
                     │     └─────────►│  .  │
                     │                │  .  │
                     │                ├─────┤
                     └───────────────►│ WD0 │
                                      └─────┘
```

### Weighting the dimensions

Currently, we only use 2 dimensions, $RC$ and $LN$.
With this, we can weigh the accuracy contribution of each dimension
as we have the density of each component in our data.

This allows us to not only learn the effects of each dimension, but also
ensure that the dimension is labelled correctly when trained.

"""
)
