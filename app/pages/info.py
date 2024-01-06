import streamlit as st

st.title("Information")
st.header("How it works ELI5")
st.write(
    """
As a simple example, lets put 2 maps and a player on a line.

The further right a map is, the harder it is, 
and the further right a player is, the better they are.
"""
)
st.code(
    """
. ┌────┐  ┌────┐  ┌────┐
──┤ M1 ├──┤ M2 ├──┤ P1 ├──►
  └────┘  └────┘  └────┘
    ◄────── ──────►
Lower Skill Higher Skill
    """,
    language="text",
)


st.write(
    """
Then the distance between a map and player is related to the 
score.
    """
)
st.warning(
    "We say distance, but it's actually displacement, as it can be negative."
)

st.code(
    """
. P1 - M1 = Higher Score
   ◄──────────────────►
  ┌────┐  ┌────┐  ┌────┐
──┤ M1 ├──┤ M2 ├──┤ P1 ├──►
  └────┘  └────┘  └────┘
           ◄──────────►
        P1 - M2 = Lower Score
""",
    language="text",
)

st.write(
    """
I believe it's intuitive that larger distances indicates larger skill gap,
therefore, higher scores.   

By using a ML model to fit this distance to a score,
we get a model that learns the best location to place these maps and players
such that the constraint is best satisfied.
"""
)
st.subheader("A 2nd Dimension")
st.write(
    """
However, maps and players are much more complex.
They can be LN or RC-difficulty biased: 
better at one type of difficulty than the other.

In fact, we can create a scenario where the distance between $M_1$ and $P_1$,
and the distance between $M_2$ and $P_1$ are the same.
"""
)

st.code(
    """
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
    """,
    language="text",
)

st.write(
    """
As shown, because $M_2$ is LN-easy, while $M_1$ is LN-hard, it compensates
the difference in RC difficulty, resulting in the same distance, thus same score.

This argument, *unfortunately*, can be extended to infinitely more dimensions.

However, with more dimensions, it's harder to understand what exactly each
dimension represents.
It's up to the creator to decide if they want an explainable model, 
or an accurate model.
    """
)

st.write(
    r"""
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

"""
)
