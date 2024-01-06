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
┌────┐  ┌────┐  ┌────┐
│ M1 ├──┤ M2 ├──┤ P1 │
└────┘  └────┘  └────┘
       ◄──────  ──────►
   Lower Skill  Higher Skill
    """,
    language="text",
)


st.write(
    """
Then the distance between a map and a player is related to the 
score of the player on the map.
    """
)
st.warning("We say distance, but it's actually displacement, as it can be negative.")

st.code(
    """
P1 - M1 = Higher Score
 ◄──────────────────►
┌────┐  ┌────┐  ┌────┐
│ M1 ├──┤ M2 ├──┤ P1 │
└────┘  └────┘  └────┘
         ◄──────────►
      P1 - M2 = Lower Score
""",
    language="text",
)

st.write(
    """
I believe it's intuitive that larger distances indicate a larger skill gap,
therefore, larger gaps equate to higher scores.   

By simply using a ML model to equate this distance to a score, minimizing
the error, we can get a model that can learn where all of these maps and players
should be placed on this line.
"""
)
st.subheader("A 2nd Dimension")
st.write(
    """
However, maps and players are actually more complex than this.
Maps and players can be LN or RC-difficulty biased, meaning that they are
better at one type of difficulty than the other.

In fact, we can even create a scenario where $P_1 - M_1 = P_1 - M_2$ despite
$M_1$ being harder than $M_2$ in this dimension.
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
the difference in RC difficulty, resulting in the same score.

This argument, unfortunately, can be extended to infinitely more dimensions.
However, with more dimensions, it's harder to understand what exactly each
dimension represents. 
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
dimension $n_D$. If we only had the dimensions $RC$ and $LN$, we would have
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
