import streamlit as st

st.title("Information")
st.header("How it works ELI5")
st.write(
    r"""
As a simple example, lets put 2 maps and a player on a line. 
**By placing any non-numerical object on a line, we call this operation
an embedding.** 

- The :orange[further right] a map is, the :orange[harder] it is, 
- The :orange[further right] a player is, the :orange[better] they are.

Let's place 2 maps :blue[M1, M2] and a player :red[P1] on a number line below.

```
. ┌────┐  ┌────┐  ┌────┐
──┤ M1 ├──┤ M2 ├──┤ P1 ├──►
  └────┘  └────┘  └────┘
     ◄────── ──────►
 Lower Skill Higher Skill
```

Let's assume that the distance between a map and player is related to accuracy
in some manner. So, because the distance between :red[P1] and :blue[M1] is 
larger than between :red[P1] and :blue[M2], the accuracy of :red[P1] on 
:blue[M1] higher than on :blue[M2].

```
. P1 - M1 = Higher Acc
   ◄──────────────────►
  ┌────┐  ┌────┐  ┌────┐
──┤ M1 ├──┤ M2 ├──┤ P1 ├──►
  └────┘  └────┘  └────┘
           ◄──────────►
        P1 - M2 = Lower Acc
```    

However, because distance can't be :violet[transformed] directly as accuracy,
we use a machine learning model to learn this :violet[transformation].

$$
\text{model}(P - M) = \text{accuracy}
$$

You can observe what relationship is learnt under the (Debug) Delta to Accuracy
Transformation expander in the main page.

There are many more details to this, which I've written in an ongoing blog post:
[opal v2: Explainable Map and Player Embedding Optimation](https://eve-ning.github.io/2024/07/26/opal-v2.html)
with details such as:
- how I got the dataset
- how I constraint the relationship to be monotonically positive
- further works on opal
- and general thoughts on the algorithm.
"""
)
