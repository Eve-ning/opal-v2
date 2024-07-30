import streamlit as st

st.title("Information")
st.header("How it works ELI5")
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

"""
)
st.info(
    "This is continually being worked on, go to my blog for a more detailed explanation of the algorithm: "
    "[opal v2: Explainable Map and Player Embedding Optimation](https://eve-ning.github.io/2024/07/26/opal-emb.html)"
)
