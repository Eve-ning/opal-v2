import streamlit as st
import plotly.express as px


def emb_polar_component(embbeddings: dict[str, float]):
    st.plotly_chart(
        px.line_polar(
            r=embbeddings.values(),
            theta=embbeddings.keys(),
            line_close=True,
        ).update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=False, range=[-2, 2]),
            )
        ),
        use_container_width=True,
    )


#
#
# left, right = st.columns(2)
# with left:
#     df_uid_i = df_uid.loc[(username, useryear), ["RC", "LN"]]
#     emb_polar(
#         {
#             **{f"RC{e}": rc for e, rc in enumerate(df_uid_i["RC"])},
#             **{f"LN{e}": ln for e, ln in enumerate(df_uid_i["LN"])},
#         }
#     )
# with right:
#     df_mid_i = df_mid.loc[(mapname, mapspeed), ["RC", "LN"]]
#     emb_polar(
#         {
#             **{f"RC{e}": rc for e, rc in enumerate(df_mid_i["RC"])},
#             **{f"LN{e}": ln for e, ln in enumerate(df_mid_i["LN"])},
#         }
#     )
