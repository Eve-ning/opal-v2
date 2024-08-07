from lightning import LightningModule
import streamlit as st
from opal.data import OsuDataModule
from opal.evaluate.evaluate import prediction_error


def prediction_error_component(m: LightningModule, dm: OsuDataModule):
    df = prediction_error(m, dm)
    st.dataframe(df)
