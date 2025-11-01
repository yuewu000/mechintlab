
import streamlit as st
import json, glob

st.title("MechInt Lab — Minimal Dashboard")
st.write("Browse recent runs and metrics.")

for path in sorted(glob.glob("runs/*/*.jsonl")):
    st.subheader(path)
    with open(path) as f:
        lines = [json.loads(x) for x in f]
    st.line_chart({ "recon": [d.get("recon", None) for d in lines],
                    "l1": [d.get("l1", None) for d in lines] })
