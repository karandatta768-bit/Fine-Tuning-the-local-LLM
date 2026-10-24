"""Local Streamlit suggestion demo. No business actions are performed."""
import streamlit as st
from src.inference import answer_customer

st.set_page_config(page_title="Local Customer Support Assistant", page_icon="💬")
st.title("Local Customer Support Assistant")
st.caption("Model-generated suggestions only. A trusted support system must verify any real order or payment action.")
message = st.text_area("Customer message", placeholder="Describe the customer's issue…", height=140)
if st.button("Generate suggestion", type="primary", disabled=not message.strip()):
    with st.spinner("Loading the local model and generating a suggestion…"):
        try:
            st.session_state["result"] = answer_customer(message)
        except Exception as exc:
            st.error(f"Unable to generate a suggestion: {exc}")
if "result" in st.session_state:
    result = st.session_state["result"]
    if "error" in result:
        st.warning(result.get("response", "Please have a support agent review this request."))
    else:
        st.subheader("Suggested classification")
        st.write(f"**Category:** {result['category']}  \n**Intent:** {result['intent']}")
        st.subheader("Suggested reply")
        st.write(result["response"])
