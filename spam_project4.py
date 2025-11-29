import streamlit as st
import joblib
import pandas as pd

#==================== PAGE STYLING =====================#
st.set_page_config(page_title="Spam Classifier", page_icon="🚫", layout="wide")

st.markdown("""
    <div style="background: linear-gradient(90deg,#ff4b2b,#ff416c);
                padding:25px;border-radius:10px;margin-bottom:25px;">
        <h1 style="text-align:center;color:white;font-size:45px;">
            🚫 Spam Classifier Project 📩
        </h1>
        <p style="text-align:center;color:white;font-size:18px;margin-top:-10px;">
            Machine Learning Based Email/SMS Spam Detection
        </p>
    </div>
""", unsafe_allow_html=True)

#==================== LOAD MODEL =====================#
model = joblib.load("spam_clf.pkl")


#==================== UI LAYOUT =====================#
col1, col2 = st.columns([1.2, 2])

#============ SINGLE MESSAGE PREDICTION SECTION ============#
with col1:
    
    st.markdown("""
        <div style="background:#f1f1f1;padding:20px;border-radius:10px;
                    box-shadow:0px 4px 15px rgba(0,0,0,0.1);">
            <h2 style="text-align:center;color:#ff416c;">🔹 Single Message Check</h2>
        </div>
    """, unsafe_allow_html=True)

    text = st.text_area("Enter Message to Verify:", height=140)

    if st.button("Predict Output", use_container_width=True):
        if text.strip() == "":
            st.warning("⚠ Please enter a message first!")
        else:
            result = model.predict([text])[0]
            
            if result == "spam":
                st.error("🚫 Result: SPAM (Irrelevant Message)")
            else:
                st.success("✅ Result: HAM (Legitimate Message)")


#============ BULK CSV PREDICTION SECTION ============#

with col2:
    st.markdown("<div class='box'><h3 style='color:#009432;'>📁 Bulk File Detection</h3>", unsafe_allow_html=True)

    file = st.file_uploader("Upload CSV/TXT Messages", type=['csv','txt'])

    if file:
        df = pd.read_csv(file, header=None, names=["Message"])
        st.info(f"File Loaded: {len(df)} Messages Found")

        if st.button("Run Bulk Prediction"):
            df["Result"] = model.predict(df["Message"])

            # --- NEW FEATURE: STATS COUNTING ---
            total = len(df)
            spam_count = (df["Result"] == "spam").sum()
            ham_count = (df["Result"] == "ham").sum()

            # --- Stats Display UI ---
            st.markdown("""
                <h3 style='text-align:center;color:#333;'>📊 Prediction Summary</h3>
                <div style='display:flex;justify-content:center;gap:30px;margin-top:15px;'>
                    <div style='background:#ffe6e6;padding:18px 30px;border-radius:8px;'>
                        <h2 style='color:#e60000;text-align:center;'>Spam</h2>
                        <h1 style='text-align:center;'>""" + str(spam_count) + """</h1>
                    </div>
                    <div style='background:#e6ffe6;padding:18px 30px;border-radius:8px;'>
                        <h2 style='color:#009432;text-align:center;'>Relevent</h2>
                        <h1 style='text-align:center;'>""" + str(ham_count) + """</h1>
                    </div>
                    <div style='background:#e6f2ff;padding:18px 30px;border-radius:8px;'>
                        <h2 style='color:#0066cc;text-align:center;'>Total</h2>
                        <h1 style='text-align:center;'>""" + str(total) + """</h1>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # Color table
            def highlight(r):
                return ['background-color:#ffcccc' if x=='spam' else 'background-color:#ccffdd' for x in r]

            st.dataframe(df.style.apply(highlight, subset=["Result"]), height=350)

    st.markdown("</div>", unsafe_allow_html=True)
