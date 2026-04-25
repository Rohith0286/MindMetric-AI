import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
import base64
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# ---------------- CONFIG & THEME ----------------
st.set_page_config(page_title="MindMetric AI | Digital Twin", layout="wide")
P_BLUE, P_GREEN, P_RED, P_YELLOW, P_GRAY, P_PURPLE = "#A0C4FF", "#BEE1B6", "#FFADAD", "#FDFFB6", "#D3D3D3", "#C1AEFC"

# ---------------- ENGINE & DATA ----------------
MODEL_DIR, DATA_DIR = "model", "data"
MODEL_PATH, DATA_PATH = os.path.join(MODEL_DIR, "mindmetric_model.pkl"), os.path.join(DATA_DIR, "mindmetric_data.csv")
FEATURES = ['sleep_hours', 'screen_time', 'study_hours', 'physical_activity', 'sleep_quality', 'focus_level']

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

def get_clean_data():
    if os.path.exists(DATA_PATH):
        try:
            df = pd.read_csv(DATA_PATH)
            if len(df.columns) != len(FEATURES) + 1:
                os.remove(DATA_PATH)
                return pd.DataFrame(columns=FEATURES + ['productivity_score'])
            return df
        except:
            if os.path.exists(DATA_PATH): os.remove(DATA_PATH)
    return pd.DataFrame(columns=FEATURES + ['productivity_score'])

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            m = joblib.load(MODEL_PATH)
            return m if hasattr(m, 'n_features_in_') and m.n_features_in_ == len(FEATURES) else None
        except: return None
    return None

# ---------------- ANIMATED LOGO ----------------
def render_animated_logo_splash():
    LOGO_FILENAME = "WhatsApp Image 2026-04-24 at 11.12.41 PM.jpeg" 
    if os.path.exists(LOGO_FILENAME):
        with open(LOGO_FILENAME, "rb") as f:
            bin_str = base64.b64encode(f.read()).decode()
        st.markdown(f"""
            <style>
            #splash-overlay {{
                position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
                background-color: #0E1117; display: flex; justify-content: center;
                align-items: center; z-index: 999999;
                animation: fadeOutOverlay 0.6s ease-in 3.2s forwards; pointer-events: none;
            }}
            #splash-logo {{ width: 300px; opacity: 0; transform: scale(0.3); animation: popIn 0.8s forwards, flyUp 0.8s 2.4s forwards; }}
            @keyframes popIn {{ to {{ opacity: 1; transform: scale(1); }} }}
            @keyframes flyUp {{ to {{ transform: translateY(-1200px); opacity: 0; }} }}
            @keyframes fadeOutOverlay {{ to {{ opacity: 0; visibility: hidden; }} }}
            .status-card {{ background-color: rgba(255, 255, 255, 0.05); padding: 15px; border-radius: 12px; border: 1px solid rgba(255, 255, 255, 0.1); text-align: center; }}
            .insight-box {{ padding: 22px; border-radius: 12px; margin-bottom: 20px; border-left: 6px solid; line-height: 1.7; font-size: 0.98rem; min-height: 160px; }}
            .factor-tag {{ font-weight: bold; text-transform: uppercase; font-size: 0.75rem; color: #aaa; margin-bottom: 8px; display: block; }}
            
            /* Sidebar Button Styling to remove bullet points */
            [data-testid="stSidebarNav"] {{display: none;}}
            .stButton > button {{
                width: 100%; border: none; background: transparent; text-align: left;
                padding: 10px 15px; font-size: 1.05rem; color: #ffffff;
            }}
            .stButton > button:hover {{ background: rgba(255,255,255,0.05); color: {P_PURPLE}; }}
            </style>
            <div id="splash-overlay"><img id="splash-logo" src="data:image/jpeg;base64,{bin_str}"></div>
            """, unsafe_allow_html=True)

render_animated_logo_splash()

# ---------------- PERSISTENT STATE ----------------
for key, val in [('s_val', 7.5), ('st_val', 4.0), ('sc_val', 3.0), ('pa_val', 45.0), ('sq_val', 3), ('fl_val', 3)]:
    if key not in st.session_state: st.session_state[key] = val
if 'current_score' not in st.session_state: st.session_state.current_score = None
if 'analysis_run' not in st.session_state: st.session_state.analysis_run = False
if 'nav' not in st.session_state: st.session_state.nav = "Real-Time Lab"

# ---------------- SIDEBAR NAVIGATION (NO BULLETS) ----------------
with st.sidebar:
    st.title("MindMetric")
    st.write("Navigate System")
    
    if st.button("Real-Time Lab"): st.session_state.nav = "Real-Time Lab"
    if st.button("Present Data"): st.session_state.nav = "Present Data"
    if st.button("Deep Insights"): st.session_state.nav = "Deep Insights"
    if st.button("Weekly Analytics"): st.session_state.nav = "Weekly Analytics"
    
    st.divider()
    score_display = st.session_state.current_score if st.session_state.current_score else 0.0
    st.metric("Live Digital Twin Score", f"{score_display:.1f}/100")
    st.divider()
    st.caption("Auto-Log & Learn: Enabled")

# ---------------- MAIN CONTENT AREA ----------------
nav = st.session_state.nav

# --- TAB 1: REAL-TIME LAB ---
if nav == "Real-Time Lab":
    st.title("Real-Time Lab")
    c1, c2 = st.columns([1.2, 1], gap="large")
    with c1:
        st.subheader("Simulation Controls")
        col_hab1, col_hab2 = st.columns(2)
        with col_hab1:
            s = st.slider("Sleep Hours", 0.0, 14.0, st.session_state.s_val)
            st_hr = st.slider("Study Hours", 0.0, 14.0, st.session_state.st_val)
        with col_hab2:
            sc = st.slider("Screen Time", 0.0, 14.0, st.session_state.sc_val)
            pa = st.slider("Exercise (mins)", 0.0, 180.0, st.session_state.pa_val)
        st.divider()
        st.info("Personalization Factors")
        col_pers1, col_pers2 = st.columns(2)
        with col_pers1:
            sq = st.select_slider("Sleep Quality", options=[1, 2, 3, 4, 5], value=st.session_state.sq_val)
        with col_pers2:
            fl = st.select_slider("Focus Level", options=[1, 2, 3, 4, 5], value=st.session_state.fl_val)
            
        total_time = s + st_hr + sc + (pa/60)
        if total_time > 24: st.error("Paradox: Day exceeds 24 hours!")
        elif st.button("Run Productivity Analysis & Log Data", key="run_main"):
            h_base = (st_hr * 12) + (s * 6) + (pa / 5) - (sc * 10)
            q_mult = ((sq * 0.4) + (fl * 0.6)) / 3
            calc_score = np.clip(h_base * q_mult, 0, 100)
            st.session_state.update({'s_val': s, 'st_val': st_hr, 'sc_val': sc, 'pa_val': pa, 'sq_val': sq, 'fl_val': fl, 'current_score': calc_score, 'analysis_run': True})
            df = get_clean_data()
            df.loc[len(df)] = [s, sc, st_hr, pa, sq, fl, calc_score]
            df.to_csv(DATA_PATH, index=False)
            if len(df) > 1:
                X, y = df[FEATURES], df['productivity_score']
                model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)
                joblib.dump(model, MODEL_PATH)
            st.rerun()
    with c2:
        if st.session_state.analysis_run:
            st.markdown(f"### Predicted Productivity: <span style='color:{P_PURPLE}'>{st.session_state.current_score:.1f}/100</span>", unsafe_allow_html=True)
            fig, ax = plt.subplots(); fig.patch.set_facecolor('none')
            ax.pie([st.session_state.s_val, st.session_state.st_val, st.session_state.sc_val, st.session_state.pa_val/60, max(0, 24-total_time)], 
                   labels=["Sleep", "Study", "Screen", "Exercise", "Other"], autopct='%1.1f%%', 
                   colors=[P_BLUE, P_GREEN, P_RED, P_YELLOW, P_GRAY], textprops={'color':"white", 'weight':'bold'})
            ax.add_artist(plt.Circle((0,0), 0.70, fc='#262730')); st.pyplot(fig)

# --- TAB 2: PRESENT DATA (RESTORED SCIENTIFIC CONTENT) ---
elif nav == "Present Data":
    st.title("Present Data")
    if st.session_state.analysis_run:
        scols = st.columns(4)
        v_list = [("Sleep", f"{st.session_state.s_val}h"), ("Screen", f"{st.session_state.sc_val}h"), 
                  ("Study", f"{st.session_state.st_val}h"), ("Exercise", f"{st.session_state.pa_val}m")]
        for i, (l, val) in enumerate(v_list):
            scols[i].markdown(f"<div class='status-card'><strong>{l}</strong><br>{val}</div>", unsafe_allow_html=True)
        st.divider()
        st.subheader("Subject-Wise Correlative Analysis")
        p1, p2 = st.columns(2)
        with p1:
            if st.session_state.s_val >= 7:
                st.markdown(f"""<div class='insight-box' style='border-color:{P_GREEN}; background:rgba(190,225,182,0.08)'>
                <span class='factor-tag'>Neural Restoration</span><b>Habit: Sleep ({st.session_state.s_val}h)</b><br>
                Optimal duration achieved. Brain toxins were effectively cleared via the glymphatic system, ensuring neuronal firing rates remain high.</div>""", unsafe_allow_html=True)
        with p2:
            if st.session_state.sq_val >= 4:
                st.markdown(f"""<div class='insight-box' style='border-color:{P_GREEN}; background:rgba(190,225,182,0.08)'>
                <span class='factor-tag'>Efficiency Multiplier</span><b>Habit: Sleep Quality ({st.session_state.sq_val}/5)</b><br>
                High REM density improved memory consolidation and information retrieval efficiency for study tasks.</div>""", unsafe_allow_html=True)
        n1, n2 = st.columns(2)
        with n1:
            if st.session_state.sc_val > 4:
                st.markdown(f"""<div class='insight-box' style='border-color:{P_RED}; background:rgba(255,173,173,0.08)'>
                <span class='factor-tag'>Attention Residue</span><b>Habit: Screen Time ({st.session_state.sc_val}h)</b><br>
                Excessive digital stimulus induced dopamine fatigue, tethering focus to low-value stimuli and decreasing deep work capacity.</div>""", unsafe_allow_html=True)
        with n2:
            if st.session_state.s_val < 6.5:
                st.markdown(f"""<div class='insight-box' style='border-color:{P_RED}; background:rgba(255,173,173,0.08)'>
                <span class='factor-tag'>Cognitive Inhibition</span><b>Habit: Sleep Debt ({st.session_state.s_val}h)</b><br>
                Low sleep inhibited the prefrontal cortex, forcing the brain to operate on metabolic emergency reserves, leading to focus fragmentation.</div>""", unsafe_allow_html=True)
    else: st.warning("Run a lab simulation first.")

# --- TAB 3: DEEP INSIGHTS ---
elif nav == "Deep Insights":
    st.title("Interactive Predictive Analytics")
    active_model = load_model()
    if active_model and st.session_state.analysis_run:
        for feat in FEATURES:
            with st.expander(f"Predictive Impact: {feat.replace('_',' ').title()}", expanded=True):
                c_p, c_r = st.columns(2)
                base = [st.session_state.s_val, st.session_state.sc_val, st.session_state.st_val, st.session_state.pa_val, st.session_state.sq_val, st.session_state.fl_val]
                idx = FEATURES.index(feat)
                gv = list(base); gv[idx] = max(0, gv[idx]-1.0) if feat == 'screen_time' else gv[idx]+1.0
                gp = active_model.predict([gv])[0]; g_diff = gp - st.session_state.current_score
                with c_p:
                    st.markdown("**Growth Prediction**"); st.title(f"{gp:.1f}")
                    st.markdown(f"<span style='background-color:#1e3d24; color:#3dd56d; padding:2px 8px; border-radius:10px;'>↑ {abs(g_diff):+.2f}</span>", unsafe_allow_html=True)
                rv = list(base); rv[idx] = rv[idx]+1.0 if feat == 'screen_time' else max(0, rv[idx]-1.0)
                rp = active_model.predict([rv])[0]; r_diff = rp - st.session_state.current_score
                with c_r:
                    st.markdown("**Risk Prediction**"); st.title(f"{rp:.1f}")
                    st.markdown(f"<span style='background-color:#1e3d24; color:#3dd56d; padding:2px 8px; border-radius:10px;'>↑ {abs(r_diff):+.2f}</span>", unsafe_allow_html=True)
    else: st.warning("Insufficient data for AI prediction.")

# --- TAB 4: WEEKLY ANALYTICS (RESTORED ORIGINAL LAYOUT) ---
elif nav == "Weekly Analytics":
    st.title("Weekly Analytics")
    df_f = get_clean_data()
    if not df_f.empty:
        df_7 = df_f.tail(7)
        l_c, r_c = st.columns(2)
        with l_c:
            st.write("### Productivity Trend")
            st.line_chart(df_7['productivity_score'], color=P_PURPLE)
        with r_c:
            st.write("### Habit Input Summary")
            st.bar_chart(df_7[['sleep_hours', 'screen_time', 'study_hours', 'physical_activity']])
        
        st.divider()
        st.subheader("Weekly Performance Summary")
        st.markdown("**Average Weekly Productivity**")
        st.title(f"{df_7['productivity_score'].mean():.2f}/100")
        
        active_model = load_model()
        if active_model:
            imps = active_model.feature_importances_
            sorted_idx = np.argsort(imps)[::-1]
            
            # Logic: If high Screen Time is a major mover, it shouldn't be "Best Factor"
            best_feat = FEATURES[sorted_idx[0]]
            if best_feat == 'screen_time' and df_7['screen_time'].iloc[-1] > 4:
                best_feat = FEATURES[sorted_idx[1]] if len(sorted_idx) > 1 else best_feat
                
            worst_feat = FEATURES[np.argmin(imps)]
            
            b_col, w_col = st.columns(2)
            with b_col:
                st.markdown(f"""<div style='background-color:rgba(190, 225, 182, 0.1); padding:20px; border-radius:10px; border-left: 5px solid {P_GREEN}'><h4 style='color:{P_GREEN}'>Best Factor: {best_feat.replace('_',' ').title()}</h4><p><strong>Advantages:</strong> This factor provided neural stability and consistent energy.</p><p><strong>Impact:</strong> It served as your cognitive anchor, ensuring high baseline performance.</p></div>""", unsafe_allow_html=True)
            with w_col:
                st.markdown(f"""<div style='background-color:rgba(255, 173, 173, 0.1); padding:20px; border-radius:10px; border-left: 5px solid {P_RED}'><h4 style='color:{P_RED}'>Worst Factor: {worst_feat.replace('_',' ').title()}</h4><p><strong>Disadvantages:</strong> This created energy leaks and fragmented attention.</p><p><strong>Impact:</strong> This was the primary bottleneck capping your total score.</p></div>""", unsafe_allow_html=True)
        st.divider()
        st.write("### Complete History Log")
        st.dataframe(df_f, use_container_width=True)
    else: st.info("Log your daily data to unlock performance analytics.")