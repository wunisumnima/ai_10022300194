# Name: Sumnima Wuni | Index Number: 10022300194
import base64
import os
import urllib.parse
from pathlib import Path

import streamlit as st

try:
    import config as project_config
except ImportError:
    project_config = None

from rag_core import RAGChatbot, _clean_api_key


@st.cache_resource(show_spinner="Loading index…")
def get_bot(strategy: str, _llm_bind: int = 0) -> RAGChatbot:
    # Force cache refresh by using _llm_bind parameter
    return RAGChatbot(docs_path="docs", strategy=strategy, top_k=3)


st.set_page_config(
    page_title="Academic City RAG Assistant",
    page_icon="🤖",
    layout="centered",
)

_ROOT = Path(__file__).resolve().parent
_bg_candidates = [
    _ROOT / "assets" / "ocean_background.png",
    _ROOT / "assets" / "ocean_bg.png",
    Path(
        r"C:\Users\Administrator\.cursor\projects\c-Files-Slides-lvl-300-Lvl-300-second-sem-Introduction-to-Artificial-Intelligence-Chatbot-2026\assets\c__Users_Administrator_AppData_Roaming_Cursor_User_workspaceStorage_f75d950f6abe2d58fc546d882d9621d6_images_WhatsApp_Image_2026-04-19_at_20.54.48-e90c3166-71cb-43be-a372-cd3b16e23d89.png"
    ),
    Path(
        r"C:\Users\Administrator\.cursor\projects\c-Files-Slides-lvl-300-Lvl-300-second-sem-Introduction-to-Artificial-Intelligence-Chatbot-2026\assets\c__Users_Administrator_AppData_Roaming_Cursor_User_workspaceStorage_f75d950f6abe2d58fc546d882d9621d6_images_WhatsApp_Image_2026-04-19_at_20.54.48-e76d79ce-5234-4392-aa18-92639f8ad19a.png"
    ),
]
_bg_b64 = ""
for _p in _bg_candidates:
    if _p.exists():
        _bg_b64 = base64.b64encode(_p.read_bytes()).decode("utf-8")
        break

_bg_layer = (
    f'background-image: url("data:image/png;base64,{_bg_b64}");'
    if _bg_b64
    else "background: linear-gradient(160deg, #003b46 0%, #001820 55%, #000a0c 100%);"
)


def _svg_data_url(svg: str) -> str:
    compact = " ".join(svg.split())
    return "data:image/svg+xml," + urllib.parse.quote(compact)


_wave_deep_svg = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 110" preserveAspectRatio="none">
  <path fill="rgba(0,52,68,0.72)" d="M0,62 C180,22 360,102 540,62 C720,22 900,98 1080,62 C1140,52 1170,68 1200,58 L1200,110 L0,110 Z"/>
</svg>
"""
_wave_foam_svg = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 96" preserveAspectRatio="none">
  <path fill="rgba(255,255,255,0.16)" d="M0,44 C220,8 420,78 640,44 C820,14 980,72 1200,48 L1200,96 L0,96 Z"/>
  <path fill="rgba(180,230,240,0.09)" d="M0,58 C260,92 480,18 760,58 C920,78 1040,32 1200,52 L1200,96 L0,96 Z"/>
  <path fill="none" stroke="rgba(255,255,255,0.2)" stroke-width="1.8"
        d="M0,34 C200,6 400,62 600,34 C800,8 1000,56 1200,32"/>
</svg>
"""
_wave_deep = _svg_data_url(_wave_deep_svg)
_wave_foam = _svg_data_url(_wave_foam_svg)

_ai_mesh_svg = """
<svg xmlns="http://www.w3.org/2000/svg" width="72" height="72" viewBox="0 0 72 72">
  <g fill="rgba(165,180,252,0.14)" stroke="rgba(129,140,248,0.18)" stroke-width="0.6">
    <circle cx="10" cy="14" r="2"/>
    <circle cx="36" cy="10" r="2"/>
    <circle cx="62" cy="18" r="2"/>
    <circle cx="22" cy="38" r="2"/>
    <circle cx="50" cy="36" r="2"/>
    <circle cx="14" cy="58" r="2"/>
    <circle cx="40" cy="62" r="2"/>
    <line x1="10" y1="14" x2="36" y2="10"/>
    <line x1="36" y1="10" x2="62" y2="18"/>
    <line x1="10" y1="14" x2="22" y2="38"/>
    <line x1="36" y1="10" x2="50" y2="36"/>
    <line x1="22" y1="38" x2="50" y2="36"/>
    <line x1="22" y1="38" x2="14" y2="58"/>
    <line x1="50" y1="36" x2="40" y2="62"/>
  </g>
</svg>
"""
_ai_mesh = _svg_data_url(_ai_mesh_svg)

css = """
<style>
[data-testid="stAppViewContainer"] {
    __BG_LAYER__
    background-size: cover;
    background-position: center center;
    background-repeat: no-repeat;
    background-attachment: fixed;
    position: relative;
    overflow-x: clip;
    animation: ocean-bg-pan 32s ease-in-out infinite alternate;
}
[data-testid="stAppViewContainer"]::before {
    content: "";
    position: fixed;
    left: 0;
    right: 0;
    bottom: 0;
    height: 110px;
    z-index: 1;
    pointer-events: none;
    background-image: url("__WAVE_DEEP__");
    background-repeat: repeat-x;
    background-position: 0 100%;
    background-size: 1200px 110px;
    opacity: 0.88;
    animation: seawave-drift-deep 19s linear infinite, wave-glimmer 7s ease-in-out infinite;
}
[data-testid="stAppViewContainer"]::after {
    content: "";
    position: fixed;
    left: 0;
    right: 0;
    bottom: 0;
    height: 96px;
    z-index: 1;
    pointer-events: none;
    background-image: url("__WAVE_FOAM__");
    background-repeat: repeat-x;
    background-position: 0 100%;
    background-size: 1200px 96px;
    opacity: 0.75;
    animation: seawave-drift-foam 26s linear infinite reverse, wave-glimmer 9s ease-in-out infinite 1s;
}
@keyframes ocean-bg-pan {
    0% { background-position: 47% 42%; }
    100% { background-position: 53% 58%; }
}
@keyframes wave-glimmer {
    0%, 100% { opacity: 0.68; filter: brightness(1); }
    50% { opacity: 0.9; filter: brightness(1.06); }
}
@keyframes seawave-drift-deep {
    0% { background-position: 0 100%; }
    100% { background-position: 1200px 100%; }
}
@keyframes seawave-drift-foam {
    0% { background-position: 0 100%; }
    100% { background-position: 1200px 100%; }
}
[data-testid="stAppViewContainer"] > .main {
    position: relative;
    z-index: 2;
    background: transparent;
}
[data-testid="stAppViewContainer"] .block-container {
    position: relative;
    padding: 1.75rem 1.5rem 2rem;
    max-width: 720px;
    margin-top: 0.5rem;
    margin-bottom: 2rem;
    background-color: rgba(14, 14, 18, 0.93);
    background-image: url("__AI_MESH__");
    background-size: 72px 72px;
    background-repeat: repeat;
    background-blend-mode: soft-light;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.45);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    animation: panel-border-shimmer 14s ease-in-out infinite;
}
[data-testid="stAppViewContainer"] .block-container::before {
    content: "";
    position: absolute;
    left: 0;
    right: 0;
    top: 0;
    height: 3px;
    border-radius: 16px 16px 0 0;
    background: linear-gradient(90deg, #2dd4bf, #6366f1, #c084fc, #38bdf8, #2dd4bf);
    background-size: 220% 100%;
    animation: ai-accent-bar 6s linear infinite;
    pointer-events: none;
    z-index: 1;
}
@keyframes ai-accent-bar {
    0% { background-position: 0% 50%; }
    100% { background-position: 100% 50%; }
}
.ai-brand-strip {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.45rem;
    margin: 0.15rem 0 0.85rem 0;
}
.ai-assistant-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.32rem 0.65rem 0.32rem 0.5rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    color: #e0e7ff;
    border: 1px solid rgba(129, 140, 248, 0.45);
    background: linear-gradient(135deg, rgba(79, 70, 229, 0.35), rgba(14, 165, 233, 0.2));
    box-shadow: 0 0 0 1px rgba(0, 0, 0, 0.2) inset;
    animation: ai-badge-pulse 3.5s ease-in-out infinite;
}
.ai-assistant-badge svg {
    width: 1.05rem;
    height: 1.05rem;
    flex-shrink: 0;
    color: #a5b4fc;
}
@keyframes ai-badge-pulse {
    0%, 100% { box-shadow: 0 0 0 1px rgba(0, 0, 0, 0.2) inset, 0 0 0 0 rgba(99, 102, 241, 0); }
    50% { box-shadow: 0 0 0 1px rgba(0, 0, 0, 0.2) inset, 0 0 14px 1px rgba(99, 102, 241, 0.35); }
}
.ai-chip {
    display: inline-block;
    font-size: 0.68rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    padding: 0.22rem 0.5rem;
    border-radius: 6px;
    color: #cbd5e1;
    border: 1px solid rgba(148, 163, 184, 0.28);
    background: rgba(24, 26, 36, 0.92);
}
.ai-chip.rag {
    color: #99f6e4;
    border-color: rgba(45, 212, 191, 0.45);
    background: rgba(15, 118, 110, 0.22);
}
.ai-chip.llm {
    color: #fde68a;
    border-color: rgba(250, 204, 21, 0.35);
    background: rgba(113, 63, 18, 0.2);
}
.ai-pipeline-hint {
    color: #94a3b8;
    font-size: 0.82rem;
    line-height: 1.45;
    margin: -0.35rem 0 1.1rem 0;
    border-left: 3px solid rgba(99, 102, 241, 0.55);
    padding-left: 0.65rem;
}
.ai-pipeline-hint strong {
    color: #c7d2fe;
    font-weight: 600;
}
@keyframes panel-border-shimmer {
    0%, 100% { border-color: rgba(255, 255, 255, 0.07); box-shadow: 0 12px 40px rgba(0, 0, 0, 0.45); }
    50% { border-color: rgba(120, 200, 220, 0.14); box-shadow: 0 14px 44px rgba(0, 40, 50, 0.35); }
}
h1 {
    font-size: 1.65rem;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 0.25rem;
    animation: title-soft 0.7s ease-out both;
}
@keyframes title-soft {
    from { opacity: 0; transform: translateY(-6px); }
    to { opacity: 1; transform: translateY(0); }
}
.subtitle { color: #c8d2e0; font-size: 0.95rem; margin-bottom: 1.25rem; }
.subtitle code {
    background: rgba(46, 160, 67, 0.2);
    color: #7ee787;
    padding: 0.1rem 0.35rem;
    border-radius: 4px;
}
div[data-testid="stExpander"] {
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 10px;
    background: rgba(22, 22, 28, 0.95);
}
div[data-testid="stExpander"] summary {
    color: #e8ecf3;
}
.stTextArea textarea {
    background-color: rgba(26, 26, 32, 0.98) !important;
    color: #f2f4f8 !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 10px !important;
}
div[data-baseweb="select"] > div {
    background-color: rgba(26, 26, 32, 0.98) !important;
    border-color: rgba(255, 255, 255, 0.12) !important;
    color: #f2f4f8 !important;
}
.stSlider label { color: #dbe4ef !important; }
.stSlider [data-baseweb="slider"] [role="slider"] {
    background-color: #e95454 !important;
    border: 2px solid #ff7a7a !important;
}
.stSlider [data-baseweb="slider"] [data-testid="stTickBarMin"],
.stSlider [data-baseweb="slider"] [data-testid="stTickBarMax"] { color: #9aa7bc !important; }
.stSlider [data-baseweb="slider"] div[data-testid="stSliderThumbValue"] {
    color: #ff8a8a !important;
}
.stButton button[kind="primary"] {
    width: 100%;
    border-radius: 8px;
    font-weight: 600;
    background-color: #e95454 !important;
    border: 1px solid #ff6b6b !important;
    color: #ffffff !important;
    transition: transform 0.18s ease, filter 0.18s ease;
    animation: btn-glow 3.2s ease-in-out infinite;
}
.stButton button[kind="primary"]:hover {
    transform: translateY(-1px);
    filter: brightness(1.06);
}
@keyframes btn-glow {
    0%, 100% { box-shadow: 0 0 0 0 rgba(233, 84, 84, 0); }
    50% { box-shadow: 0 0 20px 2px rgba(233, 84, 84, 0.28); }
}
.stMarkdown, .stMarkdown p, .stCaption { color: #e8ecf3; }
label[data-testid="stWidgetLabel"] p { color: #dbe4ef !important; }
@media (prefers-reduced-motion: reduce) {
    [data-testid="stAppViewContainer"] {
        animation: none !important;
    }
    [data-testid="stAppViewContainer"]::before,
    [data-testid="stAppViewContainer"]::after {
        animation: none !important;
        opacity: 0.82;
    }
    [data-testid="stAppViewContainer"] .block-container {
        animation: none !important;
    }
    [data-testid="stAppViewContainer"] .block-container::before {
        animation: none !important;
    }
    .ai-assistant-badge {
        animation: none !important;
    }
    h1 { animation: none !important; }
    .stButton button[kind="primary"] {
        animation: none !important;
        transition: none !important;
    }
}
</style>
"""
st.markdown(
    css.replace("__BG_LAYER__", _bg_layer)
    .replace("__WAVE_DEEP__", _wave_deep)
    .replace("__WAVE_FOAM__", _wave_foam)
    .replace("__AI_MESH__", _ai_mesh),
    unsafe_allow_html=True,
)

st.markdown(
    """
<div style="background-color: #004b9b; padding: 24px 32px; border-radius: 4px 4px 32px 32px; display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; margin-top: -10px;">
  <div>
    <div style="background-color: #1a56db; display: inline-block; padding: 4px 12px; margin-bottom: 12px;">
      <h1 style="margin: 0; font-size: 2.2rem; color: #ffffff; line-height: 1.2; animation: none;">Academic City RAG Assistant</h1>
    </div>
    <div style="color: #e2e8f0; font-size: 0.95rem;">
      A Retrieval-Augmented Generation assistant for Academic City queries.
    </div>
  </div>
  <div style="text-align: right; color: #ffffff; font-size: 0.9rem; line-height: 1.5;">
    CS4241 - Introduction to Artificial Intelligence<br>
    April 2026 Project
  </div>
</div>
<h2 style="color: #ffffff; margin-bottom: 24px; font-size: 1.7rem; font-weight: 700;">Welcome to the Academic City RAG System</h2>
    """,
    unsafe_allow_html=True,
)

if "last_logs" not in st.session_state:
    st.session_state.last_logs = None
if "llm_bind" not in st.session_state:
    st.session_state.llm_bind = 0
if "pure_baseline" not in st.session_state:
    st.session_state.pure_baseline = None

_cfg_groq = _clean_api_key(getattr(project_config, "GROQ_API_KEY", None)) if project_config else None
has_llm = bool(_clean_api_key(os.getenv("GROQ_API_KEY")) or _cfg_groq)
if not has_llm:
    st.info("Set **GROQ_API_KEY** below or put your key in **config.py**, then **Submit**.")

with st.expander("Override Groq key (optional — default is config.py)", expanded=False):
    groq_in = st.text_input("Groq API key", type="password", placeholder="Leave blank to use config.py")
    if st.button("Apply Groq key", use_container_width=True):
        if groq_in.strip():
            os.environ["GROQ_API_KEY"] = groq_in.strip()
        st.session_state.llm_bind += 1
        st.success("Applied. Submit your question again.")
        st.rerun()

c_set1, c_set2 = st.columns(2)
with c_set1:
    strategy = st.selectbox("Chunking", ["fixed", "structure"], index=0)
with c_set2:
    top_k = st.slider("Sources", 1, 5, 3)

with st.form("query_form", clear_on_submit=False):
    prompt = st.text_area(
        "Your question",
        height=120,
        placeholder="Ask about the budget PDF or election CSV…",
        label_visibility="collapsed",
    )
    submitted = st.form_submit_button("Submit", type="primary", use_container_width=True)

if submitted:
    if not (prompt or "").strip():
        st.warning("Enter a question, then press Submit.")
    else:
        st.session_state.pure_baseline = None
        bot = get_bot(strategy, st.session_state.llm_bind)
        bot.top_k = top_k
        with st.spinner("Searching dataset and drafting answer…"):
            st.session_state.last_logs = bot.query(prompt.strip(), include_pure_baseline=False)

logs = st.session_state.last_logs
if logs:
    st.markdown("### Answer")
    ans = logs["answer"]
    if ans.startswith("**From your data (no API key)"):
        st.warning(ans)
    elif "Model call failed" in ans or "Pure LLM call failed" in ans:
        st.error(ans)
    else:
        st.markdown(ans)

    with st.expander("Sources used", expanded=False):
        for item in logs["retrieved"]:
            page = item["metadata"].get("page", "—")
            st.caption(f"{item['source']} · p.{page} · score {item['combined_score']:.3f}")
            st.text(item["text"][:800] + ("…" if len(item["text"]) > 800 else ""))
            st.divider()

    with st.expander("Compare: RAG vs model-only", expanded=False):
        a, b = st.columns(2)
        with a:
            st.caption("With dataset (RAG)")
            st.write(logs["answer"])
        with b:
            st.caption("Model-only (no dataset)")
            if st.session_state.pure_baseline is not None:
                st.write(st.session_state.pure_baseline)
            elif st.button("Load model-only (extra API call)", key="load_pure"):
                b2 = get_bot(strategy, st.session_state.llm_bind)
                with st.spinner("Calling Groq…"):
                    st.session_state.pure_baseline = b2.generator.pure_llm(logs["query"])
                st.rerun()
            else:
                st.caption("Click the button to fetch a baseline answer (slower).")

    with st.expander("Technical (prompt & timing)", expanded=False):
        st.code(logs["prompt"], language="text")
        st.json(logs["latency_ms"])

    with st.expander("Was this helpful?"):
        note = st.text_input("Optional note", key="fb_note", label_visibility="collapsed", placeholder="Optional feedback…")
        c1, c2 = st.columns(2)
        bot_fb = get_bot(strategy, st.session_state.llm_bind)
        bot_fb.top_k = top_k
        with c1:
            if st.button("Yes", use_container_width=True):
                bot_fb.save_feedback(query=logs["query"], answer=logs["answer"], rating="up", note=note)
                st.toast("Thanks — saved.")
        with c2:
            if st.button("No", use_container_width=True):
                bot_fb.save_feedback(query=logs["query"], answer=logs["answer"], rating="down", note=note)
                st.toast("Thanks — saved.")

st.divider()

# Manual Logs & Experiments Section
with st.expander("📊 Manual Logs & Experiments", expanded=False):
    # Initialize session state variables
    if 'current_experiment' not in st.session_state:
        st.session_state.current_experiment = None
    
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Manual Logs", "🧪 Experiments", "💬 Chat History", "📋 View Logs"])
    
    with tab1:
        st.subheader("Add Manual Log")
        log_level = st.selectbox("Log Level", ["INFO", "WARNING", "ERROR", "DEBUG"])
        log_category = st.selectbox("Category", ["general", "manual", "debug", "performance"])
        log_message = st.text_area("Message", height=100, placeholder="Enter your log message here...")
        
        if st.button("Add Log", use_container_width=True):
            if log_message.strip():
                bot = get_bot(strategy, st.session_state.llm_bind)
                bot.log_manager.add_log(log_level, log_message, log_category)
                st.success(f"Log added: [{log_level}] {log_message}")
                st.rerun()
            else:
                st.error("Please enter a log message.")
    
    with tab2:
        st.subheader("Experiment Management")
        
        # Start New Experiment
        with st.expander("Start New Experiment", expanded=False):
            exp_name = st.text_input("Experiment Name")
            exp_description = st.text_area("Description", height=80)
            exp_params = st.text_area("Parameters (JSON format)", height=60, placeholder='{"param1": "value1", "param2": "value2"}')
            
            if st.button("Start Experiment", use_container_width=True):
                if exp_name.strip() and exp_description.strip():
                    try:
                        import json
                        params = json.loads(exp_params) if exp_params.strip() else {}
                        bot = get_bot(strategy, st.session_state.llm_bind)
                        exp_id = bot.start_experiment(exp_name, exp_description, params)
                        st.session_state.current_experiment = exp_id
                        st.success(f"Experiment started: {exp_name} (ID: {exp_id})")
                        st.rerun()
                    except json.JSONDecodeError:
                        st.error("Invalid JSON format for parameters")
                else:
                    st.error("Please enter experiment name and description")
        
        # Current Experiment Status
        if st.session_state.current_experiment:
            st.info(f"🧪 Active Experiment: {st.session_state.current_experiment}")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("End Experiment (Completed)", use_container_width=True):
                    bot = get_bot(strategy, st.session_state.llm_bind)
                    bot.end_experiment("completed")
                    st.session_state.current_experiment = None
                    st.success("Experiment marked as completed")
                    st.rerun()
            with col2:
                if st.button("End Experiment (Failed)", use_container_width=True):
                    bot = get_bot(strategy, st.session_state.llm_bind)
                    bot.end_experiment("failed")
                    st.session_state.current_experiment = None
                    st.warning("Experiment marked as failed")
                    st.rerun()
        
        # View All Experiments
        with st.expander("View All Experiments", expanded=False):
            bot = get_bot(strategy, st.session_state.llm_bind)
            experiments = bot.experiment_manager.list_experiments()
            
            if experiments:
                for exp in experiments:
                    status_icon = "✅" if exp.status == "completed" else "🔄" if exp.status == "active" else "❌"
                    st.write(f"{status_icon} **{exp.name}** ({exp.id})")
                    st.caption(f"Status: {exp.status} | Created: {exp.created_at}")
                    st.caption(f"Description: {exp.description}")
                    if exp.results:
                        with st.expander(f"Results ({len(exp.results)} metrics)", expanded=False):
                            st.json(exp.results)
                    st.divider()
            else:
                st.info("No experiments found.")
    
    with tab3:
        st.subheader("Chat History")
        bot = get_bot(strategy, st.session_state.llm_bind)
        history = bot.chat_history
        
        if history:
            st.write(f"Total messages: {len(history)}")
            
            # Clear history button
            col1, col2 = st.columns([3, 1])
            with col2:
                if st.button("Clear History", use_container_width=True):
                    bot.clear_history()
                    st.success("Chat history cleared")
                    st.rerun()
            
            # Display history
            for i, msg in enumerate(reversed(history[-20:])):  # Show last 20 messages
                role_icon = "👤" if msg.role == "user" else "🤖"
                st.write(f"{role_icon} **{msg.role.title()}** - {msg.timestamp}")
                st.write(msg.content)
                st.divider()
        else:
            st.info("No chat history yet.")
    
    with tab4:
        st.subheader("View Logs")
        bot = get_bot(strategy, st.session_state.llm_bind)
        
        # Filter options
        col1, col2 = st.columns(2)
        with col1:
            filter_category = st.selectbox("Filter by Category", ["all", "general", "system", "chat", "query", "experiment", "manual"])
        with col2:
            filter_level = st.selectbox("Filter by Level", ["all", "INFO", "WARNING", "ERROR", "DEBUG"])
        
        # Get filtered logs
        logs = bot.log_manager.get_logs()
        if filter_category != "all":
            logs = [log for log in logs if log.category == filter_category]
        if filter_level != "all":
            logs = [log for log in logs if log.level == filter_level]
        
        if logs:
            st.write(f"Showing {len(logs)} logs")
            
            # Clear logs button
            if st.button("Clear All Logs", use_container_width=True):
                bot.log_manager.clear_logs()
                st.success("All logs cleared")
                st.rerun()
            
            # Display logs
            for log in reversed(logs[-50:]):  # Show last 50 logs
                level_color = {
                    "INFO": "🔵",
                    "WARNING": "🟡", 
                    "ERROR": "🔴",
                    "DEBUG": "⚪"
                }.get(log.level, "⚪")
                
                st.write(f"{level_color} **{log.level}** [{log.category}] - {log.timestamp}")
                st.write(log.message)
                if log.experiment_id:
                    st.caption(f"Experiment: {log.experiment_id}")
                st.divider()
        else:
            st.info("No logs found.")

with st.expander("Advanced (experiments)", expanded=False):
    cq = st.text_input("Test query for chunking compare", value="What is ABFA allocation in 2024?")
    fq = st.text_input("Failure-case query", value="What is ABFA allocation trend?")
    ac1, ac2, ac3 = st.columns(3)
    with ac1:
        if st.button("Chunking compare"):
            b = RAGChatbot(docs_path="docs", strategy="fixed", top_k=top_k)
            st.dataframe(b.compare_chunking(cq), use_container_width=True)
    with ac2:
        if st.button("Failure compare"):
            b = get_bot(strategy, st.session_state.llm_bind)
            st.json(b.compare_failure_case(fq, top_k=top_k))
    with ac3:
        if st.button("Adversarial suite"):
            b = get_bot(strategy, st.session_state.llm_bind)
            r = b.run_adversarial_suite()
            st.json(r["metrics"])
            st.json(r["runs"])
