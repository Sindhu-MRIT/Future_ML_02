import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import re
import nltk
import warnings
warnings.filterwarnings('ignore')

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

nltk.download('stopwords', quiet=True)

st.set_page_config(page_title="Ticket Classifier", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=Playfair+Display:wght@700&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif}
.stApp{background:#f0f4f8}
div[data-testid="stSidebar"]{background:#0f172a}
div[data-testid="stSidebar"] *{color:#94a3b8 !important}
div[data-testid="stSidebar"] hr{border-color:#1e293b;margin:14px 0}
div[data-testid="stSidebar"] .stTextArea textarea{
    background:#1e293b !important;color:#94a3b8 !important;
    border:1px solid #293548 !important;font-size:13px !important;border-radius:6px !important}
div[data-testid="stSidebar"] .stButton button{
    background:#3b82f6 !important;color:#fff !important;border:none !important;
    border-radius:6px !important;font-weight:600 !important;width:100% !important}
.headline{font-family:'Playfair Display',serif;font-size:48px;font-weight:700;color:#0f172a;line-height:1;margin-bottom:6px}
.subline{font-size:14px;color:#64748b;font-weight:300;margin-bottom:20px}
.hint-box{background:#dbeafe;border-radius:6px;padding:14px 18px;font-size:13px;color:#1e40af;line-height:1.7;margin-bottom:24px}
.stat-card{background:#fff;border-radius:8px;padding:20px 18px;height:100%}
.stat-top{height:3px;border-radius:3px;margin-bottom:14px}
.stat-label{font-size:10px;letter-spacing:2px;text-transform:uppercase;color:#94a3b8;margin-bottom:6px}
.stat-num{font-size:28px;font-weight:600;color:#0f172a;line-height:1}
.stat-desc{font-size:12px;color:#94a3b8;margin-top:8px;line-height:1.6}
.box{background:#fff;border-radius:8px;padding:22px}
.box-title{font-size:15px;font-weight:600;color:#0f172a;margin-bottom:3px}
.box-sub{font-size:12px;color:#94a3b8;margin-bottom:16px}
.result-wrap{background:#f8fafc;border-radius:6px;padding:16px;border:1px solid #e2e8f0;margin-bottom:14px}
.ticket-text{font-size:12px;color:#475569;line-height:1.7;margin-bottom:12px;font-style:italic}
.badge{display:inline-block;padding:4px 14px;border-radius:99px;font-size:11px;font-weight:600;margin-right:6px}
.badge-cat{background:#dbeafe;color:#1e40af}
.badge-high{background:#fee2e2;color:#991b1b}
.badge-med{background:#fef3c7;color:#92400e}
.badge-low{background:#d1fae5;color:#065f46}
.conf-track{height:6px;background:#e2e8f0;border-radius:3px;margin:8px 0 4px}
.conf-fill{height:100%;border-radius:3px;background:#3b82f6}
.prob-row{display:flex;align-items:center;gap:8px;font-size:11px;color:#475569;margin-bottom:5px}
.prob-bar-wrap{flex:1;height:5px;background:#f1f5f9;border-radius:3px}
.prob-bar{height:100%;border-radius:3px}
.section-label{font-size:10px;letter-spacing:2px;text-transform:uppercase;color:#3b82f6;margin-bottom:6px}
.section-title{font-family:'Playfair Display',serif;font-size:22px;color:#0f172a;margin-bottom:14px}
.action-card{background:#0f172a;border-radius:8px;padding:20px;height:100%}
.ac-num{font-size:9px;letter-spacing:2px;text-transform:uppercase;color:#334155;margin-bottom:8px}
.ac-title{font-size:14px;font-weight:600;color:#f8fafc;margin-bottom:8px}
.ac-body{font-size:13px;color:#475569;line-height:1.8}
.ac-hi{color:#3b82f6;font-weight:500}
</style>
""", unsafe_allow_html=True)

STOP_WORDS = set(stopwords.words('english'))

PRIORITY_MAP = {
    'Hardware': 'High', 'Access': 'High', 'Administrative rights': 'High',
    'Storage': 'Medium', 'Purchase': 'Medium', 'Internal Project': 'Medium',
    'HR Support': 'Low', 'Miscellaneous': 'Low'
}

CAT_COLORS = {
    'Hardware': '#3b82f6', 'HR Support': '#3b82f6',
    'Access': '#60a5fa', 'Miscellaneous': '#94a3b8',
    'Storage': '#94a3b8', 'Purchase': '#94a3b8',
    'Internal Project': '#cbd5e1', 'Administrative rights': '#cbd5e1'
}

CHART = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='#ffffff',
    font=dict(color='#94a3b8', family='DM Sans', size=12),
    margin=dict(l=0, r=0, t=10, b=0)
)
GRID = dict(gridcolor='#f1f5f9', zeroline=False, showline=True, linecolor='#e2e8f0')


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in STOP_WORDS and len(w) > 2]
    return ' '.join(words)


@st.cache_resource
def load_and_train():
    df = pd.read_csv('all_tickets_processed_improved_v3.csv')
    df = df.dropna(subset=['Document', 'Topic_group'])
    df['clean'] = df['Document'].apply(clean_text)
    df['Priority'] = df['Topic_group'].map(PRIORITY_MAP)

    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X = vec.fit_transform(df['clean'])
    y_cat = df['Topic_group']
    y_pri = df['Priority']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cat, test_size=0.2, random_state=42, stratify=y_cat)
    cat_model = LogisticRegression(max_iter=1000, random_state=42)
    cat_model.fit(X_train, y_train)
    cat_acc = accuracy_score(y_test, cat_model.predict(X_test))
    cm = confusion_matrix(y_test, cat_model.predict(X_test), labels=cat_model.classes_)

    Xp_tr, Xp_te, yp_tr, yp_te = train_test_split(
        X, y_pri, test_size=0.2, random_state=42, stratify=y_pri)
    pri_model = LogisticRegression(max_iter=1000, random_state=42)
    pri_model.fit(Xp_tr, yp_tr)
    pri_acc = accuracy_score(yp_te, pri_model.predict(Xp_te))

    return df, vec, cat_model, pri_model, cat_acc, pri_acc, cm


with st.spinner("Training model on 47,000 tickets..."):
    df, vec, cat_model, pri_model, cat_acc, pri_acc, cm = load_and_train()

cat_counts = df['Topic_group'].value_counts()
pri_counts = df['Priority'].value_counts()


with st.sidebar:
    st.markdown("""
    <div style='padding:4px 0 18px'>
        <div style='font-family:Playfair Display,serif;font-size:18px;color:#3b82f6'>Ticket Classifier</div>
        <div style='font-size:10px;letter-spacing:2px;text-transform:uppercase;color:#3b82f6;margin-top:4px;font-weight:700'>Support Operations</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='font-size:10px;letter-spacing:2px;text-transform:uppercase;color:#3b82f6;margin-bottom:8px;font-weight:700'>Try it live</div>", unsafe_allow_html=True)
    ticket_input = st.text_area(
        "Ticket text",
        value="My laptop screen stopped working and I cannot access any of my files or do any work.",
        height=110,
        label_visibility="collapsed"
    )
    classify_btn = st.button("Classify this ticket")

    st.markdown("---")
    st.markdown("<div style='font-size:10px;letter-spacing:2px;text-transform:uppercase;color:#3b82f6;margin-bottom:10px;font-weight:700'>Page guide</div>", unsafe_allow_html=True)
    sections = [
        ("4 key numbers", "Quick snapshot of the system"),
        ("Live result", "What the model decided"),
        ("Probability bars", "Confidence across categories"),
        ("Category chart", "Ticket volume breakdown"),
        ("Confusion matrix", "Where the model makes mistakes"),
        ("What this fixes", "Business impact"),
    ]
    for title, desc in sections:
        st.markdown(f"""
        <div style='border-left:2px solid #3b82f6;padding:5px 10px;margin-bottom:7px'>
            <div style='font-size:12px;color:#bc9f7e;font-weight:1000'>{title}</div>
            <div style='font-size:11px;color:#3b82f6;margin-top:1px;line-height:1.5'>{desc}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:11px;color:#3b82f6;line-height:2.2;font-weight:700'>
        Tickets — 47,000<br>
        Categories — 8<br>
        Priority levels — 3<br>
        Model — Logistic Regression<br>
        Source — Kaggle
    </div>
    """, unsafe_allow_html=True)


category, priority, confidence, all_probs = None, None, None, None
if classify_btn and ticket_input.strip():
    cleaned = clean_text(ticket_input)
    vec_input = vec.transform([cleaned])
    category = cat_model.predict(vec_input)[0]
    priority = pri_model.predict(vec_input)[0]
    probs = cat_model.predict_proba(vec_input)[0]
    confidence = probs.max() * 100
    all_probs = sorted(zip(cat_model.classes_, probs), key=lambda x: x[1], reverse=True)


st.markdown('<div class="headline">Which ticket needs attention first?</div>', unsafe_allow_html=True)
st.markdown('<div class="subline"><strong>A machine learning system that reads support tickets and instantly sorts them by category and urgency — trained on 47,000 real IT support tickets.</strong></div>', unsafe_allow_html=True)
st.markdown("""
<div class="hint-box">
    Type any support ticket into the box on the left and hit classify. The system assigns a category,
    flags the priority level and shows how confident it is. Scroll down to see how it performs
    across all 47,000 tickets in the dataset.
</div>
""", unsafe_allow_html=True)


s1, s2, s3, s4 = st.columns(4, gap="small")
stats = [
    ("#3b82f6", "Tickets trained on", f"{len(df):,}", "Real IT support tickets the model learned from."),
    ("#10b981", "Categories", str(df['Topic_group'].nunique()), "Hardware, Access, HR, Storage and 4 more."),
    ("#f59e0b", "Category accuracy", f"{cat_acc*100:.0f}%", "How often the model picks the right category."),
    ("#ef4444", "Priority accuracy", f"{pri_acc*100:.0f}%", "How often the model assigns the right priority."),
]
for col, (color, label, num, desc) in zip([s1, s2, s3, s4], stats):
    with col:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-top" style="background:{color}"></div>
            <div class="stat-label"><strong>{label}</strong></div>
            <div class="stat-num">{num}</div>
            <div class="stat-desc">{desc}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

left, right = st.columns([1.1, 0.9], gap="small")

with left:
    st.markdown('<div class="section-label"><span style="font-weight:1000">Live classifier</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">What the model decided</div>', unsafe_allow_html=True)

    if category:
        p_badge = f'badge-{"high" if priority=="High" else "med" if priority=="Medium" else "low"}'
        conf_pct = f"{confidence:.0f}%"
        st.markdown(f"""
        <div class="result-wrap">
            <div class="ticket-text">"{ticket_input[:120]}..."</div>
            <span class="badge badge-cat">{category}</span>
            <span class="badge {p_badge}">{priority} priority</span>
            <div style='margin-top:14px;font-size:11px;color:#94a3b8;font-weight:700'>Confidence</div>
            <div class="conf-track"><div class="conf-fill" style="width:{conf_pct}"></div></div>
            <div style='font-size:13px;color:#1e40af;font-weight:600;margin-bottom:10px'>{conf_pct}</div>
            <div style='font-size:12px;color:#64748b;line-height:1.7'>
                {category} tickets are flagged {priority} priority because
                {"they directly stop employees from working." if priority=="High" else
                 "they affect workflows but are not immediately blocking." if priority=="Medium" else
                 "they are informational and less time-sensitive."}
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<div style='font-size:13px;font-weight:600;color:#0f172a;margin-bottom:10px'>Probability across all categories</div>", unsafe_allow_html=True)
        for cat, prob in all_probs:
            pct = prob * 100
            color = '#3b82f6' if cat == category else '#e2e8f0'
            st.markdown(f"""
            <div class="prob-row">
                <span style='width:150px;color:{"#0f172a" if cat==category else "#94a3b8"};
                font-weight:{"600" if cat==category else "400"}'>{cat}</span>
                <div class="prob-bar-wrap">
                    <div class="prob-bar" style="width:{pct:.1f}%;background:{color}"></div>
                </div>
                <span style='color:{"#1e40af" if cat==category else "#94a3b8"};
                font-weight:{"600" if cat==category else "400"}'>{pct:.1f}%</span>
            </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="result-wrap" style='text-align:center;padding:40px 20px'>
            <div style='font-size:14px;color:#94a3b8;margin-bottom:8px'><span style='font-weight:1000'>No ticket classified yet</span></div>
            <div style='font-size:12px;color:#cbd5e1;line-height:1.7'><span style='font-weight:700'>Type a support ticket in the left sidebar and hit the button to see the classifier in action.</span></div>
            </div>
        </div>""", unsafe_allow_html=True)

with right:
    st.markdown('<div class="section-label">Ticket volume</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Category breakdown</div>', unsafe_allow_html=True)
    st.markdown("<div style='font-size:13px;color:#bc9f7e;margin-bottom:16px;font-weight:500'>How all 47,000 tickets split across the 8 categories the model was trained on.</div>", unsafe_allow_html=True)

    sorted_cats = cat_counts.sort_values(ascending=True)
    bar_colors = [CAT_COLORS.get(c, '#94a3b8') for c in sorted_cats.index]
    fig = go.Figure(go.Bar(
        x=sorted_cats.values, y=sorted_cats.index,
        orientation='h', marker=dict(color=bar_colors),
        hovertemplate='%{y} — %{x:,} tickets<extra></extra>'
    ))
    fig.update_layout(**CHART,
        xaxis=dict(**GRID, tickformat=',d', tickcolor='#bc9f7e', tickfont=dict(color='#bc9f7e')),
        yaxis=dict(gridcolor='rgba(0,0,0,0)', zeroline=False, tickcolor='#bc9f7e', tickfont=dict(color='#bc9f7e')), height=280)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div style='font-size:13px;font-weight:600;color:#0f172a;margin:16px 0 10px'>Priority split across all tickets</div>", unsafe_allow_html=True)
    p_colors = {'High': '#ef4444', 'Medium': '#f59e0b', 'Low': '#10b981'}
    fig2 = go.Figure(go.Bar(
        x=list(pri_counts.index), y=list(pri_counts.values),
        marker=dict(color=[p_colors.get(p, '#94a3b8') for p in pri_counts.index]),
        hovertemplate='%{x} — %{y:,} tickets<extra></extra>'
    ))
    fig2.update_layout(**CHART,
        xaxis=dict(**GRID, tickcolor='#bc9f7e', tickfont=dict(color='#bc9f7e')), yaxis=dict(**GRID, tickformat=',d', tickcolor='#bc9f7e', tickfont=dict(color='#bc9f7e')), height=200)
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

st.markdown('<div class="section-label"><span style="font-weight:1000">Model accuracy</span></div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Where the model gets it right — and where it slips</div>', unsafe_allow_html=True)
st.markdown("<div style='font-size:13px;color:#bc9f7e;margin-bottom:16px;font-weight:500'>Each row is the real category. Each column is what the model predicted. Darker diagonal means more accurate.</div>", unsafe_allow_html=True)

fig3 = go.Figure(go.Heatmap(
    z=cm, x=list(cat_model.classes_), y=list(cat_model.classes_),
    colorscale=[[0, '#f0f4f8'], [0.5, '#93c5fd'], [1, '#1d4ed8']],
    hovertemplate='Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>',
    text=cm, texttemplate='%{text}', textfont=dict(size=11, color='#0f172a')
))
fig3.update_layout(**CHART,
    xaxis=dict(title='Predicted', tickangle=-30, showgrid=False, tickcolor='#bc9f7e', tickfont=dict(color='#bc9f7e')),
    yaxis=dict(title='Actual', showgrid=False, autorange='reversed', tickcolor='#bc9f7e', tickfont=dict(color='#bc9f7e')), height=420)
st.plotly_chart(fig3, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

st.markdown('<div class="section-label"><span style="font-weight:1000">Why this matters</span></div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">What this fixes for a support team</div>', unsafe_allow_html=True)

a1, a2, a3 = st.columns(3, gap="small")
actions = [
    ("Benefit 01", "No more manual sorting",
     "Every ticket is classified the moment it arrives. <span class='ac-hi'>Hardware and Access</span> issues go straight to the high priority queue without anyone reading them first."),
    ("Benefit 02", "Urgent tickets never get buried",
     "The model flags <span class='ac-hi'>High priority</span> tickets immediately so agents see them first — not after going through 50 low priority password resets."),
    ("Benefit 03", "Agents focus on solving, not reading",
     "With category and priority pre-assigned, agents spend their time <span class='ac-hi'>resolving issues</span> rather than figuring out what the ticket is even about."),
]
for col, (num, title, body) in zip([a1, a2, a3], actions):
    with col:
        st.markdown(f"""
        <div class="action-card">
            <div class="ac-num" style="color:#bc9f7e;font-weight:bold">{num}</div>
            <div class="ac-title">{title}</div>
            <div class="ac-body">{body}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("""
<div style='text-align:center;color:#94a3b8;font-size:11px;letter-spacing:1.5px;
            text-transform:uppercase;border-top:1px solid #e2e8f0;
            padding-top:20px;margin-top:40px'>
    Future Interns ML Internship &nbsp;·&nbsp;
    IT Support Ticket Dataset, Kaggle &nbsp;·&nbsp; Logistic Regression
</div>
""", unsafe_allow_html=True)