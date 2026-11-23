"""HelmetGuard | A professional image-first YOLO11 motorcycle safety workstation.

Launch from the project root: python -m streamlit run app.py
A *custom-trained* models/best.pt is required; generic YOLO11 is not a helmet detector.
"""
from __future__ import annotations

from datetime import datetime
from io import BytesIO
import json
from pathlib import Path
import os
import uuid
from zipfile import ZipFile, ZIP_DEFLATED

import pandas as pd
import streamlit as st

from src.detector import analyze, model_path, normalize_class, png_bytes, prepare_image
from src.storage import clear_scans, load_scans, save_scan
from src.ui_helpers import (filter_history, model_has_helmet_classes, overlay_images,
                            recent_daily_counts, summarize_history)
from src.video_pipeline import process_uploaded_video

ROOT = Path(__file__).resolve().parent
# Keep Roboflow/EdgeVision's class IDs intact; the last class is a generic helmet.
ENHANCED_CLASSES = [
    'Full-faced', 'Half-faced', 'Invalid', 'motorcycle-rider',
    'not wearing helmet', 'Helmet',
]
ORIGINAL_CLASSES = ENHANCED_CLASSES[:5]
MODEL_TARGET = ROOT / 'models' / 'best.pt'
MAX_CHECKPOINT_MB = 140
st.set_page_config(
    page_title='HelmetGuard | Motorcycle Helmet Detection • LSPU-LB',
    page_icon='🛡️', layout='wide', initial_sidebar_state='expanded',
)

# LSPU-inspired portal styles are included directly in app.py.
# No separate assets/portal.css file is required.
PORTAL_CSS = r'''
/* HelmetGuard academic portal — inspired by LSPU's blue campus identity.
   Independent student project, not an official LSPU portal. */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Plus+Jakarta+Sans:wght@500;600;700;800&display=swap');
:root {
  --portal-navy:#083461;
  --portal-navy-deep:#08294d;
  --portal-blue:#145b9b;
  --portal-bright:#1974bd;
  --portal-cyan:#21a4d1;
  --portal-teal:#0c877c;
  --portal-bg:#f3f6fa;
  --portal-card:#ffffff;
  --portal-border:#dfe7f1;
  --portal-muted:#64758c;
  --portal-text:#1a3049;
}
html,body,[data-testid="stApp"], [data-testid="stSidebar"] {font-family:'DM Sans',sans-serif!important}
[data-testid="stAppViewContainer"]{background:#f4f7fb!important;color:#1a3049!important}
[data-testid="stAppViewContainer"] > .main{background:radial-gradient(ellipse at 95% 0%, #eaf4ff 0%, #f4f7fb 45%, #f4f7fb 75%)}
[data-testid="stHeader"]{background:transparent!important}
[data-testid="stToolbar"]{background:#fff;border-radius:8px}
.block-container{max-width:1500px;padding:1.1rem 2.25rem 4.0rem;}
h1,h2,h3,h4{font-family:'Plus Jakarta Sans','DM Sans',sans-serif!important;letter-spacing:-.035em!important;color:#173654}
h1{font-size:2.15rem!important;font-weight:800!important;margin:0 0 .35rem!important;line-height:1.25!important}
h2{font-size:1.35rem!important;font-weight:800!important}
h3{font-size:1.06rem!important;font-weight:800!important}
p{line-height:1.65}
/* Sidebar */
[data-testid="stSidebar"]{background:linear-gradient(185deg,#0b427b 0%,#0a3569 47%,#08294e 100%)!important;border-right:0!important;}
[data-testid="stSidebar"]>div:first-child{background:transparent!important}
[data-testid="stSidebar"] [data-testid="stSidebarContent"]{padding-top:1.2rem!important}
[data-testid="stSidebar"] p,[data-testid="stSidebar"] label,[data-testid="stSidebar"] span,[data-testid="stSidebar"] [data-testid="stCaptionContainer"]{color:#e8f3ff!important}
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p{font-size:10px;letter-spacing:.13em;text-transform:uppercase;font-weight:800!important;color:#a7c9eb!important}
[data-testid="stSidebar"] [data-testid="stRadio"] label{border-radius:11px;padding:9px 12px;margin:4px 0;transition:background .15s;}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover{background:#ffffff17}
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked){background:#ffffff24;box-shadow:inset 3px 0 0 #7cdbf6}
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) p{font-weight:800!important;color:#fff!important}
[data-testid="stSidebar"] hr{border-color:#ffffff2a!important;margin:1.15rem 0!important}
[data-testid="stSidebar"] [data-testid="stToggle"] label{color:#eaf5ff!important}
[data-testid="stSidebar"] [data-testid="stSlider"] [data-baseweb="slider"]{margin-top:8px}
[data-testid="stSidebar"] button{border-radius:9px!important}
.portal-sidebar-logo{display:flex;align-items:center;gap:12px;margin-bottom:8px}
.portal-monogram{width:45px;height:45px;border-radius:12px;background:linear-gradient(145deg,#fff,#d9f2ff);display:flex;align-items:center;justify-content:center;color:#0b5c9c;font-family:'Plus Jakarta Sans',sans-serif;font-size:19px;font-weight:800;box-shadow:0 5px 15px #001b3c44}
.portal-sidebar-brand{font-family:'Plus Jakarta Sans',sans-serif;font-size:19px;color:white;letter-spacing:-.055em;font-weight:800;line-height:1.12}
.portal-sidebar-brand small{font-family:'DM Sans',sans-serif;letter-spacing:.12em;color:#b9ddf8;font-size:9px;font-weight:800;display:block;margin-top:6px}
.portal-schoolbox{font-size:11px;line-height:1.5;color:#cae7ff;border:1px solid #ffffff29;background:#ffffff0c;border-radius:12px;padding:12px;margin:15px 0 11px}
.portal-schoolbox strong{color:#fff;font-weight:800}
.portal-sidebar-status{padding:10px 12px;border-radius:10px;background:#ffffff10;border:1px solid #ffffff25;font-size:11px;font-weight:700;color:#e9f5ff;display:flex;align-items:center;gap:9px;margin-bottom:5px}
.portal-dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:#f5bd62;box-shadow:0 0 0 4px #f5bd6224}
.portal-dot.ok{background:#69e7c9;box-shadow:0 0 0 4px #69e7c924}
.portal-sidebar-foot{font-size:11px;line-height:1.6;color:#c5ddf1;border-top:1px solid #ffffff28;padding-top:16px;margin-top:14px}
/* Header bar and typography */
.portal-topbar{display:flex;align-items:center;justify-content:space-between;gap:16px;padding:10px 17px;margin:2px 0 27px;background:#fff;border:1px solid var(--portal-border);border-radius:13px;box-shadow:0 1px 4px #16355108}
.portal-crumb{display:flex;align-items:center;gap:11px;flex-wrap:wrap;font-size:12px;color:#64809b;font-weight:700}
.portal-crumb .portal-root{color:#0d558d;font-weight:800}
.portal-header-actions{display:flex;gap:10px;align-items:center}
.portal-top-tag{padding:7px 10px;background:#f1f6fc;border:1px solid #e2ebf5;border-radius:9px;color:#326285;font-weight:800;font-size:11px;white-space:nowrap}
.portal-top-tag.ready{background:#eaf8f3;border-color:#cceee4;color:#187964}
.portal-eyebrow{display:inline-block;color:#1467a9;font-size:11px;letter-spacing:.14em;text-transform:uppercase;font-weight:800;margin-bottom:7px}
.portal-subtitle{color:#677a91;font-size:14px;line-height:1.65;max-width:800px;margin:3px 0 25px}
.portal-hero{position:relative;overflow:hidden;background:linear-gradient(116deg,#0c467f 0%,#1365a7 61%,#1c8bbb 100%);border-radius:18px;margin:6px 0 26px;min-height:174px;display:flex;justify-content:space-between;align-items:center;padding:29px 34px;color:white;box-shadow:0 11px 24px #0d457b23}
.portal-hero:before{content:"";position:absolute;border:1px solid #ffffff26;border-radius:50%;width:340px;height:340px;right:-75px;top:-110px}
.portal-hero:after{content:"";position:absolute;border:1px solid #ffffff18;border-radius:50%;width:260px;height:260px;right:38px;top:35px}
.portal-hero-main{z-index:1;max-width:760px}
.portal-hero-kicker{display:inline-block;border:1px solid #ffffff48;background:#ffffff19;border-radius:30px;padding:5px 11px;color:#d9f5ff;font-size:10px;font-weight:800;letter-spacing:.12em;margin-bottom:13px}
.portal-hero h2{color:white!important;font-size:1.7rem!important;letter-spacing:-.035em;margin:0 0 10px!important;line-height:1.22}
.portal-hero p{color:#dcedff!important;margin:0;max-width:690px;font-size:13px;line-height:1.7}
.portal-hero-glyph{position:relative;z-index:1;min-width:113px;height:113px;border-radius:25px;background:#ffffff20;border:1px solid #ffffff4a;display:flex;align-items:center;justify-content:center;font-size:55px;transform:rotate(-9deg);margin-right:44px}
.portal-pill{display:inline-flex;align-items:center;gap:5px;font-size:11px;font-weight:800;padding:6px 11px;border-radius:30px;background:#e9f8f2;color:#0c725f;border:1px solid #bee9dc}
.portal-pill.red{background:#fff0f1;border-color:#f6cfd7;color:#a83c55}
.portal-pill.neutral{background:#edf2f7;border-color:#dae6f1;color:#657b94}
.portal-guide{display:flex;flex-wrap:wrap;gap:8px;margin:5px 0 24px}
.portal-guide span{background:#fff;border:1px solid #e0e8f1;border-radius:9px;color:#506e89;padding:10px 13px;font-size:11px;font-weight:800}
.portal-guide span b{color:#0e64a8;margin-right:8px}
.portal-mini-section{color:#1d3855;font-size:12px;text-transform:uppercase;letter-spacing:.12em;font-weight:800;border-bottom:1px solid #e1e8f0;padding-bottom:12px;margin:23px 0 15px}
.portal-empty{background:#fff;border:1px dashed #bcd0e3;border-radius:15px;padding:34px;text-align:center;color:#687d92}
.portal-empty strong{display:block;color:#264b70;margin-bottom:8px}
.portal-note{color:#647891;font-size:12px;line-height:1.7}
.portal-footer{text-align:center;color:#8391a3;font-size:11px;border-top:1px solid #dde5ef;margin-top:35px;padding-top:18px}
.portal-feature{height:100%;background:#fff;border:1px solid #e0e9f2;border-radius:13px;padding:18px 20px;box-shadow:0 3px 12px #0d345008}
.portal-feature-icon{width:36px;height:36px;display:flex;align-items:center;justify-content:center;background:#ecf6fe;border-radius:9px;color:#1171b8;font-size:19px;margin-bottom:11px}
.portal-feature b{font-size:13px;color:#234363}
.portal-feature p{font-size:12px;color:#6f8196;margin-top:7px;line-height:1.65;margin-bottom:0}
/* Native widgets and data tables */
[data-testid="stMetric"]{background:#fff!important;border:1px solid #e0e8f2!important;border-left:4px solid #1c7db5!important;border-radius:13px!important;padding:18px 20px!important;box-shadow:0 5px 16px #153c5c08}
[data-testid="stMetricLabel"]{color:#698098!important;font-size:12px!important;font-weight:700!important}
[data-testid="stMetricValue"]{font-family:'Plus Jakarta Sans',sans-serif!important;font-size:1.7rem!important;font-weight:800!important;letter-spacing:-.04em!important;color:#183a60!important}
[data-testid="stMetricDelta"]{color:#238466}
[data-testid="stFileUploader"]{background:#f9fcff;border:1.5px dashed #a9c6de!important;border-radius:13px;padding:14px!important}
[data-testid="stFileUploader"]:hover{border-color:#1778bd!important;background:#f2f9ff}
[data-testid="stImage"] img{border:1px solid #e0e8f1;border-radius:12px;box-shadow:0 3px 12px #102c450c}
[data-testid="stVerticalBlockBorderWrapper"] > div{border-radius:14px!important;border-color:#e0e9f2!important;background:#fff}
[data-testid="stVerticalBlockBorderWrapper"]{box-shadow:0 2px 9px #10375706}
.stButton>button,.stDownloadButton>button{min-height:42px;border-radius:9px;font-weight:800!important;border:1px solid #d2e0ee;color:#285b83;background:#fff}
.stButton>button:hover,.stDownloadButton>button:hover{border-color:#1880c4;color:#075a96;background:#f2f8ff}
.stButton>button[kind="primary"],.stDownloadButton>button[kind="primary"]{background:#1466a7!important;border:1px solid #1466a7!important;color:white!important;box-shadow:0 4px 12px #1466a72d}
.stButton>button[kind="primary"]:hover,.stDownloadButton>button[kind="primary"]:hover{background:#0e5289!important;border-color:#0e5289!important}
[data-testid="stAlert"]{border-radius:11px!important}
[data-testid="stDataFrame"]{border-radius:12px;overflow:hidden;border:1px solid #e3ebf2}
[data-testid="stTabs"] [data-baseweb="tab-list"]{gap:7px;border-bottom:1px solid #e0e8f2}
[data-testid="stTabs"] [data-baseweb="tab"]{color:#658099;font-weight:750;border-radius:8px 8px 0 0}
[data-testid="stTabs"] [aria-selected="true"]{color:#0e6daf!important;background:#eaf4fd!important}
[data-testid="stExpander"]{border-radius:11px;overflow:hidden}
hr{border-color:#e1e8f0!important}
[data-testid="stCaptionContainer"]{color:#74869c!important}
@media(max-width:980px){.block-container{padding:1rem 1.25rem 3rem}.portal-hero{padding:24px;min-height:150px}.portal-hero-glyph{margin-right:0}}
@media(max-width:640px){.block-container{padding:.65rem .8rem 2rem}.portal-topbar{flex-wrap:wrap}.portal-top-tag{font-size:10px}.portal-hero{padding:19px 18px;min-height:0}.portal-hero-glyph{display:none}.portal-hero h2{font-size:1.35rem!important}h1{font-size:1.7rem!important}}

'''
PORTAL_CSS += r'''
/* Repair: light-portal text contrast, dark-code contrast, and native uploader controls. */
[data-testid="stAppViewContainer"] { color:#203950!important; }
[data-testid="stAppViewContainer"] h1,
[data-testid="stAppViewContainer"] h2,
[data-testid="stAppViewContainer"] h3,
[data-testid="stAppViewContainer"] h4 { letter-spacing:0!important;word-spacing:.08em!important; }
[data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"]>p,
[data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] li { color:#2a425b; }
[data-testid="stAppViewContainer"] [data-testid="stWidgetLabel"] p,
[data-testid="stAppViewContainer"] [data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stAppViewContainer"] [data-testid="stFileUploaderDropzoneInstructions"] small { color:#274762!important; }
[data-testid="stAppViewContainer"] [data-testid="stFileUploaderDropzone"] { background:#f8fbff!important;border:1.5px dashed #97b9d9!important;border-radius:12px; }
[data-testid="stAppViewContainer"] [data-testid="stFileUploaderFile"] { background:#eaf3fc!important;border-radius:8px!important; }
[data-testid="stAppViewContainer"] [data-testid="stFileUploaderFile"] * { color:#173a58!important; }
[data-testid="stAppViewContainer"] [data-testid="stFileUploader"] button { border:1px solid #b9cee3!important;color:#175b92!important;background:#fff!important; }
[data-testid="stAppViewContainer"] [data-testid="stCodeBlock"] { border:1px solid #28425e!important;background:#102235!important;border-radius:12px!important; }
[data-testid="stAppViewContainer"] [data-testid="stCodeBlock"] pre,
[data-testid="stAppViewContainer"] [data-testid="stCodeBlock"] pre code,
[data-testid="stAppViewContainer"] [data-testid="stCodeBlock"] pre span { color:#eaf3fd!important; }
[data-testid="stAppViewContainer"] [data-testid="stExpander"] summary {color:#244766!important;}
[data-testid="stAppViewContainer"] [data-testid="stDataFrame"] {background:#fff!important;}
[data-testid="stAppViewContainer"] [data-testid="stCaptionContainer"] p {color:#566e86!important;}
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {color:#dceeff!important;}
/* Readability for chart labels when a browser uses dark-mode system defaults. */
[data-testid="stVegaLiteChart"] { background:#fff;border-radius:14px;padding:8px; }
'''

# Contrast fixes for Streamlit widgets when the host theme remains dark.
# Scoped to the main panel; the navy sidebar and original portal layout stay intact.
PORTAL_CSS += r"""
/* Accessible portal content and native controls */
[data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] { color:#173753; }
[data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] > p,
[data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] li,
[data-testid="stAppViewContainer"] [data-testid="stWidgetLabel"] p { color:#244460 !important; }

/* Alerts: Streamlit can inherit a dark theme even when the page is light. */
[data-testid="stAppViewContainer"] [data-testid="stAlert"],
[data-testid="stAppViewContainer"] [data-baseweb="notification"] {
  background:#e9f4ff !important;
  color:#153653 !important;
  border:1px solid #bfd8f2 !important;
  border-radius:11px !important;
  box-shadow:none !important;
}
[data-testid="stAppViewContainer"] [data-testid="stAlert"] :is(p,span,strong,em,li,a,code),
[data-testid="stAppViewContainer"] [data-baseweb="notification"] :is(p,span,strong,em,li,a,code) {
  color:#153653 !important;
  -webkit-text-fill-color:#153653 !important;
}
[data-testid="stAppViewContainer"] [data-testid="stAlert"] svg,
[data-testid="stAppViewContainer"] [data-baseweb="notification"] svg {color:#1765a6 !important;fill:currentColor;}
[data-testid="stAppViewContainer"] [data-testid="stAlert"]:has([data-testid*="Success"]),
[data-testid="stAppViewContainer"] [data-baseweb="notification"]:has([data-testid*="Success"]) {
  background:#e7f7f0 !important;border-color:#b6e2cd !important;
}
[data-testid="stAppViewContainer"] [data-testid="stAlert"]:has([data-testid*="Warning"]),
[data-testid="stAppViewContainer"] [data-baseweb="notification"]:has([data-testid*="Warning"]) {
  background:#fff4df !important;border-color:#f2d69b !important;
}
[data-testid="stAppViewContainer"] [data-testid="stAlert"]:has([data-testid*="Error"]),
[data-testid="stAppViewContainer"] [data-baseweb="notification"]:has([data-testid*="Error"]) {
  background:#fff0f0 !important;border-color:#f2bfc5 !important;
}

/* Active actions: white labels on blue, never dark-blue on blue. */
[data-testid="stAppViewContainer"] .stButton > button[kind="primary"]:not(:disabled),
[data-testid="stAppViewContainer"] .stDownloadButton > button[kind="primary"]:not(:disabled),
[data-testid="stAppViewContainer"] button[data-testid="stBaseButton-primary"]:not(:disabled) {
  background:#1666a6 !important;border-color:#1666a6 !important;
  color:#fff !important;opacity:1 !important;
}
[data-testid="stAppViewContainer"] .stButton > button[kind="primary"]:not(:disabled) *,
[data-testid="stAppViewContainer"] .stDownloadButton > button[kind="primary"]:not(:disabled) *,
[data-testid="stAppViewContainer"] button[data-testid="stBaseButton-primary"]:not(:disabled) * {
  color:#fff !important;-webkit-text-fill-color:#fff !important;fill:currentColor;
}
[data-testid="stAppViewContainer"] .stButton > button:disabled,
[data-testid="stAppViewContainer"] .stDownloadButton > button:disabled,
[data-testid="stAppViewContainer"] button[data-testid^="stBaseButton"]:disabled {
  background:#e4edf6 !important;border:1px solid #c5d6e7 !important;
  color:#526981 !important;opacity:1 !important;box-shadow:none !important;cursor:not-allowed !important;
}
[data-testid="stAppViewContainer"] .stButton > button:disabled *,
[data-testid="stAppViewContainer"] .stDownloadButton > button:disabled *,
[data-testid="stAppViewContainer"] button[data-testid^="stBaseButton"]:disabled * {
  color:#526981 !important;-webkit-text-fill-color:#526981 !important;fill:currentColor;
}
/* Links, secondary actions, form controls */
[data-testid="stAppViewContainer"] .stLinkButton > a,
[data-testid="stAppViewContainer"] [data-testid="stLinkButton"] a {
  background:#fff !important;border:1px solid #b7cde2 !important;
  color:#155a94 !important;text-decoration:none !important;border-radius:9px !important;
}
[data-testid="stAppViewContainer"] .stLinkButton > a *,
[data-testid="stAppViewContainer"] [data-testid="stLinkButton"] a * {
  color:#155a94 !important;-webkit-text-fill-color:#155a94 !important;
}
[data-testid="stAppViewContainer"] .stLinkButton > a:hover,
[data-testid="stAppViewContainer"] [data-testid="stLinkButton"] a:hover {background:#eaf4fc !important;}
[data-testid="stAppViewContainer"] [data-testid="stFileUploaderDropzone"] {
  background:#f8fbff !important;border:1px dashed #a0c1e0 !important;
}
[data-testid="stAppViewContainer"] [data-testid="stFileUploaderDropzone"] :is(p,span,small),
[data-testid="stAppViewContainer"] [data-testid="stFileUploaderDropzoneInstructions"] :is(span,small) {
  color:#24496a !important;-webkit-text-fill-color:#24496a !important;
}
[data-testid="stAppViewContainer"] [data-testid="stFileUploader"] button {
  background:#fff !important;border-color:#a8c8e4 !important;color:#155a94 !important;
}
[data-testid="stAppViewContainer"] [data-testid="stFileUploader"] button * {color:#155a94 !important;}
[data-testid="stAppViewContainer"] [data-baseweb="select"] > div,
[data-testid="stAppViewContainer"] [data-baseweb="input"] > div {
  background:#fff !important;color:#1e3a55 !important;border-color:#c9d9e9 !important;
}
[data-testid="stAppViewContainer"] [data-baseweb="select"] *,
[data-testid="stAppViewContainer"] [data-baseweb="input"] input {color:#1e3a55 !important;}

/* Keep code blocks dark but readable; avoid darkening regular content. */
[data-testid="stAppViewContainer"] [data-testid="stCodeBlock"] {background:#102235 !important;}
[data-testid="stAppViewContainer"] [data-testid="stCodeBlock"] pre,
[data-testid="stAppViewContainer"] [data-testid="stCodeBlock"] pre :is(span,code) {
  color:#e9f3ff !important;-webkit-text-fill-color:#e9f3ff !important;
}
/* Ensure custom hero/brand text keeps its intended white foreground. */
[data-testid="stAppViewContainer"] .portal-hero :is(h2,p,span),
[data-testid="stSidebar"] .portal-sidebar-brand {color:#fff !important;}
"""


# Sidebar control and clean light header.
PORTAL_CSS += r"""
/* Clean native top chrome instead of leaving an unstyled white strip. */
[data-testid="stHeader"] {
  background:#ffffff !important;
  border-bottom:1px solid #e0eaf3 !important;
  box-shadow:0 2px 9px rgba(19,54,83,.06) !important;
  z-index:990 !important;
}
[data-testid="stToolbar"] {background:transparent !important;}
/* Older and newer Streamlit releases use different sidebar expand test IDs. */
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"],
[data-testid="stExpandSidebarButton"] {
  visibility:visible !important;
  opacity:1 !important;
  z-index:2000 !important;
  background:#fff !important;
  border:1px solid #bdd2e6 !important;
  border-radius:10px !important;
  box-shadow:0 2px 10px rgba(12,54,90,.14) !important;
}
[data-testid="stSidebarCollapsedControl"]:is(div,button),
[data-testid="collapsedControl"]:is(div,button) {
  position:fixed !important;
  top:7px !important;
  left:12px !important;
  padding:3px !important;
}
[data-testid="stSidebarCollapsedControl"] button,
[data-testid="collapsedControl"] button,
[data-testid="stExpandSidebarButton"] button {
  min-width:32px !important;min-height:32px !important;
  color:#0d548e !important;opacity:1 !important;
}
[data-testid="stSidebarCollapsedControl"] svg,
[data-testid="collapsedControl"] svg,
[data-testid="stExpandSidebarButton"] svg {
  color:#0d548e !important;stroke:#0d548e !important;
  fill:currentColor !important;
}
/* Keep the collapse arrow visibly available *inside* the expanded navy sidebar. */
[data-testid="stSidebarCollapseButton"] {opacity:1 !important;z-index:30 !important;}
[data-testid="stSidebarCollapseButton"] button,
[data-testid="stSidebarCollapseButton"] svg {
  color:#e5f3ff !important;stroke:#e5f3ff !important;
}
/* Main header adapts to the full width when the sidebar is collapsed. */
.st-key-portal_header {
  background:#fff !important;
  border:1px solid #dde8f2 !important;
  border-radius:13px !important;
  box-shadow:0 2px 10px rgba(12,54,90,.06) !important;
  padding:10px 16px !important;
  margin:4px 0 23px !important;
}
.st-key-portal_header .portal-topbar {
  background:transparent !important;border:0 !important;
  box-shadow:none !important;margin:0 !important;padding:0 !important;
  min-height:42px !important;
}
.st-key-portal_header .portal-crumb {gap:8px !important;}
.st-key-portal_header .portal-header-actions {flex-shrink:0 !important;}
.st-key-portal_header [data-testid="stSelectbox"] [data-baseweb="select"] > div {
  background:#f6faff !important;border-color:#bfd4e7 !important;
  color:#173a59 !important;min-height:38px !important;
}
.st-key-portal_header [data-testid="stSelectbox"] [data-baseweb="select"] span {color:#173a59 !important;}
@media(max-width:800px) {
  .st-key-portal_header{padding:9px 11px !important;}
  .st-key-portal_header .portal-topbar {flex-wrap:wrap;gap:8px;}
  .st-key-portal_header .portal-crumb {font-size:11px !important;}
}
"""



# Final UI accessibility override.  The existing theme can otherwise cause
# low-contrast text on a navy sidebar, and render the collapsed sidebar icon
# as a nearly invisible pale arrow on the white Streamlit header.
PORTAL_CSS += r"""
/* HELMETGUARD • LSPU–LB • accessible navigation and contrast */
/* Sidebar foreground must beat the application's light-theme typography. */
[data-testid="stSidebar"] :is(
  [data-testid="stRadio"] label,
  [data-testid="stRadio"] label p,
  [data-testid="stRadio"] label span,
  [data-testid="stRadio"] label div,
  [data-testid="stSlider"] label,
  [data-testid="stSlider"] p,
  [data-testid="stSlider"] span,
  [data-testid="stSelectSlider"] label,
  [data-testid="stSelectSlider"] p,
  [data-testid="stSelectSlider"] span,
  [data-testid="stToggle"] label,
  [data-testid="stToggle"] p,
  [data-testid="stToggle"] span,
  [data-testid="stWidgetLabel"] p,
  [data-testid="stMarkdownContainer"] p,
  [data-testid="stMarkdownContainer"] strong,
  [data-testid="stMarkdownContainer"] span,
  [data-testid="stCaptionContainer"] p
) {
  color:#f0f7ff !important;
  -webkit-text-fill-color:#f0f7ff !important;
  opacity:1 !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
  background:rgba(255,255,255,.13) !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
  background:#2a6193 !important;
  border-left:3px solid #63e1d9 !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) :is(p,span,div) {
  color:#fff !important;-webkit-text-fill-color:#fff !important;
}
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
  font-weight:750 !important;letter-spacing:.06em !important;
}
[data-testid="stSidebar"] .portal-schoolbox,
[data-testid="stSidebar"] .portal-schoolbox span,
[data-testid="stSidebar"] .portal-sidebar-foot,
[data-testid="stSidebar"] .portal-sidebar-status,
[data-testid="stSidebar"] .portal-sidebar-brand small {
  color:#edf7ff !important;-webkit-text-fill-color:#edf7ff !important;
}
[data-testid="stSidebar"] .portal-monogram {
  color:#0e508b !important;-webkit-text-fill-color:#0e508b !important;
}
[data-testid="stSidebar"] .portal-sidebar-brand {
  color:#fff !important;-webkit-text-fill-color:#fff !important;
}
/* Make collapsed sidebar's ACTUAL interactive button visible on a white header.
   Different Streamlit releases use different data-testid attributes. */
:is([data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"],
    [data-testid="stExpandSidebarButton"]) {
  opacity:1 !important;visibility:visible !important;
  z-index:2500 !important;
  background:transparent !important;
}
:is([data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"],
    [data-testid="stExpandSidebarButton"]) button,
button[aria-label="Open sidebar"],
button[aria-label="Expand sidebar"],
button[title="Open sidebar"] {
  display:flex !important;align-items:center !important;justify-content:center !important;
  min-width:43px !important;min-height:40px !important;
  color:#fff !important;background:#104d83 !important;
  border:2px solid #0b416f !important;border-radius:10px !important;
  box-shadow:0 2px 8px rgba(9,52,94,.23) !important;
  opacity:1 !important;visibility:visible !important;
  position:relative !important;cursor:pointer !important;
}
:is([data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"],
    [data-testid="stExpandSidebarButton"]) button:hover,
button[aria-label="Open sidebar"]:hover,
button[aria-label="Expand sidebar"]:hover {
  background:#1768a8 !important;
}
/* Use exactly the native Streamlit sidebar arrow. No synthetic hamburger. */
:is([data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"],
    [data-testid="stExpandSidebarButton"]) button::after,
button[aria-label="Open sidebar"]::after,
button[aria-label="Expand sidebar"]::after {
  content:none !important;
  display:none !important;
}
:is([data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"],
    [data-testid="stExpandSidebarButton"]) button svg,
button[aria-label="Open sidebar"] svg,
button[aria-label="Expand sidebar"] svg {
  visibility:visible !important;
  opacity:1 !important;
  color:#0b4c81 !important;
  stroke:#0b4c81 !important;
  fill:currentColor !important;
}
/* Header action keeps an accessible name, native hover, and remains clickable. */
[data-testid="stHeader"] {min-height:46px !important;}
[data-testid="stHeader"] [data-testid="stSidebarCollapsedControl"] {
  position:fixed !important;left:10px !important;top:5px !important;
}
[data-testid="stSidebarCollapseButton"] button {
  opacity:1 !important;min-width:34px !important;min-height:34px !important;
  background:rgba(255,255,255,.13) !important;border-radius:8px !important;
}
[data-testid="stSidebarCollapseButton"] button svg {
  color:#fff !important;opacity:1 !important;fill:#fff !important;
}
/* Main panel and alerts: avoid dark-text-on-dark-surface widgets. */
[data-testid="stAppViewContainer"] [data-testid="stAlert"],
[data-testid="stAppViewContainer"] [data-baseweb="notification"] {
  background:#e6f3ff !important;border:1px solid #b6d2ee !important;
  color:#163b5c !important;
}
[data-testid="stAppViewContainer"] [data-testid="stAlert"] :is(p,div,span,strong,small),
[data-testid="stAppViewContainer"] [data-baseweb="notification"] :is(p,div,span,strong,small) {
  color:#163b5c !important;-webkit-text-fill-color:#163b5c !important;
}
/* Checkpoint success/green banners remain clearly legible. */
[data-testid="stAppViewContainer"] [data-testid="stAlert"]:has([data-testid*="Success"]),
[data-testid="stAppViewContainer"] [data-baseweb="notification"]:has([data-testid*="Success"]) {
  background:#e6f7ec !important;border-color:#b2dfc5 !important;
}
/* Ensure native buttons, sliders, and navigation are not faded. */
[data-testid="stAppViewContainer"] [data-testid="stBaseButton-primary"]:not(:disabled),
[data-testid="stAppViewContainer"] .stButton button[kind="primary"]:not(:disabled) {
  background:#1264a3 !important;color:#fff !important;
}
[data-testid="stAppViewContainer"] [data-testid="stBaseButton-primary"]:not(:disabled) *,
[data-testid="stAppViewContainer"] .stButton button[kind="primary"]:not(:disabled) * {
  color:#fff !important;-webkit-text-fill-color:#fff !important;
}
[data-testid="stAppViewContainer"] [data-testid="stBaseButton-primary"]:disabled,
[data-testid="stAppViewContainer"] .stButton button[kind="primary"]:disabled {
  background:#dce8f4 !important;color:#536a80 !important;opacity:1 !important;
}
[data-testid="stAppViewContainer"] [data-testid="stBaseButton-primary"]:disabled *,
[data-testid="stAppViewContainer"] .stButton button[kind="primary"]:disabled * {
  color:#536a80 !important;-webkit-text-fill-color:#536a80 !important;
}
/* Breadcrumbs remain readable. */
.st-key-portal_header .portal-crumb {color:#3b5873 !important;}
.st-key-portal_header .portal-root {color:#095992 !important;}
.st-key-portal_header [data-testid="stSelectbox"] [data-baseweb="select"] {
  border-radius:9px !important;
}
.st-key-portal_header [data-testid="stSelectbox"] [data-baseweb="select"] :is(span,div) {
  color:#173a59 !important;-webkit-text-fill-color:#173a59 !important;
}
"""

# The app uses a light main panel. Override Streamlit's dark notification skin,
# including the inner alert element (some releases paint that child directly).
PORTAL_CSS += r"""
/* HELMETGUARD accessibility hotfix: readable status messages */
[data-testid="stAppViewContainer"] :is(
    [data-testid="stAlert"],
    [data-testid="stAlertContainer"],
    [data-baseweb="notification"]
) {
  background:#edf5ff !important;
  border:1px solid #bfd5ed !important;
  color:#113a5a !important;
  border-radius:10px !important;
  box-shadow:none !important;
}
[data-testid="stAppViewContainer"] :is(
    [data-testid="stAlert"],
    [data-testid="stAlertContainer"],
    [data-baseweb="notification"]
) :is(p,span,small,strong,em,li,code,div) {
  color:#113a5a !important;
  -webkit-text-fill-color:#113a5a !important;
  opacity:1 !important;
}
[data-testid="stAppViewContainer"] :is(
    [data-testid="stAlert"],
    [data-testid="stAlertContainer"],
    [data-baseweb="notification"]
) svg {
  color:#126aa4 !important;
  opacity:1 !important;
}
/* Prefer a subtle light background over dark-green/dark-blue for all statuses. */
[data-testid="stAppViewContainer"] [data-testid="stAlert"] > div,
[data-testid="stAppViewContainer"] [data-testid="stAlertContainer"] > div,
[data-testid="stAppViewContainer"] [data-baseweb="notification"] > div {
  background-color:transparent !important;
  color:#113a5a !important;
}
/* The one real sidebar expand arrow is dark navy on a white button. */
:is([data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"],
    [data-testid="stExpandSidebarButton"]) button,
button[aria-label="Open sidebar"],
button[aria-label="Expand sidebar"] {
  background:#fff !important;
  border:1px solid #a8c5e0 !important;
  box-shadow:0 1px 5px rgba(11,58,96,.12) !important;
  opacity:1 !important;
  visibility:visible !important;
}
:is([data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"],
    [data-testid="stExpandSidebarButton"]) button::after,
button[aria-label="Open sidebar"]::after,
button[aria-label="Expand sidebar"]::after {
  content:none !important;
  display:none !important;
}
:is([data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"],
    [data-testid="stExpandSidebarButton"]) svg,
button[aria-label="Open sidebar"] svg,
button[aria-label="Expand sidebar"] svg {
  opacity:1 !important;
  visibility:visible !important;
  color:#0a4c80 !important;
  stroke:#0a4c80 !important;
  fill:currentColor !important;
}
/* Keep expanded-sidebar close icon clearly visible too. */
[data-testid="stSidebarCollapseButton"] svg {
  opacity:1 !important;
  color:#fff !important;
  stroke:#fff !important;
}
/* Fix compact header after removing the redundant navigation dropdown. */
.st-key-portal_header .portal-topbar {
  min-height:42px !important;
  align-items:center !important;
  justify-content:space-between !important;
}
.st-key-portal_header .portal-header-actions {
  display:flex !important;
  align-items:center !important;
  gap:9px !important;
}
"""

st.markdown('<style>' + PORTAL_CSS + '</style>', unsafe_allow_html=True)

@st.cache_resource(show_spinner='Loading your trained YOLO11 model...')
def load_model(location: str, modified_at: float):
    from ultralytics import YOLO
    model = YOLO(location)
    names = getattr(model, 'names', {})
    if not model_has_helmet_classes(names):
        raise ValueError('This checkpoint is missing recognized Helmet and No Helmet classes. Train YOLO11 on the selected motorcycle dataset and upload the resulting best.pt.')
    return model, names


def current_model_classes(model):
    """Return class labels in class-ID order, not dictionary insertion order."""
    if model is None:
        return []
    labels = getattr(model, 'names', {})
    if isinstance(labels, dict):
        return [str(labels.get(i, labels.get(str(i), ''))) for i in range(len(labels))]
    return [str(label) for label in labels]


def enhanced_checkpoint(model):
    return current_model_classes(model) == ENHANCED_CLASSES


def report_matches_model(report, model):
    """None means report has no class metadata; do not claim verification."""
    if model is None:
        return None
    expected = current_model_classes(model)
    if 'classes' in report and isinstance(report['classes'], list):
        return report['classes'] == expected
    per_class = report.get('per_class', [])
    if per_class:
        for result in per_class:
            cid = result.get('class_id')
            if not isinstance(cid, int) or cid < 0 or cid >= len(expected):
                return False
            if result.get('class') != expected[cid]:
                return False
        # Partial results do not verify the entire class schema.
        return None
    return None


def get_model_status():
    path = model_path()
    if not path:
        return None, None, 'No custom-trained checkpoint found.'
    try:
        model, names = load_model(str(path), path.stat().st_mtime)
        return model, path, None
    except Exception as exc:
        return None, path, str(exc)


def page_intro(kicker: str, heading: str, description: str):
    from html import escape
    st.markdown(f'<div class="portal-eyebrow">{escape(kicker)}</div>', unsafe_allow_html=True)
    st.title(heading)
    st.markdown(f'<p class="portal-subtitle">{escape(description)}</p>', unsafe_allow_html=True)


def portal_hero(kicker: str, heading: str, description: str, glyph: str = '🛡'):
    from html import escape
    st.markdown(
        '<div class="portal-hero"><div class="portal-hero-main">'
        f'<div class="portal-hero-kicker">{escape(kicker)}</div>'
        f'<h2>{escape(heading)}</h2><p>{escape(description)}</p>'
        f'</div><div class="portal-hero-glyph" aria-hidden="true">{escape(glyph)}</div></div>',
        unsafe_allow_html=True)


def steps(*items):
    from html import escape
    st.markdown('<div class="portal-guide">' + ''.join(
        f'<span><b>{i+1:02d}</b> {escape(label)}</span>' for i, label in enumerate(items)
    ) + '</div>', unsafe_allow_html=True)


def status_pill(flagged: bool, neutral: bool = False):
    if flagged:
        return '<span class="portal-pill red">● Possible no-helmet</span>'
    if neutral:
        return '<span class="portal-pill neutral">● No known detections</span>'
    return '<span class="portal-pill">● Helmet detected</span>'


def needs_model():
    st.warning('A custom-trained YOLO11 checkpoint is required. Open **Model lab** in the sidebar to install your Colab `best.pt` file.')
    if model_error and checkpoint:
        st.error('The checkpoint file exists but cannot be used: ' + model_error)
    st.caption('The standard `yolo11n.pt` detects general COCO objects, not helmet/no-helmet categories. Do not use it as a helmet model.')


def download_batch_zip(batch):
    stream = BytesIO()
    with ZipFile(stream, 'w', ZIP_DEFLATED) as bundle:
        csv = pd.DataFrame([{
            'filename': r['name'], 'helmet_boxes': r['helmet'], 'no_helmet_boxes': r['no_helmet'],
            'other_boxes': r['other'], 'runtime_ms': r['elapsed_ms']
        } for r in batch])
        bundle.writestr('detection_summary.csv', csv.to_csv(index=False))
        for idx, r in enumerate(batch, 1):
            safe = ''.join(ch for ch in Path(r['name']).stem if ch.isalnum() or ch in '-_ ')[:48].strip() or 'image'
            bundle.writestr(f'{idx:02d}_{safe}_annotated.png', r['annotated'])
    return stream.getvalue()


def install_custom_model(file):
    """Validate custom checkpoint classes before atomically installing it locally.

    Only load a PyTorch checkpoint created by you or another trusted source:
    loading third-party .pt files can execute malicious code.
    """
    if file is None:
        raise ValueError('Choose a model checkpoint first.')
    if not file.name.lower().endswith('.pt'):
        raise ValueError('Select a YOLO .pt checkpoint.')
    data = file.getvalue()
    if not data or len(data) > MAX_CHECKPOINT_MB * 1024**2:
        raise ValueError(f'Model must be nonempty and smaller than {MAX_CHECKPOINT_MB} MB.')
    MODEL_TARGET.parent.mkdir(exist_ok=True)
    temporary = MODEL_TARGET.parent / f'.candidate_{uuid.uuid4().hex}.pt'
    try:
        temporary.write_bytes(data)
        from ultralytics import YOLO
        candidate = YOLO(str(temporary))
        if not model_has_helmet_classes(candidate.names):
            raise ValueError('Your model does not have recognizable helmet and no-helmet class names. Upload the fine-tuned best.pt from your Colab motorcycle dataset.')
        # Release checkpoint before replacement so Windows does not retain a file handle.
        del candidate
        os.replace(temporary, MODEL_TARGET)
        load_model.clear()
    finally:
        temporary.unlink(missing_ok=True)


if 'analysis' not in st.session_state:
    st.session_state.analysis = []
if 'video_analysis' not in st.session_state:
    st.session_state.video_analysis = None

NAV_PAGES = ['Overview', 'Image studio', 'Video studio', 'History', 'Model lab']

if ('current_page' not in st.session_state
        or st.session_state.current_page not in NAV_PAGES):
    st.session_state.current_page = 'Overview'


def navigate(target: str):
    st.session_state.current_page = target


model, checkpoint, model_error = get_model_status()
ready = model is not None

with st.sidebar:
    st.markdown('''<div class="portal-sidebar-logo">
      <div class="portal-monogram">HG</div>
      <div class="portal-sidebar-brand">HelmetGuard<small>MOTORCYCLE HELMET DETECTION</small></div>
    </div>''', unsafe_allow_html=True)
    st.markdown('''<div class="portal-schoolbox"><strong>LSPU–LB · CSST 106</strong><br>Motorcycle Helmet Detection System<br><span style="font-size:10px;color:#a8cce8">Independent student system · not an official university service</span></div>''', unsafe_allow_html=True)
    st.markdown('<div class="portal-sidebar-status"><span class="portal-dot ' + ('ok' if ready else '') + '"></span>' + ('Model online · best.pt' if ready else 'Checkpoint setup needed') + '</div>', unsafe_allow_html=True)
    st.divider()
    page = st.radio('MODULES', NAV_PAGES, label_visibility='visible',
                    key='current_page')
    st.divider()
    st.markdown('**Detection settings**')
    confidence = st.slider('Confidence threshold', 0.05, 0.85, 0.20, 0.05,
        help='Minimum detection confidence. It is not measured model accuracy.')
    iou = st.slider('Box overlap (IoU)', 0.20, 0.80, 0.45, 0.05,
        help='Non-maximum suppression overlap threshold.')
    inference_size = st.select_slider('Detection resolution', options=[640, 960, 1280], value=640,
        help='Try 960 for distant, small helmet objects. Larger values process more slowly.')
    save_images = st.toggle('Save annotated copies', False,
        help='Saves output photos locally in .data/saved_images; off by default.')
    st.markdown('<div class="portal-sidebar-foot">STUDENT RESEARCH APPLICATION<br>Motorcycle safety monitoring<br><strong style="color:#fff">Local processing · Human review required</strong></div>', unsafe_allow_html=True)

from html import escape as _escape_html
# One navigation system: the left Streamlit sidebar. The native sidebar arrow
# remains visible after collapse; no duplicate floating hamburger or top dropdown.
with st.container(border=False, key='portal_header'):
    st.markdown(f'''<div class="portal-topbar">
      <div class="portal-crumb"><span class="portal-root">LSPU–LB</span><span>/</span> HelmetGuard · Helmet Detection <span>/</span> <strong>{_escape_html(page)}</strong></div>
      <div class="portal-header-actions"><span class="portal-top-tag">CSST 106</span><span class="portal-top-tag {'ready' if ready else ''}">{'● Model ready' if ready else '○ Setup required'}</span></div>
    </div>''', unsafe_allow_html=True)

if page == 'Overview':
    page_intro('LSPU–LB / STUDENT PROJECT', 'Motorcycle Helmet Detection Dashboard',
               'Review actual scan activity, check your model, and move from image upload to transparent detection results.')
    portal_hero('STUDENT RESEARCH · CSST 106', 'Helmet monitoring, simplified.', 'An academic YOLO11 workspace for reviewing uploaded motorcycle images, detecting helmet-related regions, and recording explainable results.', '🛡')
    rows = load_scans()
    stats = summarize_history(rows)
    a, b, c, d = st.columns(4)
    a.metric('Images inspected', stats['images'])
    b.metric('Flagged images', stats['flagged_images'])
    c.metric('Helmet boxes', stats['helmet_boxes'])
    d.metric('No-helmet boxes', stats['no_helmet_boxes'])
    left, right = st.columns([1.55, 1], gap='large')
    with left:
        st.subheader('Last 7 days')
        if rows:
            daily = pd.DataFrame(recent_daily_counts(rows))
            st.bar_chart(daily, x='date', y=['Scans', 'Flagged images'], height=285)
        else:
            st.info('Once you analyze images, your daily activity chart appears here.')
    with right:
        st.subheader('System status')
        with st.container(border=True):
            st.markdown('**Checkpoint**')
            if ready:
                st.success('Ready · ' + checkpoint.name)
            else:
                st.warning('Custom model not installed')
            active_classes = current_model_classes(model)
            st.caption(f"{'Enhanced six-class model' if enhanced_checkpoint(model) else 'Original five-class model' if active_classes == ORIGINAL_CLASSES else 'Custom checkpoint'} · {len(active_classes)} classes")
            st.divider()
            st.markdown('**Important distinction**')
            st.caption('A helmet box is not necessarily attached to a rider. Video detection does not count unique riders, and no-box results do not prove compliance.')
    st.subheader('Recent inspections')
    if rows:
        frame = pd.DataFrame(rows[:6])
        st.dataframe(frame[['timestamp', 'filename', 'helmet', 'no_helmet', 'other']].rename(columns={
            'timestamp':'Time', 'filename':'Image', 'helmet':'Helmet', 'no_helmet':'No helmet', 'other':'Other'
        }), hide_index=True, use_container_width=True)
    else:
        st.info('No inspections yet. Start with the **Image studio** page.')
    st.markdown('<div class="portal-mini-section">Available modules</div>', unsafe_allow_html=True)
    p1, p2, p3 = st.columns(3, gap='medium')
    with p1:
        st.markdown('<div class="portal-feature"><div class="portal-feature-icon">◫</div><b>Image detection</b><p>Analyze motorcycle photos and compare before and after results.</p></div>', unsafe_allow_html=True)
        st.button('Open image studio', use_container_width=True, key='go_images', on_click=navigate, args=('Image studio',))
    with p2:
        st.markdown('<div class="portal-feature"><div class="portal-feature-icon">◷</div><b>Detection history</b><p>Filter local scans and export a CSV for academic reporting.</p></div>', unsafe_allow_html=True)
        st.button('View history', use_container_width=True, key='go_history', on_click=navigate, args=('History',))
    with p3:
        st.markdown('<div class="portal-feature"><div class="portal-feature-icon">⌁</div><b>Model lab</b><p>Install best.pt and inspect your model evaluation evidence.</p></div>', unsafe_allow_html=True)
        st.button('Open model lab', use_container_width=True, key='go_models', on_click=navigate, args=('Model lab',))
    st.markdown('<div class="portal-mini-section">Detection workflow</div>', unsafe_allow_html=True)
    steps('Upload photo', 'Run YOLO11', 'Inspect detections', 'Export results', 'Evaluate')

elif page == 'Image studio':
    page_intro('IMAGE STUDIO / OBJECT DETECTION', 'Helmet Detection Studio',
               'Upload a photo or a batch of motorcycle images, then compare before-and-after results with full confidence details.')
    portal_hero('IMAGE DETECTION LAB', 'Upload. Detect. Review.', 'Compare original images and model predictions in one place. Export annotated evidence for your CSST 106 defense.', '◈')
    if not ready:
        needs_model()
    with st.container(border=True):
        st.markdown('#### Upload images')
        files = st.file_uploader('Choose motorcycle images', type=['jpg','jpeg','png','webp'],
                                 accept_multiple_files=True, help='Up to 25 images. Each image should be no larger than 15 MB.')
        with st.expander('Use device camera instead'):
            camera = st.camera_input('Capture image')
        left, right = st.columns([1,1])
        with left:
            st.caption(f'{len(files or []) + int(camera is not None)} image(s) selected • Confidence {confidence:.0%} • IoU {iou:.2f}')
        with right:
            run = st.button('✦  Analyze selected images', type='primary', use_container_width=True,
                            disabled=not (ready and (files or camera)))
    if run:
        queue = list(files or []) + ([camera] if camera else [])
        if len(queue) > 25:
            st.error('Select 25 images or fewer per batch.')
        else:
            outputs = []
            bar = st.progress(0, text='Preparing images...')
            for i, item in enumerate(queue):
                try:
                    original = item.getvalue()
                    img = prepare_image(original)
                    prediction = analyze(model, img, confidence, iou, imgsz=inference_size)
                    annotated = png_bytes(prediction['annotated'])
                    save_scan(item.name, checkpoint.name, prediction, annotated if save_images else None)
                    # The app shows the preprocessed original so overlay dimensions match exactly.
                    source = png_bytes(img)
                    outputs.append({
                        'name': item.name, 'original': source, 'annotated': annotated,
                        'helmet': prediction['helmet'], 'no_helmet': prediction['no_helmet'],
                        'other': prediction['other'], 'elapsed_ms': prediction['elapsed_ms'],
                        'detections': [det.as_dict() for det in prediction['detections']],
                        'threshold_used': confidence, 'resolution_used': inference_size,
                        'model_used': checkpoint.name,
                    })
                except Exception as exc:
                    st.warning(f'Unable to analyze {item.name}: {exc}')
                bar.progress((i+1)/len(queue), text=f'Analyzing {i+1}/{len(queue)}')
            bar.empty()
            st.session_state.analysis = outputs
            if outputs:
                st.success(f'Analysis complete: {len(outputs)} image(s).')
            else:
                st.error('No images could be analyzed.')
    batch = st.session_state.analysis
    if batch:
        st.subheader('Detection results')
        c1,c2,c3,c4 = st.columns(4)
        c1.metric('Images processed', len(batch))
        c2.metric('Helmet boxes', sum(r['helmet'] for r in batch))
        c3.metric('No-helmet boxes', sum(r['no_helmet'] for r in batch))
        c4.metric('Average runtime', f"{sum(r['elapsed_ms'] for r in batch)/len(batch):.0f} ms")
        st.caption('These are model-generated boxes, not unique-rider counts or verified offenses.')
        selection = st.selectbox('Inspect image', range(len(batch)), format_func=lambda i:f"{i+1:02d}  ·  {batch[i]['name']}")
        current = batch[selection]
        if (current.get('threshold_used') != confidence or current.get('resolution_used') != inference_size
                or current.get('model_used') != checkpoint.name):
            st.info('Detection settings or checkpoint changed after this run. Click **Analyze selected images** again for updated results.')
        st.markdown(status_pill(current['no_helmet'] > 0, not (current['helmet'] or current['other'])), unsafe_allow_html=True)
        st.markdown('')
        tab_a,tab_b,tab_c = st.tabs(['Side-by-side comparison','Interactive overlay','Detection details'])
        with tab_a:
            a,b = st.columns(2, gap='medium')
            with a:
                st.caption('ORIGINAL / PREPROCESSED')
                st.image(current['original'], use_container_width=True)
            with b:
                st.caption('YOLO11 / ANNOTATED')
                st.image(current['annotated'], use_container_width=True)
        with tab_b:
            strength = st.slider('Show annotated layer (%)', min_value=0, max_value=100, value=70, step=5,
                                 key=f'overlay_{selection}')
            st.image(overlay_images(current['original'], current['annotated'], strength), use_container_width=True)
            st.caption('Move left to reveal the original; move right to show the detection boxes.')
        with tab_c:
            if current['detections']:
                df = pd.DataFrame(current['detections'])
                df['confidence'] = df['confidence'].map(lambda x:f'{x:.1%}')
                st.dataframe(df.rename(columns={'label':'Group', 'model_class':'Dataset class', 'confidence':'Confidence'}),
                             use_container_width=True, hide_index=True)
            else:
                st.info('The model returned no recognized boxes for this image. This does not establish helmet compliance.')
            st.write(f"**Per-image totals:** {current['helmet']} helmet · {current['no_helmet']} no helmet · {current['other']} other")
        with st.expander('Detection diagnostics — why are some helmets missed?', expanded=not (current['helmet'] or current['no_helmet'])):
            st.write(f"**Raw boxes returned:** {len(current['detections'])} at confidence {current.get('threshold_used', confidence):.0%} and resolution {current.get('resolution_used', 640)}")
            if current['detections']:
                from collections import Counter
                raw_counts = Counter(d['model_class'] for d in current['detections'])
                st.dataframe(pd.DataFrame([{'Original YOLO class': name, 'Boxes': count}
                    for name, count in raw_counts.items()]), hide_index=True, use_container_width=True)
                if current['other'] and not (current['helmet'] or current['no_helmet']):
                    st.warning('The model returned boxes, but none mapped to Helmet or No Helmet. Check the raw classes shown above.')
            else:
                st.warning('The model returned no boxes above the selected threshold. Try 0.10 confidence and 960 resolution, then rerun. If it still fails, test your trained best.pt on labeled held-out photos; UI changes cannot correct a weak checkpoint.')
            st.caption('A missing box does not mean the rider has a helmet. The model cannot automatically prove a legal violation.')
        a,b=st.columns(2)
        with a:
            st.download_button('↓ Download this annotated image', current['annotated'],
                               file_name=f"helmetguard_{selection+1:02d}.png", mime='image/png', use_container_width=True)
        with b:
            st.download_button('↓ Download all results + CSV', download_batch_zip(batch),
                               file_name='HelmetGuard_detection_results.zip', mime='application/zip', use_container_width=True)
        with st.expander('View all batch files'):
            st.dataframe(pd.DataFrame([{k:r[k] for k in ['name','helmet','no_helmet','other','elapsed_ms']} for r in batch]),
                         hide_index=True, use_container_width=True)
    elif ready:
        st.info('Upload one or more images above and click **Analyze selected images** to begin.')

elif page == 'Video studio':
    page_intro('VIDEO STUDIO / FRAME ANALYSIS', 'Video and camera tools',
               'Sample frames from a road-safety clip, or launch the local webcam for live predictions.')
    first,second = st.tabs(['Analyze video', 'Live camera'])
    with first:
        st.info('This tool analyzes up to 30 sampled video frames. The same rider may appear repeatedly; these are frame-level counts, not unique-rider counts.')
        if not ready:
            needs_model()
        video = st.file_uploader('Upload a motorcycle video', type=['mp4','avi','mov','mkv'], key='clip', help='Maximum video size: 150 MB')
        sample_count = st.slider('Sampled frames', min_value=4, max_value=30, value=12, step=2)
        if video:
            st.video(video)
        if st.button('✦  Analyze video frames', type='primary', use_container_width=True, disabled=not (ready and video)):
            try:
                with st.spinner('Analyzing sampled video frames...'):
                    # Support both the original and the revised video_pipeline.py signatures.
                    from inspect import signature
                    args = (video.getvalue(), Path(video.name).suffix, model, confidence, iou, sample_count)
                    if 'imgsz' in signature(process_uploaded_video).parameters:
                        output = process_uploaded_video(*args, imgsz=inference_size)
                    else:
                        output = process_uploaded_video(*args)
                st.session_state.video_analysis = {'name': video.name, 'result': output}
                st.success(f"Analyzed {len(output['sampled_frames'])} frames.")
            except Exception as exc:
                st.error(f'Could not process video: {exc}')
                st.session_state.video_analysis = None
        stored = st.session_state.video_analysis
        if stored:
            frames = stored['result']['sampled_frames']
            a,b,c,d=st.columns(4)
            a.metric('Frames reviewed',len(frames))
            b.metric('Helmet boxes',sum(f['helmet'] for f in frames))
            c.metric('No-helmet boxes',sum(f['no_helmet'] for f in frames))
            d.metric('Clip length',f"{stored['result']['duration_sec']:.1f}s" if stored['result']['duration_sec'] is not None else 'Unknown')
            summary=pd.DataFrame([{'Frame':f['frame'],'Time (s)':f['time_sec'], 'Helmet':f['helmet'], 'No helmet':f['no_helmet'], 'Other':f['other']} for f in frames])
            st.dataframe(summary, hide_index=True, use_container_width=True)
            if not (summary['Helmet'].sum() or summary['No helmet'].sum()):
                st.warning('No helmet-related boxes were detected in the sampled frames. Try lowering confidence to 0.10, increasing resolution, or testing a clearer close-up video. Your model may need more representative training images.')
            with BytesIO() as out:
                with ZipFile(out,'w',ZIP_DEFLATED) as bundle:
                    bundle.writestr('frame_summary.csv',summary.to_csv(index=False))
                    for f in frames:
                        bundle.writestr(f"frame_{f['frame']:06d}.png",f['annotated'])
                st.download_button('↓ Download annotated frames and CSV',out.getvalue(),
                    file_name='HelmetGuard_video_results.zip',mime='application/zip',type='primary')
            st.subheader('Frame previews')
            for j in range(0,min(len(frames),12),3):
                cols=st.columns(3)
                for col,f in zip(cols,frames[j:j+3]):
                    with col:
                        st.image(f['annotated'],use_container_width=True)
                        st.caption(f"Frame {f['frame']} · {f['no_helmet']} possible no-helmet box(es)")
    with second:
        st.markdown('#### Browser snapshot')
        st.write('Use the **Image studio** camera input to capture and analyze one still image.')
        st.markdown('#### Live Windows webcam')
        st.write('Open a terminal inside your project folder and run:')
        st.code('python scripts/live_camera.py',language='powershell')
        st.caption('The webcam opens in a separate OpenCV window. Press Q to quit and S to save a screenshot. Live video does not track unique riders.')

elif page == 'History':
    page_intro('RECORDS / LOCAL DATABASE', 'Detection history',
               'Review previous image analyses, filter possible safety concerns, and export records for your CSST 106 documentation.')
    data = load_scans()
    total = summarize_history(data)
    a,b,c = st.columns(3)
    a.metric('Saved image scans', total['images'])
    b.metric('Images with no-helmet boxes', total['flagged_images'])
    c.metric('Helmet boxes', total['helmet_boxes'])
    a,b=st.columns([1.6,1])
    with a:
        search=st.text_input('Find filename',placeholder='Search a photo name...')
    with b:
        status=st.selectbox('Filter by finding',['All scans','Flagged images','Images with helmets','No recognized detection'])
    found=filter_history(data,search,status)
    st.caption(f'Showing {len(found)} of {len(data)} saved records.')
    if found:
        view = pd.DataFrame(found)
        cols=['timestamp','filename','helmet','no_helmet','other','avg_conf','elapsed_ms','model']
        st.dataframe(view[cols].rename(columns={
            'timestamp':'Timestamp','filename':'File','helmet':'Helmet','no_helmet':'No helmet',
            'other':'Other','avg_conf':'Avg. confidence','elapsed_ms':'Time (ms)','model':'Checkpoint'
        }), use_container_width=True, hide_index=True)
        st.download_button('↓ Export visible records to CSV',view.to_csv(index=False).encode('utf-8'),
            'HelmetGuard_history.csv','text/csv')
        saved=[r for r in found if r.get('image_path') and Path(r['image_path']).is_file()]
        if saved:
            with st.expander(f'View {len(saved)} saved annotated image(s)'):
                for i in range(0,min(12,len(saved)),3):
                    cols=st.columns(3)
                    for col,r in zip(cols,saved[i:i+3]):
                        with col:
                            st.image(r['image_path'], use_container_width=True)
                            st.caption(r['filename'])
    else:
        st.info('No matching scans yet. Run Image studio to create records.')
    with st.expander('Data management / clear all history'):
        st.warning('This deletes all local scan records and saved annotated image copies. It cannot be undone.')
        confirmed=st.checkbox('I understand that all local detection history will be deleted')
        if st.button('Clear all local history',disabled=not confirmed):
            clear_scans()
            st.session_state.analysis=[]
            st.rerun()

elif page == 'Model lab':
    page_intro('ARTIFICIAL INTELLIGENCE / LABORATORY', 'Model & evaluation',
               'Install your trained YOLO11 checkpoint, inspect dataset quality, and present real evaluation measurements.')
    portal_hero('TRAINED MODEL CENTER', 'One checkpoint. Complete workflow.', 'Install your Google Colab best.pt file, review the Roboflow dataset, and evaluate your YOLO11 detector with held-out data.', '⚙')
    if ready:
        st.success(f'Validated custom helmet detector: {checkpoint.name} ({checkpoint.stat().st_size/1024**2:.1f} MB)')
        class_names = current_model_classes(model)
        if class_names == ENHANCED_CLASSES:
            st.success('Enhanced six-class checkpoint detected. This model supports the added generic Helmet class.')
        elif class_names == ORIGINAL_CLASSES:
            st.warning('This is the ORIGINAL five-class model. After Colab retraining, install the NEW six-class best.pt to use the enhanced dataset.')
        else:
            st.info(f'Custom model with {len(class_names)} classes. Verify it matches the dataset used to generate your evaluation report.')
        with st.expander('View checkpoint classes'):
            st.write(', '.join(f'{i}: {name}' for i, name in enumerate(class_names)))
    elif model_error and checkpoint:
        st.error('Existing checkpoint is not usable: '+model_error)
    else:
        st.warning('Model not installed — upload your trained Colab best.pt below.')
    with st.container(border=True):
        st.markdown('#### Install or replace the model')
        st.write('Select **best.pt** from your own training run. HelmetGuard will check that it has both helmet and no-helmet classes before installing it.')
        upload = st.file_uploader('YOLO11 model checkpoint (.pt)',type=['pt'],key='model_upload',help=f'Maximum {MAX_CHECKPOINT_MB} MB. Only upload trusted models produced by your own training.')
        st.caption('Security note: PyTorch checkpoint files can contain executable content. Only upload a checkpoint you created or trust.')
        if st.button('Install selected model',type='primary',disabled=upload is None,use_container_width=True):
            try:
                with st.spinner('Validating checkpoint classes...'):
                    install_custom_model(upload)
                st.success('Installed best.pt successfully. Loading the model...')
                st.rerun()
            except Exception as exc:
                st.error(f'Could not install checkpoint: {exc}')
    st.markdown('#### Training roadmap')
    steps('Export Roboflow ZIP','Run Colab notebook','Download best.pt','Install model above','Test and evaluate')
    with st.expander('Step-by-step training instructions',expanded=not ready):
        st.markdown(
            '1. Open `HelmetGuard_Enhanced_YOLO11_Training.ipynb` in **Google Colab** and select a GPU runtime.\n'
            '2. Upload `helmetguard_enhanced.zip` (six classes). The notebook audits and logs annotation repairs.\n'
            '3. Train and evaluate on the held-out test split, then download the new `best.pt`.\n'
            '4. Install that checkpoint above, keeping a backup of your original `best.pt`.\n'
            '5. Extract `helmetguard_evaluation_results.zip` from Colab into your project root, preserving its `reports/` folder.\n'
            '6. Compare your class-level metrics and confusion matrix, especially for `not wearing helmet`.')
        st.code('python scripts/train.py --data datasets/helmetguard_enhanced/data.yaml --epochs 50 --batch 8',language='powershell')
    st.link_button('Open motorcycle helmet dataset', 'https://universe.roboflow.com/data-science-3-m1fa2data-preprocessing-5vgxq/motorcycle-and-helmet-detection/dataset/1')
    st.markdown('#### Dataset checks')
    # The Colab notebook exports its own audited/repaired dataset report.
    # Prefer it if it matches the installed six-class checkpoint; a local audit
    # may still reflect the old five-class data or pre-repair labels.
    audit_candidates = [
        ROOT / 'reports' / 'dataset_audit_colab.json',
        ROOT / 'reports' / 'dataset_audit.json',
    ]
    audit_records = []
    for candidate in audit_candidates:
        if candidate.is_file():
            try:
                parsed = json.loads(candidate.read_text(encoding='utf-8'))
                if isinstance(parsed, dict):
                    audit_records.append((candidate, parsed))
            except (OSError, json.JSONDecodeError):
                st.warning(f'Could not parse {candidate.name}.')
    expected_classes = current_model_classes(model)
    matched_records = [r for r in audit_records if r[1].get('classes') == expected_classes]
    chosen_audit = (matched_records[0] if matched_records else audit_records[0]) if audit_records else None
    if chosen_audit:
        audit_path, audit = chosen_audit
        audit_classes = audit.get('classes', [])
        if expected_classes and audit_classes != expected_classes:
            st.warning('The available dataset audit does not match the installed checkpoint classes. This report may be from an older dataset version. Re-run the enhanced audit or import the new Colab report.')
        elif audit.get('errors') or audit.get('problems'):
            st.warning('Dataset audit has findings or label repairs. Inspect the report and review changed annotations.')
        else:
            st.success('Dataset audit report is available.')
        st.caption(f'Report: {audit_path.name} · {len(audit_classes)} classes')
        cols = st.columns(3)
        for col, key in zip(cols, ['train', 'val', 'test']):
            col.metric(f'{key.title()} images', audit.get('splits', {}).get(key, {}).get('images', '—'))
        for line in (audit.get('errors', []) + audit.get('warnings', []))[:6]:
            st.write('• ' + str(line))
        test_boxes = audit.get('splits', {}).get('test', {}).get('boxes_by_class', {})
        no_helmet_count = int(test_boxes.get('not wearing helmet', 0))
        if no_helmet_count == 0:
            st.warning('No labeled No-Helmet boxes are present in this audit test split. Class recall cannot be established.')
        else:
            st.info(f'The audited test split contains {no_helmet_count} labeled No-Helmet boxes; use your actual measured test metrics to evaluate the new detector.')
        preview = ROOT / 'reports' / 'dataset_preview.jpg'
        if preview.is_file():
            st.image(str(preview), caption='Ground-truth annotations — manually verify labels', use_container_width=True)
        repair_log = ROOT / 'reports' / 'annotation_repair_log.csv'
        if repair_log.is_file():
            with st.expander('Review Colab annotation repair log'):
                repair_table = pd.read_csv(repair_log)
                st.dataframe(repair_table, hide_index=True, use_container_width=True)
    else:
        st.info('No audit report found. Run the enhanced dataset audit locally or extract the Colab evaluation ZIP into the project root.')
    st.code('python scripts/audit_dataset.py --data datasets/helmetguard_enhanced/data.yaml', language='powershell')

    st.markdown('#### Measured model performance')
    metrics_path = ROOT / 'reports' / 'latest_metrics.json'
    if metrics_path.is_file():
        try:
            scores = json.loads(metrics_path.read_text(encoding='utf-8'))
            matching = report_matches_model(scores, model)
            if matching is False:
                raise ValueError('This evaluation report has different class labels from the installed checkpoint. Replace it with the report from the current six-class model before displaying results.')
            elif matching is None:
                st.warning('The report does not fully identify the installed checkpoint. Check its training run and class order before presenting these figures.')
            if scores.get('split') != 'test':
                st.warning(f"This report used the {scores.get('split', 'unknown')} split, not the held-out test split.")
            cols = st.columns(4)
            for col, key, label in zip(cols,
                                       ('precision', 'recall', 'mAP50', 'mAP50_95'),
                                       ('Precision', 'Recall', 'mAP@0.50', 'mAP@0.50:0.95')):
                col.metric(label, f"{float(scores[key]):.1%}")
            st.caption(f"Reported model: {scores.get('weights', 'Unknown')} · split: {scores.get('split', 'Unknown')} · dataset: {scores.get('dataset', 'Unknown')}")
            per_class = scores.get('per_class', [])
            if per_class:
                st.markdown('##### Per-class evaluation')
                display_rows = []
                for item in per_class:
                    display_rows.append({
                        'Class': item.get('class', 'Unknown'),
                        'Test boxes': item.get('test_ground_truth_boxes', '—'),
                        'Precision': f"{float(item.get('precision', 0)):.1%}",
                        'Recall': f"{float(item.get('recall', 0)):.1%}",
                        'mAP50': f"{float(item.get('mAP50', 0)):.1%}",
                        'mAP50–95': f"{float(item.get('mAP50_95', 0)):.1%}",
                    })
                st.dataframe(pd.DataFrame(display_rows), hide_index=True, use_container_width=True)
                no_helmet = next((entry for entry in per_class if entry.get('class') == 'not wearing helmet'), None)
                if no_helmet:
                    st.markdown('##### No-Helmet detection (class ID 4)')
                    a, b = st.columns(2)
                    a.metric('No-Helmet precision', f"{float(no_helmet.get('precision', 0)):.1%}")
                    b.metric('No-Helmet recall', f"{float(no_helmet.get('recall', 0)):.1%}")
                    if no_helmet.get('test_ground_truth_boxes') is not None:
                        st.caption(f"Evaluated on {int(no_helmet['test_ground_truth_boxes'])} annotated no-helmet boxes.")
            cm = ROOT / 'reports' / 'object_detection' / 'confusion_matrix.png'
            if cm.is_file():
                st.image(str(cm), caption='Object-detection confusion matrix on labeled test data', use_container_width=True)
            with st.expander('Additional evaluation charts'):
                for chart in ('confusion_matrix_normalized.png', 'BoxPR_curve.png', 'BoxF1_curve.png'):
                    chart_path = ROOT / 'reports' / 'object_detection' / chart
                    if chart_path.is_file():
                        st.image(str(chart_path), caption=chart, use_container_width=True)
            st.download_button('Download evaluation JSON', metrics_path.read_bytes(),
                               'helmetguard_metrics.json', 'application/json')
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            st.error(f'Could not load evaluation report: {exc}')
    else:
        st.info('No local evaluation report found. From your new Colab notebook, download helmetguard_evaluation_results.zip and extract its reports/ folder into this project.')
    st.code('python scripts/evaluate.py --data datasets/helmetguard_enhanced/data.yaml --model models/best.pt --split test --device cpu', language='powershell')
    st.caption('Detection confidence is not accuracy. Keep train, validation, and test images separate. Class-level results require labeled test examples.')

st.markdown('<div class="portal-footer">HelmetGuard · CSST 106 Final Project · LSPU-LB student work · YOLO11 object detection</div>',unsafe_allow_html=True)
