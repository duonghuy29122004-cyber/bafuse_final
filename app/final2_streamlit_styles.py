from __future__ import annotations


CSS = """
<style>
:root {
  --navy: #10243a;
  --teal: #149c9a;
  --ink: #1d2b3a;
  --muted: #64748b;
  --line: #e4eaf0;
  --surface: #ffffff;
  --canvas: #f4f7fa;
}
.stApp { background: var(--canvas); color: var(--ink); }
.block-container { max-width: 1320px; padding-top: 1.1rem; padding-bottom: 3rem; }
.hero {
  background: linear-gradient(110deg, #10243a 0%, #163d56 72%, #176a70 100%);
  color: #fff; padding: 1.65rem 2rem; border-radius: 18px; margin-bottom: 1.25rem;
  box-shadow: 0 12px 30px rgba(16,36,58,.15);
}
.hero h1 { color: #fff; font-size: 2rem; margin: 0 0 .25rem 0; letter-spacing: -.03em; }
.hero p { color: #d6e5ec; margin: 0; font-size: 1.02rem; }
.section-title { color: var(--navy); font-size: 1.2rem; font-weight: 700; margin: 1.1rem 0 .65rem; }
[data-testid="stVerticalBlockBorderWrapper"] {
  background: var(--surface); border-color: var(--line); border-radius: 15px;
  box-shadow: 0 4px 14px rgba(20,40,60,.04);
}
[data-testid="stMetricLabel"] { color: var(--muted); font-weight: 600; }
[data-testid="stMetricValue"] { color: var(--navy); }
[data-testid="stRadio"] > div { gap: .55rem; }
[data-testid="stRadio"] label {
  background: #fff; border: 1px solid var(--line); border-radius: 999px;
  padding: .35rem .85rem; margin: 0;
}
[data-testid="stRadio"] label:has(input:checked) {
  border-color: var(--teal); box-shadow: 0 0 0 1px var(--teal); color: var(--navy);
}
.stButton > button[kind="primary"] {
  background: var(--teal); border: 0; border-radius: 10px; font-weight: 700;
  min-height: 2.8rem; padding: .5rem 1.25rem;
}
.stButton > button[kind="primary"]:hover { background: #0e817f; border: 0; }
.rag-status { background: #fff8e6; border-left: 4px solid #e6a23c; padding: .8rem 1rem; border-radius: 8px; }
.small-note { color: var(--muted); font-size: .88rem; }
</style>
"""
