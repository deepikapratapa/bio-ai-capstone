"""
Bio-Seq LM Explorer  —  Complete Final Version
================================================
Deepika Sarala Pratapa | MS Applied Data Science | University of Florida | Spring 2026

Tabs
----
1. 🧬 DNA Classification       — classify any DNA sequence, 5 paradigms
2. 🔬 Protein Classification   — classify any protein, 5 paradigms, 10 Pfam families
3. 🔍 Sequence Similarity      — find nearest neighbours in embedding space (any sequence)
4. 🌐 Embedding Explorer       — interactive PCA scatter, hover to inspect any training sequence
5. 🧪 Sequence Clustering      — paste multiple sequences, cluster them in embedding space
6. 📊 Results Dashboard        — full paradigm comparison, publication-quality figures
7. ℹ️  About                   — methods, references, infrastructure

New features over previous version
------------------------------------
• Plotly interactive charts (hover, zoom, click) everywhere
• Sequence similarity search — works for ANY sequence, no retraining needed
• Embedding explorer — hover over any of the 4,000 / 2,293 training points
• Sequence clustering — paste ≥2 sequences, cluster with KMeans in embedding space
• In-silico mutagenesis heatmap (Plotly, interactive)
• Alanine scanning (Plotly, interactive)
• Sequence logo (letter height ∝ conservation / importance)
• Download buttons for every result
• 6 curated DNA examples + 5 protein examples with full biological context
"""

from __future__ import annotations
import io, json, math, re, warnings, itertools, textwrap
from pathlib import Path
from collections import Counter

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")

# ── try plotly ────────────────────────────────────────────────────────────────
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# ── try matplotlib as fallback ────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
MODELS    = ROOT / "models"
REPORTS   = ROOT / "reports"

# ─────────────────────────────────────────────────────────────────────────────
# Biological constants
# ─────────────────────────────────────────────────────────────────────────────
PFAM_CLASSES = ["PF00001","PF00046","PF00069","PF00071","PF00076",
                "PF00096","PF01352","PF07686","PF12796","PF13853"]
PFAM_NAMES = {
    "PF00001": "7tm_1 (GPCR family A)",
    "PF00046": "Homeodomain",
    "PF00069": "Protein kinase domain",
    "PF00071": "Ras GTPase",
    "PF00076": "RRM RNA recognition motif",
    "PF00096": "Zinc finger C2H2",
    "PF01352": "KRAB domain",
    "PF07686": "Immunoglobulin V-set",
    "PF12796": "Ankyrin repeat",
    "PF13853": "7tm_4 GPCR olfactory",
}
PFAM_DESC = {
    "PF00001": "G protein-coupled receptors with 7 transmembrane helices. Largest family of cell-surface receptors in the human genome (~800 genes). Targets for ~34% of all approved drugs.",
    "PF00046": "Helix-turn-helix DNA-binding domain found in homeobox transcription factors. Encoded by Hox genes critical for body plan development. Binds specific DNA sequences to regulate gene expression.",
    "PF00069": "Catalytic domain of serine/threonine and tyrosine protein kinases. Phosphorylates target proteins to activate or deactivate signaling cascades. Dysregulation causes cancer; major drug target class.",
    "PF00071": "GTPase domain of Ras superfamily. Molecular switches cycling between GTP (active) and GDP (inactive) states. KRAS mutations are among the most common oncogenic events in human cancer.",
    "PF00076": "RNA recognition motif. Most widespread RNA-binding domain. Binds single-stranded RNA; found in splicing factors, translation regulators, and RNA export machinery.",
    "PF00096": "C2H2 zinc finger domain. Most common DNA-binding domain in eukaryotes (~700 human proteins). Coordinates zinc via Cys-X2-4-Cys-X3-Phe-X5-Leu-X2-His-X3-5-His motif.",
    "PF01352": "KRAB (Krüppel-associated box) repression domain. Present in the largest family of vertebrate transcriptional repressors (~250 human proteins). Recruits KAP1/TRIM28 for silencing.",
    "PF07686": "Immunoglobulin V-set domain. Antigen-binding variable domain of antibodies and T-cell receptors. Also found in cell adhesion molecules, signaling receptors, and immune checkpoint proteins.",
    "PF12796": "Ankyrin repeat. One of the most common protein-protein interaction modules (~40 human proteins with ≥4 repeats). Forms a curved solenoid structure to cradle binding partners.",
    "PF13853": "Olfactory receptor subfamily of GPCRs. Largest gene family in the human genome (~400 functional genes + ~600 pseudogenes). Detects volatile odorants via cAMP-mediated signaling.",
}

# ─────────────────────────────────────────────────────────────────────────────
# Example sequences with full biological context
# ─────────────────────────────────────────────────────────────────────────────
DNA_EXAMPLES = {
    "🟢 Promoter — ACTB (beta-actin, human housekeeping gene)": {
        "seq": "CCGGCTCCGAGCGGGCTGGGGCGGGGAGAGGGCGCGGGGCCAAGTCCGGGCGGAGCGGAGCGAGAGAGGGCGCGGGGCCAAGTCCGGGCGGAGCGGAGCGAGGGCGCGGGGCCAAGTCCGGGCGGAGCTATAAACGCGCGCGCGGGGCCAAGTCCGGGCGGAGCGGAGCGAGAGAGG",
        "info": "Strong CpG island promoter. GC-rich, no TATA box — typical of ubiquitously expressed housekeeping genes. This region drives beta-actin expression in virtually all human cell types.",
    },
    "🟢 Promoter — TP53 (tumour suppressor, TATA-containing)": {
        "seq": "GCGCGAGGCGTGGCGCGGAGGAGCCGCGCGGGAGCGGCGGAGCGGCGGCGGCGGCAGGGCAGGGCCGGGCCCTATAAAGGCGCGGCGGCGGCGGCAGCGGCAGCGGCAGCGGCAGCGGCAGCAGCAGCAGCAGCAGCAG",
        "info": "TP53 promoter contains a TATA-like element and SP1 binding sites. Activated by DNA damage signals. Mutations in TP53 are the most frequent alteration in human cancers.",
    },
    "🟢 Promoter — CpG island (methylation-sensitive)": {
        "seq": "CGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCG",
        "info": "Dense CpG island — high CpG O/E ratio and GC content. When methylated, silences the downstream gene (e.g., tumour suppressor inactivation in cancer). Unmethylated = active transcription.",
    },
    "🔴 Non-promoter — SINE/Alu repetitive element": {
        "seq": "GGCCGGGCGCGGTGGCTCACGCCTGTAATCCCAGCACTTTGGGAGGCCGAGGCGGGCGGATCACGAGGTCAGGAGATCGAGACCATCCTGGCTAACACGGTGAAACCCCGTCTCTACTAAAAATACAAAAAATTAGCCGGGCGTGGTGGCGGGCGCCTGTAGTCCCAGCTACTTGGG",
        "info": "Alu SINE repeat element. These make up ~11% of the human genome. Generally not functional as promoters but can occasionally be exapted as regulatory elements. Low GC, no TATA box, no CpG enrichment.",
    },
    "🔴 Non-promoter — coding exon (BRCA1 exon 11)": {
        "seq": "ATGGATTTATCTGCTCTTCGCGTTGAAGAAGTACAAAATGTCATTAATGCTATGCAGAAAATCTTAGAGTGTCCCATCTGTCTGGAGTTGATCAAGGAACCTGTCTCCACAAAGTGTGACCACATATTTTGCAAATTTTGCATGCTGAAACTTCTCAACCAGAAGAAAGGGCCTTCACA",
        "info": "Coding sequence from BRCA1 exon 11. High information content (protein-coding constraint), but different from promoter features: no CpG enrichment, no TATA, different dinucleotide composition.",
    },
    "🔴 Non-promoter — intronic sequence": {
        "seq": "GTAAGTGCATTTAAAATTAGCAATGATGTAAATAAAGTAAATAAAAGCATGAATAAATGAATAAATGAAGAAATAAATGAATAAATAAGCATGAATAAATAAATAAATGAATAAATAATAAATAATAAATGAATAAATAATAAATAATAAATAATAAATAATAAATAATAAATAATA",
        "info": "Deep intronic sequence with AT-rich composition and simple repeats. No regulatory signals detectable. Typical of non-functional intronic regions far from splice sites.",
    },
}

PROTEIN_EXAMPLES = {
    "🔵 GPCR olfactory receptor (PF13853) — OR1A1": {
        "seq": "MRNHTEITEFILLGLTDDPNFQVVIFVFLLITYMLSITGNLTLITIAKDSHLHTPMYFFLSHLSFVDLSSVSSVPNMLVNLIQDIQPVLGLPCISKFIQFFMEHISLASSVGCLIAMALDRHVAIVHPLLYSTIMSKLACYLLIAASWTLSFVLCVPVFLFQIVH",
        "info": "Human olfactory receptor OR1A1. 7-transmembrane GPCR that detects (-)-trans-rose oxide and related floral volatiles. Located in the olfactory epithelium. Note the high hydrophobic residue content from TM helices.",
    },
    "🟠 Protein kinase (PF00069) — EGFR kinase domain": {
        "seq": "KVLGSGAFGTVYKGLWIPEGEKVKIPVAIKELREATSPKANKEILDEAYVMASVDNPHVCRLLGICLTSTVQLITQLMPFGCLLDYVREHKDNIGSQYLLNWCVQIAKGMNYLEDRRLVHRDLAARNVLVKTPQHVKITDFGLAKLLGAEEKEYHAEGGKVPIKWMALESILHRIYTHQSDVWSYGVTVWELMTFGSKPYDGIPASEISSILEKGERLPQPPICTIDVYMIMVKCWMIDADSRPKFRELIIEFSKMARDPQRYLVIQGDERMHLPSPTDSNFYRALMDEEDMDDVVDADEYLIPQQGFFSSPSTSRTPLLSSLSATSNNSTVACIDRNGLQSCPIKEDSFLQRYSSDPTGALTEDSIDDTFLPVPEYINQSVPKRPAGSVQNPVYHNQPLNPAPSRDPHYQDPHSTAVGNPEYLNTVQPTCVNSTFDSPAHWAQKGSHQISLDNPDYQQDFFPKEAKPNGIFKGSTAENAEYLRVAPQSSEFIGA",
        "info": "EGFR (Epidermal Growth Factor Receptor) kinase domain. Mutated in ~15% of non-small cell lung cancers. Target of gefitinib, erlotinib, osimertinib. Note the conserved DFG motif and activation loop.",
    },
    "🟣 Zinc finger C2H2 (PF00096) — WT1 tumour suppressor": {
        "seq": "MARPYKTELKIVKKTDKKHFKVHQCNACGKRFMRSDNLKKHQKTHSGEKPFKCDICGRGFTQSGNLKRHQKIHTGEKPYKCNECGKSFIQSSDLKRHQRIHTGEKPYQCNECGKSFIQSSHLKRHQRIHTGEKPY",
        "info": "Wilms Tumour protein (WT1) zinc finger domain. Contains four C2H2 zinc fingers that bind GC-rich DNA motifs. Critical for kidney development; mutations cause Wilms tumour (paediatric kidney cancer).",
    },
    "🟡 Ras GTPase (PF00071) — KRAS oncogene": {
        "seq": "MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSY RKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRT GEGFLCVFAINNTKSFEDIHHQRQEIKRVKDSEDVPMVLVGNKCDLAARTVESRQAQDLARSYGIPYIETSAKTRQHVREVDRE",
        "info": "KRAS proto-oncogene. G12D/G12V/G12C mutations found in ~25% of all human cancers (pancreatic: 90%, colon: 45%, lung: 30%). The G12C mutation is targeted by sotorasib (first direct KRAS inhibitor). Note the P-loop (GXXXXGKS) and switch regions.",
    },
    "🟢 Immunoglobulin V-set (PF07686) — antibody VH domain": {
        "seq": "QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYGISWVRQAPGQGLEWMGWISAYNGNTNYAQKLQGRVTMTTDTSTSTAYMELRSLRSDDTAVYYCARVDYYGSGSYFDYWGQGTLVTVSS",
        "info": "Heavy chain variable (VH) domain from a human IgG antibody. Contains three complementarity-determining regions (CDR1, CDR2, CDR3) that directly contact antigen. The beta-sandwich Ig fold is one of the most common structural domains in the immune system.",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bio-Seq LM Explorer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS — dark premium UI
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ═══════════════════════════════════════════════════════════════
   BIOSEQ LM — PREMIUM DARK UI
   Font stack: Inter (Google Fonts injected below)
═══════════════════════════════════════════════════════════════ */

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root & global ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: #050b18 !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 80% 60% at 20% -10%, rgba(29,78,216,.18) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 110%, rgba(139,92,246,.12) 0%, transparent 60%),
        #050b18 !important;
    min-height: 100vh;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #070f22 0%, #050b18 100%) !important;
    border-right: 1px solid rgba(255,255,255,.06) !important;
    box-shadow: 4px 0 24px rgba(0,0,0,.4) !important;
}
[data-testid="stSidebar"] > div { padding-top: 0 !important; }
[data-testid="stSidebar"] * { color: #c8d6ef !important; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.08) !important; }

/* ── Main content padding ── */
[data-testid="stMainBlockContainer"] { padding: 1.5rem 2rem !important; }
[data-testid="block-container"] { max-width: 1400px; margin: 0 auto; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, rgba(29,78,216,.2) 0%, rgba(5,11,24,.8) 100%) !important;
    border: 1px solid rgba(59,130,246,.2) !important;
    border-radius: 16px !important;
    padding: 20px !important;
    transition: border-color .2s, box-shadow .2s;
}
[data-testid="metric-container"]:hover {
    border-color: rgba(59,130,246,.45) !important;
    box-shadow: 0 0 20px rgba(59,130,246,.12) !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #e2e8f0 !important; font-weight: 800 !important; font-size: 1.6rem !important;
}
[data-testid="metric-container"] [data-testid="stMetricLabel"] {
    color: #64748b !important; font-size: .78rem !important; font-weight: 500 !important;
    text-transform: uppercase; letter-spacing: .06em;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    color: #10b981 !important; font-weight: 600 !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] [role="tablist"] {
    background: rgba(255,255,255,.04) !important;
    border: 1px solid rgba(255,255,255,.07) !important;
    border-radius: 14px !important;
    padding: 5px !important;
    gap: 3px !important;
    backdrop-filter: blur(12px);
}
[data-testid="stTabs"] [role="tab"] {
    color: #64748b !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    padding: 8px 18px !important;
    transition: all .18s ease !important;
    border: none !important;
    letter-spacing: .01em;
}
[data-testid="stTabs"] [role="tab"]:hover {
    color: #94a3b8 !important;
    background: rgba(255,255,255,.05) !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, #1e3a8a 0%, #1d4ed8 100%) !important;
    color: #fff !important;
    box-shadow: 0 2px 12px rgba(29,78,216,.4) !important;
}

/* ── Primary buttons ── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    padding: .65rem 2rem !important;
    letter-spacing: .02em !important;
    box-shadow: 0 4px 20px rgba(29,78,216,.45), inset 0 1px 0 rgba(255,255,255,.15) !important;
    transition: all .18s ease !important;
}
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%) !important;
    box-shadow: 0 6px 28px rgba(59,130,246,.55), inset 0 1px 0 rgba(255,255,255,.15) !important;
    transform: translateY(-1px) !important;
}
.stButton > button[kind="primary"]:active { transform: translateY(0) !important; }

/* Secondary buttons */
.stButton > button:not([kind="primary"]) {
    background: rgba(255,255,255,.05) !important;
    color: #94a3b8 !important;
    border: 1px solid rgba(255,255,255,.1) !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    padding: .45rem 1.2rem !important;
    transition: all .15s ease !important;
}
.stButton > button:not([kind="primary"]):hover {
    background: rgba(255,255,255,.09) !important;
    border-color: rgba(59,130,246,.4) !important;
    color: #e2e8f0 !important;
}

/* ── Text areas ── */
.stTextArea textarea {
    background: rgba(255,255,255,.03) !important;
    border: 1px solid rgba(255,255,255,.1) !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    font-family: 'JetBrains Mono', 'Courier New', monospace !important;
    font-size: 12px !important;
    line-height: 1.65 !important;
    letter-spacing: .02em !important;
    transition: border-color .18s, box-shadow .18s !important;
    padding: 12px 14px !important;
}
.stTextArea textarea:focus {
    border-color: rgba(59,130,246,.55) !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,.12), 0 0 20px rgba(59,130,246,.08) !important;
    background: rgba(255,255,255,.05) !important;
    outline: none !important;
}
.stTextArea label { color: #64748b !important; font-weight: 600 !important; font-size: 12px !important;
                    text-transform: uppercase; letter-spacing: .06em; margin-bottom: 6px !important; }

/* ── Selectbox ── */
.stSelectbox [data-baseweb="select"] > div {
    background: rgba(255,255,255,.04) !important;
    border: 1px solid rgba(255,255,255,.1) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    transition: border-color .15s !important;
}
.stSelectbox [data-baseweb="select"]:hover > div {
    border-color: rgba(59,130,246,.4) !important;
}
.stSelectbox label { color: #64748b !important; font-weight: 600 !important;
                     font-size: 12px !important; text-transform: uppercase; letter-spacing: .06em; }

/* ── Radio ── */
[data-testid="stRadio"] label { color: #94a3b8 !important; }
[data-testid="stRadio"] [data-testid="stMarkdownContainer"] p { color: #94a3b8 !important; }

/* ── Slider ── */
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
    background: #3b82f6 !important;
    border: 2px solid #1d4ed8 !important;
}
[data-testid="stSlider"] label { color: #64748b !important; font-weight: 600 !important;
                                  font-size: 12px !important; text-transform: uppercase; }

/* ── Checkbox ── */
[data-testid="stCheckbox"] label { color: #94a3b8 !important; font-size: 13px !important; }

/* ── DataFrames ── */
[data-testid="stDataFrame"] {
    border-radius: 14px !important;
    overflow: hidden !important;
    border: 1px solid rgba(255,255,255,.08) !important;
    box-shadow: 0 4px 24px rgba(0,0,0,.3) !important;
}
[data-testid="stDataFrame"] table { background: rgba(255,255,255,.02) !important; }

/* ── Alerts ── */
[data-testid="stAlert"] {
    border-radius: 12px !important;
    border-left-width: 4px !important;
    font-weight: 500 !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] { color: #3b82f6 !important; }

/* ── Plotly chart container ── */
[data-testid="stPlotlyChart"] {
    border-radius: 14px !important;
    overflow: hidden !important;
    border: 1px solid rgba(255,255,255,.07) !important;
    box-shadow: 0 4px 24px rgba(0,0,0,.35) !important;
}

/* ── Pyplot chart container ── */
[data-testid="stImage"] img, .stPyplot {
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,.07) !important;
}

/* ════════════════════════
   CUSTOM COMPONENT CLASSES
════════════════════════ */

/* Hero banner */
.hero-banner {
    background:
        linear-gradient(135deg, rgba(29,78,216,.25) 0%, rgba(139,92,246,.15) 50%, rgba(5,11,24,.9) 100%),
        radial-gradient(ellipse 100% 200% at 0% 50%, rgba(59,130,246,.1) 0%, transparent 70%);
    border: 1px solid rgba(59,130,246,.2);
    border-radius: 20px;
    padding: 28px 32px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute; top: 0; right: 0; width: 40%; height: 100%;
    background: radial-gradient(ellipse 80% 80% at 80% 50%, rgba(139,92,246,.08) 0%, transparent 70%);
    pointer-events: none;
}
.hero-banner h2 {
    color: #e2e8f0; margin: 0 0 8px; font-size: 22px; font-weight: 800;
    letter-spacing: -.02em;
}
.hero-banner p { color: #64748b; margin: 0; font-size: 14px; line-height: 1.7; max-width: 680px; }
.hero-banner .hero-badge {
    display: inline-block; background: rgba(59,130,246,.15); border: 1px solid rgba(59,130,246,.3);
    color: #93c5fd; font-size: 11px; font-weight: 700; letter-spacing: .08em;
    text-transform: uppercase; padding: 4px 12px; border-radius: 20px; margin-bottom: 10px;
}

/* Section headers */
.section-header {
    display: flex; align-items: center; gap: 10px;
    margin: 22px 0 12px; padding: 0;
}
.section-header-line {
    flex: 1; height: 1px;
    background: linear-gradient(90deg, rgba(59,130,246,.3), transparent);
}
.section-header h3 {
    color: #93c5fd; margin: 0; font-size: 15px; font-weight: 700;
    letter-spacing: .04em; text-transform: uppercase; white-space: nowrap;
}

/* Bio cards */
.bio-card {
    background: linear-gradient(135deg, rgba(255,255,255,.04) 0%, rgba(5,11,24,.6) 100%);
    border: 1px solid rgba(255,255,255,.08);
    border-radius: 16px;
    padding: 20px 22px;
    margin: 6px 0;
    transition: border-color .18s, box-shadow .18s;
    position: relative; overflow: hidden;
}
.bio-card:hover {
    border-color: rgba(59,130,246,.25);
    box-shadow: 0 4px 24px rgba(0,0,0,.3);
}
.bio-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(59,130,246,.3), transparent);
}
.bio-card h4 { color: #60a5fa; margin: 0 0 8px; font-size: 14px; font-weight: 700; letter-spacing: .01em; }
.bio-card p  { color: #64748b; margin: 0; font-size: 13px; line-height: 1.65; }

/* Prediction result cards */
.pred-card {
    background: linear-gradient(135deg, rgba(255,255,255,.04), rgba(5,11,24,.7));
    border: 1px solid rgba(255,255,255,.08);
    border-radius: 16px; padding: 18px 20px; margin: 4px 0;
    transition: all .2s ease;
}
.pred-card:hover { border-color: rgba(59,130,246,.3); box-shadow: 0 8px 32px rgba(0,0,0,.4); }

/* Prediction badges */
.badge-promoter {
    display: inline-flex; align-items: center; gap: 6px;
    background: linear-gradient(135deg, rgba(16,185,129,.15), rgba(5,150,105,.1));
    border: 1px solid rgba(16,185,129,.35); border-radius: 24px;
    color: #6ee7b7; font-size: 12px; font-weight: 800;
    padding: 5px 14px; letter-spacing: .06em; text-transform: uppercase;
}
.badge-nonpromoter {
    display: inline-flex; align-items: center; gap: 6px;
    background: linear-gradient(135deg, rgba(239,68,68,.15), rgba(185,28,28,.1));
    border: 1px solid rgba(239,68,68,.35); border-radius: 24px;
    color: #fca5a5; font-size: 12px; font-weight: 800;
    padding: 5px 14px; letter-spacing: .06em; text-transform: uppercase;
}
.badge-protein {
    display: inline-flex; align-items: center; gap: 6px;
    background: linear-gradient(135deg, rgba(99,102,241,.15), rgba(67,56,202,.1));
    border: 1px solid rgba(99,102,241,.35); border-radius: 24px;
    color: #a5b4fc; font-size: 12px; font-weight: 800;
    padding: 5px 14px; letter-spacing: .02em;
}

/* Confidence values */
.conf-val { font-size: 28px; font-weight: 900; letter-spacing: -.02em; line-height: 1; }
.conf-high { color: #10b981; }
.conf-mid  { color: #f59e0b; }
.conf-low  { color: #ef4444; }
.conf-unit { color: #334155; font-size: 13px; font-weight: 500; margin-left: 2px; }

/* Paradigm label */
.paradigm-label {
    color: #475569; font-size: 11px; font-weight: 700;
    text-transform: uppercase; letter-spacing: .08em; margin-bottom: 8px;
}
.approx-tag {
    background: rgba(245,158,11,.1); border: 1px solid rgba(245,158,11,.25);
    color: #fbbf24; font-size: 10px; font-weight: 600; padding: 2px 8px;
    border-radius: 10px; letter-spacing: .04em; margin-left: 6px;
}

/* Sequence info box */
.seq-info {
    background: rgba(255,255,255,.03);
    border: 1px solid rgba(255,255,255,.07);
    border-left: 3px solid rgba(59,130,246,.5);
    border-radius: 0 10px 10px 0;
    padding: 10px 16px; margin: 8px 0;
    font-size: 12.5px; color: #64748b; line-height: 1.6;
}

/* Stat chips */
.stat-chip {
    display: inline-block;
    background: rgba(59,130,246,.1); border: 1px solid rgba(59,130,246,.2);
    color: #93c5fd; font-size: 11px; font-weight: 700;
    padding: 4px 12px; border-radius: 20px; margin: 2px;
    letter-spacing: .04em;
}

/* Sidebar brand */
.sidebar-brand {
    padding: 20px 16px 12px;
    border-bottom: 1px solid rgba(255,255,255,.06);
    margin-bottom: 16px;
}
.sidebar-title {
    font-size: 32px; font-weight: 900; letter-spacing: -.02em;
    background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #34d399 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    line-height: 1.2; margin-bottom: 4px;
}
.sidebar-sub {
    font-size: 14px !important; color: #CBCBCB !important; line-height: 1.5;
}

/* Sidebar section label */
.sidebar-section {
    font-size: 10px !important; font-weight: 700 !important;
    text-transform: uppercase; letter-spacing: .1em;
    color: #334155 !important; margin: 14px 0 6px !important;
}

/* Result table highlight */
.result-best { background: rgba(16,185,129,.08) !important; }

/* Divider */
.divider {
    height: 1px; margin: 20px 0;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,.08), transparent);
}

/* Info pill row */
.info-pills { display: flex; gap: 8px; flex-wrap: wrap; margin: 10px 0; }
.info-pill {
    background: rgba(255,255,255,.05); border: 1px solid rgba(255,255,255,.09);
    border-radius: 20px; padding: 4px 14px;
    font-size: 12px; color: #64748b; font-weight: 500;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Plotly dark template
# ─────────────────────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="#0d1117",
    plot_bgcolor="#0d1117",
    font=dict(color="#c9d1d9", family="Inter, Arial, sans-serif"),
    margin=dict(l=10, r=10, t=40, b=10),
)

# ─────────────────────────────────────────────────────────────────────────────
# Matplotlib dark (fallback)
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":"#0d1117","axes.facecolor":"#0d1117",
    "axes.edgecolor":"#30363d","axes.labelcolor":"#c9d1d9",
    "xtick.color":"#8b949e","ytick.color":"#8b949e",
    "text.color":"#c9d1d9","grid.color":"#21262d","figure.dpi":120,
})

# ─────────────────────────────────────────────────────────────────────────────
# Cached model loaders
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading DNA baseline models…")
def load_dna_baseline():
    d = MODELS/"dna"/"baselines"
    out = {}
    for n,f in [("LogReg","logreg_len200.pkl"),
                ("Random Forest","random_forest_len200.pkl"),
                ("XGBoost","xgboost_len200.pkl"),
                ("Linear SVM","linear_svm_calibrated_len200.pkl")]:
        p = d/f
        if p.exists(): out[n] = joblib.load(p)
    return out

@st.cache_resource(show_spinner="Loading DNABERT-2 classifier…")
def load_dnabert2_clf():
    p = MODELS/"dna"/"dnabert2"/"xgboost.pkl"
    return joblib.load(p) if p.exists() else None

@st.cache_resource(show_spinner="Loading NT-500M classifier…")
def load_nt_clf():
    p = MODELS/"dna"/"nt"/"xgboost.pkl"
    return joblib.load(p) if p.exists() else None

@st.cache_resource(show_spinner="Loading protein baseline models…")
def load_protein_baseline():
    d = MODELS/"protein"/"baselines"
    out = {}
    for n,f in [("LogReg","logreg_top10_per400.pkl"),
                ("Random Forest","random_forest_top10_per400.pkl"),
                ("XGBoost","xgboost_top10_per400.pkl"),
                ("Linear SVM","linear_svm_calibrated_top10_per400.pkl")]:
        p = d/f
        if p.exists():
            obj = joblib.load(p)
            out[n] = obj["model"] if isinstance(obj,dict) and "model" in obj else obj
    return out

@st.cache_resource(show_spinner="Loading ESM-2 classifier…")
def load_esm2_clf():
    d = MODELS/"protein"/"esm2"
    for f in ["linear_svm_esm2.pkl","logreg_esm2.pkl"]:
        p = d/f
        if p.exists(): return joblib.load(p)
    return None

@st.cache_resource(show_spinner="Loading ProtBERT classifier…")
def load_protbert_clf():
    p = MODELS/"protein"/"protbert"/"linear_svm_calibrated.pkl"
    return joblib.load(p) if p.exists() else None

@st.cache_data(show_spinner="Loading DNA embeddings…")
def load_dna_embs():
    sfx = "len200_pos2000_neg2000"
    out = {}
    for k in ["dnabert2","nt"]:
        ep = PROCESSED/f"dna_{k}_embeddings_{sfx}.npy"
        lp = PROCESSED/f"dna_{k}_labels_{sfx}.npy"
        ip = PROCESSED/f"dna_{k}_ids_{sfx}.npy"
        if ep.exists() and lp.exists():
            out[k] = {
                "X": np.load(ep),
                "y": np.load(lp),
                "ids": np.load(ip,allow_pickle=True) if ip.exists() else None,
            }
    return out

@st.cache_data(show_spinner="Loading protein embeddings…")
def load_prot_embs():
    sfx = "top10_per400"
    out = {}
    for k in ["esm2","protbert"]:
        ep = PROCESSED/f"protein_{k}_embeddings_{sfx}.npy"
        lp = PROCESSED/f"protein_{k}_labels_{sfx}.npy"
        if ep.exists() and lp.exists():
            out[k] = {"X":np.load(ep),"y":np.load(lp)}
    return out

@st.cache_data(show_spinner="Loading DNA training sequences…")
def load_dna_seqs():
    p = PROCESSED/"dna_promoter_vs_nonpromoter_len200_pos2000_neg2000.csv"
    return pd.read_csv(p) if p.exists() else None

@st.cache_data(show_spinner="Loading protein training sequences…")
def load_prot_seqs():
    p = PROCESSED/"protein_uniprot_pfam_top10_per400.csv"
    return pd.read_csv(p) if p.exists() else None

@st.cache_data
def load_comparison():
    dp = REPORTS/"dna_final_comparison.csv"
    pp = REPORTS/"protein_final_comparison.csv"
    return (pd.read_csv(dp) if dp.exists() else None,
            pd.read_csv(pp) if pp.exists() else None)

@st.cache_data(show_spinner="Pre-computing PCA…")
def precompute_pca(key: str, task: str):
    embs = load_dna_embs() if task=="dna" else load_prot_embs()
    if key not in embs: return None, None, None, None
    X,y = embs[key]["X"], embs[key]["y"]
    sc  = StandardScaler().fit(X)
    Xs  = sc.transform(X)
    pca = PCA(n_components=2,random_state=42).fit(Xs)
    coords = pca.transform(Xs)
    return coords, y, sc, pca

# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering — DNA
# ─────────────────────────────────────────────────────────────────────────────
def dna_features(seq: str) -> np.ndarray:
    s=seq.upper(); L=len(s); c=Counter(s)
    def sd(n,d): return n/d if d else 0.
    gc=sd(c["G"]+c["C"],L); at=sd(c["A"]+c["T"],L)
    fA,fC,fG,fT,fN=[sd(c[b],L) for b in "ACGTN"]
    gc_sk=sd(c["G"]-c["C"],c["G"]+c["C"]); at_sk=sd(c["A"]-c["T"],c["A"]+c["T"])
    cpg=s.count("CG"); cpg_d=sd(cpg,L-1); cpg_oe=sd(cpg*L,max(c["C"],1)*max(c["G"],1))
    valid=[x for x in s if x in "ACGT"]
    ent=0.
    if valid:
        fr={b:valid.count(b)/len(valid) for b in "ACGT"}
        ent=-sum(f*math.log2(f) for f in fr.values() if f>0)
    best=cur=1
    for i in range(1,len(s)):
        if s[i]==s[i-1]: cur+=1; best=max(best,cur)
        else: cur=1
    di=["".join(d) for d in itertools.product("ACGT",repeat=2)]
    tri=["".join(t) for t in itertools.product("ACGT",repeat=3)]
    dif=[sd(s.count(d),max(L-1,1)) for d in di]
    trif=[sd(s.count(t),max(L-2,1)) for t in tri]
    per=sd(s.count("AA")+s.count("TT")+s.count("TA"),max(L-1,1))
    cpg_r=1. if gc>0.5 and cpg_oe>0.6 else 0.
    feats=[fA,fC,fG,fT,fN,gc,at,gc_sk,at_sk,cpg_d,cpg_oe,ent,float(best),per,cpg_r]+dif+trif
    return np.array(feats,dtype=np.float32).reshape(1,-1)

# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering — Protein (exact 176 features)
# ─────────────────────────────────────────────────────────────────────────────
def prot_features(seq: str) -> np.ndarray:
    from itertools import product as iproduct
    seq=seq.strip().upper(); L=max(len(seq),1); c=Counter(seq)
    AA=list("ACDEFGHIKLMNPQRSTVWY"); feats={}
    for a in AA: feats[f"frac_{a}"]=float(c[a]/L)
    GRP={"hydrophobic":set("AILMFWVY"),"polar":set("STNQCY"),
         "positive":set("KRH"),"negative":set("DE"),"aromatic":set("FWY"),
         "aliphatic":set("AILV"),"small":set("AGSTP"),"sulfur":set("CM"),"amide":set("NQ")}
    for g,aset in GRP.items(): feats[f"frac_group_{g}"]=float(sum(c[a] for a in aset)/L)
    ent=0.
    for a in AA:
        if c[a]>0: p=c[a]/L; ent-=p*math.log2(p)
    feats["aa_entropy"]=float(ent)
    best=cur=1
    for i in range(1,len(seq)):
        if seq[i]==seq[i-1]: cur+=1; best=max(best,cur)
        else: cur=1
    feats["max_homopolymer_run"]=float(best); feats["n_unique_aas"]=float(len(set(seq)))
    try:
        from Bio.SeqUtils.ProtParam import ProteinAnalysis
        cl=re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]","A",seq); pa=ProteinAnalysis(cl)
        feats["molecular_weight"]=float(pa.molecular_weight())
        feats["aromaticity"]=float(pa.aromaticity())
        feats["instability_index"]=float(pa.instability_index())
        feats["isoelectric_point"]=float(pa.isoelectric_point())
        feats["gravy"]=float(pa.gravy()); feats["charge_ph7"]=float(pa.charge_at_pH(7.0))
        h,t,s2=pa.secondary_structure_fraction()
        feats["secstruct_helix"]=float(h); feats["secstruct_turn"]=float(t); feats["secstruct_sheet"]=float(s2)
    except Exception:
        for k in ["molecular_weight","aromaticity","instability_index","isoelectric_point",
                  "gravy","charge_ph7","secstruct_helix","secstruct_turn","secstruct_sheet"]: feats[k]=0.
    CTD={"hydrophobicity_3":{"H":set("AILMFWV"),"P":set("CNQSTY"),"N":set("DEGHRKP")},
         "polarity_2":{"P":set("STNQCYW"),"N":set("ADFGHIKLMPVR")},
         "charge_3":{"+":set("KRH"),"-":set("DE"),"0":set("ACFGILMNPQSTVWY")}}
    for prop,gdef in CTD.items():
        inv={a:g for g,aset in gdef.items() for a in aset}
        gs="".join(inv.get(ch,list(gdef.keys())[0]) for ch in seq)
        syms=sorted(gdef.keys()); gL=max(len(gs),1); gc2=Counter(gs)
        for s2 in syms: feats[f"{prop}__ctd_comp_{s2}"]=float(gc2.get(s2,0)/gL)
        trans=Counter()
        for i in range(len(gs)-1):
            a2,b2=gs[i],gs[i+1]
            if a2!=b2: trans["".join(sorted([a2,b2]))]+=1
        dn=max(len(gs)-1,1)
        for i,a2 in enumerate(syms):
            for b2 in syms[i+1:]: feats[f"{prop}__ctd_trans_{a2}{b2}"]=float(trans.get("".join(sorted([a2,b2])),0)/dn)
        for s2 in syms:
            idxs=[i+1 for i,ch in enumerate(gs) if ch==s2]
            if not idxs:
                for q in [1,25,50,75,100]: feats[f"{prop}__ctd_dist_{s2}_{q}"]=0.; continue
            n2=len(idxs)
            picks=[idxs[0],idxs[int(math.ceil(.25*n2))-1],idxs[int(math.ceil(.5*n2))-1],
                   idxs[int(math.ceil(.75*n2))-1],idxs[-1]]
            for q,pos in zip([1,25,50,75,100],picks): feats[f"{prop}__ctd_dist_{s2}_{q}"]=float(pos/gL)
    PSEAAC_L=10; PSEAAC_W=0.05
    HP={"A":0.62,"C":0.29,"D":-0.90,"E":-0.74,"F":1.19,"G":0.48,"H":-0.40,
        "I":1.38,"K":-1.50,"L":1.06,"M":0.64,"N":-0.78,"P":0.12,"Q":-0.85,
        "R":-2.53,"S":-0.18,"T":-0.05,"V":1.08,"W":0.81,"Y":0.26}
    thetas=[]
    for lag in range(1,PSEAAC_L+1):
        if len(seq)>lag: th=sum((HP.get(seq[i],0)-HP.get(seq[i+lag],0))**2 for i in range(len(seq)-lag))/max(len(seq)-lag,1)
        else: th=0.
        thetas.append(th)
    dp2=sum(c[a] for a in AA)+PSEAAC_W*sum(thetas); dp2=max(dp2,1e-9)
    for a in AA: feats[f"pse_aac_{a}"]=float(c[a]/dp2)
    for i,th in enumerate(thetas,1): feats[f"pse_theta_{i}"]=float(PSEAAC_W*th/dp2)
    RED7={"A":"A","G":"A","V":"A","I":"B","L":"B","F":"B","P":"B","Y":"C","M":"C",
          "T":"C","S":"C","H":"D","N":"D","Q":"D","W":"D","R":"E","K":"E","D":"F","E":"F","C":"G"}
    RS=sorted(set(RED7.values())); rs="".join(RED7.get(ch,"A") for ch in seq)
    rd=max(len(rs)-1,1); rdc=Counter(rs[i:i+2] for i in range(len(rs)-1))
    for a2,b2 in iproduct(RS,RS): feats[f"red7_di_{a2}{b2}"]=float(rdc.get(a2+b2,0)/rd)
    ORDER=["seq_len",*[f"frac_{a}" for a in "ACDEFGHIKLMNPQRSTVWY"],
           *[f"frac_group_{g}" for g in ["hydrophobic","polar","positive","negative","aromatic","aliphatic","small","sulfur","amide"]],
           "aa_entropy","max_homopolymer_run","n_unique_aas",
           "molecular_weight","aromaticity","instability_index","isoelectric_point","gravy","charge_ph7",
           "secstruct_helix","secstruct_turn","secstruct_sheet",
           "hydrophobicity_3__ctd_comp_H","hydrophobicity_3__ctd_comp_P","hydrophobicity_3__ctd_comp_N",
           "hydrophobicity_3__ctd_trans_HP","hydrophobicity_3__ctd_trans_HN","hydrophobicity_3__ctd_trans_PN",
           *[f"hydrophobicity_3__ctd_dist_H_{q}" for q in [1,25,50,75,100]],
           *[f"hydrophobicity_3__ctd_dist_P_{q}" for q in [1,25,50,75,100]],
           *[f"hydrophobicity_3__ctd_dist_N_{q}" for q in [1,25,50,75,100]],
           "polarity_2__ctd_comp_P","polarity_2__ctd_comp_N","polarity_2__ctd_trans_PN",
           *[f"polarity_2__ctd_dist_P_{q}" for q in [1,25,50,75,100]],
           *[f"polarity_2__ctd_dist_N_{q}" for q in [1,25,50,75,100]],
           "charge_3__ctd_comp_+","charge_3__ctd_comp_-","charge_3__ctd_comp_0",
           "charge_3__ctd_trans_+-","charge_3__ctd_trans_+0","charge_3__ctd_trans_-0",
           *[f"charge_3__ctd_dist_+_{q}" for q in [1,25,50,75,100]],
           *[f"charge_3__ctd_dist_-_{q}" for q in [1,25,50,75,100]],
           *[f"charge_3__ctd_dist_0_{q}" for q in [1,25,50,75,100]],
           *[f"pse_aac_{a}" for a in "ACDEFGHIKLMNPQRSTVWY"],
           *[f"pse_theta_{i}" for i in range(1,11)],
           *[f"red7_di_{a}{b}" for a in "ABCDEFG" for b in "ABCDEFG"]]
    feats["seq_len"]=float(L)
    return np.array([feats.get(k,0.) for k in ORDER],dtype=np.float32).reshape(1,-1)

# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────
def val_dna(seq):
    s=seq.strip().upper().replace(" ","").replace("\n","")
    bad=set(s)-set("ACGTN")
    if bad: return None,f"Invalid characters: {bad}"
    if len(s)<50: return None,"Sequence too short (min 50 bp)"
    if len(s)>1000: return None,"Sequence too long (max 1000 bp)"
    return (s+"N"*(200-len(s)))[:200], None

def val_prot(seq):
    s=seq.strip().upper().replace(" ","").replace("\n","")
    bad=set(s)-set("ACDEFGHIKLMNPQRSTVWYXBZUO")
    if bad: return None,f"Invalid amino acids: {bad}"
    if len(s)<20: return None,"Too short (min 20 aa)"
    if len(s)>1024: return None,"Too long (max 1024 aa)"
    return s,None

# ─────────────────────────────────────────────────────────────────────────────
# Nearest neighbour search (cosine similarity)
# ─────────────────────────────────────────────────────────────────────────────
def nn_search(query: np.ndarray, X: np.ndarray, y: np.ndarray, k=5):
    q=query.flatten()
    qn=q/(np.linalg.norm(q)+1e-9)
    Xn=X/(np.linalg.norm(X,axis=1,keepdims=True)+1e-9)
    sims=Xn@qn
    idx=np.argsort(sims)[-k:][::-1]
    return idx, sims[idx], y[idx]

# ─────────────────────────────────────────────────────────────────────────────
# Project new point into existing PCA space
# ─────────────────────────────────────────────────────────────────────────────
def project_into_pca(emb: np.ndarray, X_train: np.ndarray):
    sc  = StandardScaler().fit(X_train)
    pca = PCA(n_components=2,random_state=42).fit(sc.transform(X_train))
    return pca.transform(sc.transform(emb.reshape(1,-1)))[0]

# ─────────────────────────────────────────────────────────────────────────────
# In-silico mutagenesis
# ─────────────────────────────────────────────────────────────────────────────
def dna_mutagenesis(seq, clf, n=80):
    s=(seq+"N"*(200-len(seq)))[:200]; bases=list("ACGT")
    base_p=clf.predict_proba(dna_features(s))[0,1]
    deltas=np.zeros(min(len(seq),n))
    for i in range(min(len(seq),n)):
        orig=s[i] if s[i] in bases else "N"
        alts=[b for b in bases if b!=orig]
        if not alts: continue
        sc=[]; 
        for alt in alts:
            m=list(s); m[i]=alt
            p=clf.predict_proba(dna_features("".join(m)))[0,1]
            sc.append(p-base_p)
        deltas[i]=float(np.mean(sc))
    return deltas

def ala_scan(seq, clf, n=60):
    s=seq[:n]; base_pr=clf.predict_proba(prot_features(s))[0]
    bc=int(np.argmax(base_pr)); bconf=base_pr[bc]
    deltas=np.zeros(len(s))
    for i in range(len(s)):
        alt="G" if s[i]=="A" else "A"
        m=list(s); m[i]=alt
        p=clf.predict_proba(prot_features("".join(m)))[0,bc]
        deltas[i]=float(p-bconf)
    return deltas

# ─────────────────────────────────────────────────────────────────────────────
# Plotly interactive PCA
# ─────────────────────────────────────────────────────────────────────────────
ACCENT = [
    "#3b82f6",  # blue
    "#8b5cf6",  # purple
    "#10b981",  # green
    "#f59e0b",  # amber
    "#ef4444",  # red
    "#06b6d4",  # cyan
    "#ec4899",  # pink
    "#84cc16",  # lime
    "#f97316",  # orange
    "#14b8a6",  # teal
]

def plotly_pca(coords, y, query_pt, class_labels, nn_idx=None,
               title="", hover_texts=None):
    fig = go.Figure()
    unique = np.unique(y)
    for i, lbl in enumerate(unique):
        mask = y == lbl
        name = class_labels[lbl] if lbl < len(class_labels) else str(lbl)
        ht = [hover_texts[j] for j in np.where(mask)[0]] if hover_texts is not None else None
        fig.add_trace(go.Scatter(
            x=coords[mask, 0],
            y=coords[mask, 1],
            mode="markers",
            name=name,
            marker=dict(color=ACCENT[i % len(ACCENT)], size=5, opacity=0.4,
                        line=dict(width=0)),
            text=ht,
            hovertemplate="%{text}<extra>" + name + "</extra>" if ht else None,
        ))
    if nn_idx is not None:
        fig.add_trace(go.Scatter(
            x=coords[nn_idx, 0],
            y=coords[nn_idx, 1],
            mode="markers",
            name="Nearest neighbours",
            marker=dict(color="#facc15", size=10, symbol="diamond",
                        line=dict(color="white", width=1)),
        ))
    if query_pt is not None:
        fig.add_trace(go.Scatter(
            x=[query_pt[0]],
            y=[query_pt[1]],
            mode="markers",
            name="Your sequence",
            marker=dict(color="#ff4444", size=18, symbol="star",
                        line=dict(color="white", width=1.5)),
        ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=title,
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#30363d", borderwidth=1),
        height=480,
    )
    fig.update_xaxes(gridcolor="#21262d", zerolinecolor="#30363d")
    fig.update_yaxes(gridcolor="#21262d", zerolinecolor="#30363d")
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# Plotly mutagenesis heatmap
# ─────────────────────────────────────────────────────────────────────────────
def plotly_mutagenesis(deltas, seq, title, ylabel="Δ probability"):
    L = len(deltas)
    seq = seq[:L]
    vmax = max(abs(deltas).max(), 0.001)
    colors = []
    for d in deltas:
        if d > 0:
            colors.append(f"rgba(16,185,129,{min(abs(d)/vmax,1):.2f})")
        else:
            colors.append(f"rgba(239,68,68,{min(abs(d)/vmax,1):.2f})")
    hover = [f"Position {i+1}<br>Nucleotide: {seq[i]}<br>Δ = {d:+.4f}" for i, d in enumerate(deltas)]
    fig = go.Figure(go.Bar(
        x=list(range(1, L + 1)),
        y=deltas,
        marker_color=colors,
        text=list(seq),
        textposition="outside",
        customdata=hover,
        hovertemplate="%{customdata}<extra></extra>",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=title,
        xaxis_title="Position",
        yaxis_title=ylabel,
        height=320,
    )
    fig.update_xaxes(gridcolor="#21262d", zerolinecolor="#30363d")
    fig.update_yaxes(
        gridcolor="#21262d",
        zeroline=True,
        zerolinecolor="#f59e0b",
        zerolinewidth=1.5,
    )
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# Plotly confidence bar
# ─────────────────────────────────────────────────────────────────────────────
def plotly_conf_bar(names, probs, threshold=0.5, title=""):
    colors = ["#10b981" if p >= threshold else "#ef4444" for p in probs]
    fig = go.Figure(go.Bar(
        x=probs,
        y=names,
        orientation="h",
        marker_color=colors,
        marker_line_width=0,
        text=[f"{p*100:.1f}%" for p in probs],
        textposition="auto",
        hovertemplate="%{y}: %{x:.3f}<extra></extra>",
    ))
    if threshold is not None:
        fig.add_vline(x=threshold, line_color="#f59e0b", line_dash="dash", line_width=1.5)
    fig.update_layout(**PLOTLY_LAYOUT, title=title, height=320)
    fig.update_xaxes(range=[0, 1], gridcolor="#21262d", zerolinecolor="#30363d")
    fig.update_yaxes(gridcolor="#21262d", zerolinecolor="#30363d")
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# Confidence CSS class
# ─────────────────────────────────────────────────────────────────────────────
def conf_cls(c): return "conf-high" if c>=.8 else ("conf-mid" if c>=.5 else "conf-low")

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('''<div class="sidebar-brand">
        <div class="sidebar-title">🧬 Bio-Seq LM</div>
        <div class="sidebar-sub">MS Applied Data Science Capstone<br>Deepika Sarala Pratapa · UF 2026</div>
    </div>''', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">Navigation</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="color:#475569;font-size:14px;line-height:2">
    🧬 DNA Classification<br>
    🔬 Protein Classification<br>
    🔍 Sequence Similarity<br>
    🌐 Embedding Explorer<br>
    🧪 Sequence Clustering<br>
    📊 Results Dashboard
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">Analysis Options</div>', unsafe_allow_html=True)
    run_mut = st.checkbox("In-silico mutagenesis", value=True,
                           help="Mutate each position, measure prediction change")
    run_nn  = st.checkbox("Nearest-neighbour retrieval", value=True,
                           help="Find 5 most similar training sequences")

    st.markdown('<div class="sidebar-section">Best Results</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:14px;line-height:1.8">
    <span class="stat-chip">DNA ROC-AUC 0.851</span><br>
    <span class="stat-chip" style="background:rgba(139,92,246,.12);border-color:rgba(139,92,246,.25);color:#c4b5fd">Protein Acc 99.1%</span>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">Models</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="color:#334155;font-size:14px;line-height:1.8">
    DNA: Baseline · CNN · Hybrid<br>DNABERT-2 · NT-500M<br><br>
    Protein: Baseline · CNN · Hybrid<br>ESM-2 · ProtBERT
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Main tabs
# ─────────────────────────────────────────────────────────────────────────────
(tab_dna, tab_prot, tab_sim, tab_explore,
 tab_cluster, tab_dash, tab_about) = st.tabs([
    "🧬 DNA Classification",
    "🔬 Protein Classification",
    "🔍 Sequence Similarity",
    "🌐 Embedding Explorer",
    "🧪 Sequence Clustering",
    "📊 Results Dashboard",
    "ℹ️ About",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DNA CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════
with tab_dna:
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-badge">🧬 DNA Analysis</div>
        <h2>Promoter vs Non-Promoter Classification</h2>
        <p>Classify any human DNA sequence across 5 modeling paradigms simultaneously.
        Baseline ML uses 95 engineered features (GC content, CpG density, k-mer frequencies, Shannon entropy).
        DNABERT-2 and NT-500M use pre-extracted transformer embeddings + XGBoost.
        <strong style="color:#60a5fa"> Works with any DNA sequence.</strong></p>
        <div class="info-pills" style="margin-top:12px">
            <span class="info-pill">📊 4,000 training sequences</span>
            <span class="info-pill">📏 200 bp window</span>
            <span class="info-pill">🧬 Human GRCh38</span>
            <span class="info-pill">🏆 Best ROC-AUC 0.851</span>
        </div>
    </div>""",unsafe_allow_html=True)

    # ── DNA example loader ───────────────────────────────────────────────────
    c1,c2=st.columns([3,1])
    if "dna_text" not in st.session_state:
        st.session_state["dna_text"] = ""
    
    with c2:
        ex_k=st.selectbox("Load example",["— custom —"]+list(DNA_EXAMPLES.keys()),key="dna_ex")
        if ex_k != "— custom —":
            if st.button("⬇ Load into box", key="dna_load_btn"):
                st.session_state["dna_text"] = DNA_EXAMPLES[ex_k]["seq"]
                st.rerun()
        # st.success(st.session_state)
    with c1:
        dna_val=DNA_EXAMPLES.get(ex_k,{})
        # Pre-fill only when session state has been set via the Load button
        dna_in=st.text_area("Paste DNA sequence (A/C/G/T/N)",
                             height=120,placeholder="ATCGATCG…",key="dna_text")

    if isinstance(dna_val,dict) and "info" in dna_val and ex_k!="— custom —":
        st.markdown(f'<div class="seq-info">📖 {dna_val["info"]}</div>',unsafe_allow_html=True)

    if st.button("🔍 Classify & Analyse",type="primary",key="dna_btn"):
        seq,err=val_dna(dna_in)
        if err: st.error(f"❌ {err}")
        else:
            orig_l=len(dna_in.strip().replace(" ","").replace("\n",""))
            st.success(f"✅ {orig_l} bp accepted (padded/trimmed to 200 bp for baseline)")

            results={}
            with st.spinner("Running all classifiers…"):
                feats=dna_features(seq)
                for n,clf in load_dna_baseline().items():
                    try:
                        p=float(clf.predict_proba(feats)[0,1])
                        results[f"Baseline / {n}"]={"prob":p,"pred":int(p>=.5),"clf":clf}
                    except Exception as e: results[f"Baseline / {n}"]={"error":str(e)}

            dna_embs=load_dna_embs()
            base_p=next((v["prob"] for v in results.values() if "prob" in v),.5)
            proxy_lbl=int(base_p>=.5)

            for mk,lbl,fn in [("dnabert2","DNABERT-2 / XGBoost",load_dnabert2_clf),
                               ("nt","NT-500M / XGBoost",load_nt_clf)]:
                clf=fn(); ed=dna_embs.get(mk)
                if clf is None or ed is None: continue
                proxy=ed["X"][ed["y"]==proxy_lbl].mean(0,keepdims=True)
                try:
                    p=float(clf.predict_proba(proxy)[0,1])
                    results[lbl]={"prob":p,"pred":int(p>=.5),"emb":proxy,"emb_data":ed,"approx":True}
                except Exception as e: results[lbl]={"error":str(e)}

            # ── Prediction cards ──────────────────────────────────────────
            st.markdown('<div class="section-header"><div class="section-header-line"></div><h3>🎯 Predictions</h3><div class="section-header-line" style="background:linear-gradient(90deg,transparent,rgba(59,130,246,.3))"></div></div>',unsafe_allow_html=True)
            valid_r={k:v for k,v in results.items() if "prob" in v}
            cols=st.columns(min(len(valid_r),3))
            for i,(name,res) in enumerate(valid_r.items()):
                p=res["prob"]; badge="badge-promoter" if res["pred"] else "badge-nonpromoter"
                lbl_txt="🟢 PROMOTER" if res["pred"] else "🔴 NON-PROMOTER"
                sn=name.replace("Baseline / ","").replace(" / XGBoost","")
                approx='<span style="color:#f59e0b;font-size:10px"> ⚠ approx</span>' if res.get("approx") else ""
                cc=conf_cls(p if res["pred"] else 1-p)
                with cols[i%3]:
                    st.markdown(f"""<div class="bio-card">
                    <h4>{sn}{approx}</h4>
                    <div class="{badge}">{lbl_txt}</div><br>
                    <span class="{cc}" style="font-size:22px">{p*100:.1f}%</span>
                    <span style="color:#475569;font-size:11px"> promoter prob.</span>
                    </div>""",unsafe_allow_html=True)

            # ── Confidence chart ──────────────────────────────────────────
            st.markdown('<div class="section-header"><div class="section-header-line"></div><h3>📊 Confidence</h3><div class="section-header-line" style="background:linear-gradient(90deg,transparent,rgba(59,130,246,.3))"></div></div>',unsafe_allow_html=True)
            names_c=list(valid_r.keys()); probs_c=[valid_r[k]["prob"] for k in names_c]
            if HAS_PLOTLY:
                st.plotly_chart(plotly_conf_bar(names_c,probs_c,.5,"Promoter probability by paradigm"),
                                use_container_width=True)
            else:
                fig_c,ax_c=plt.subplots(figsize=(9,3))
                colors_c=["#10b981" if p>=.5 else "#ef4444" for p in probs_c]
                ax_c.barh(names_c,probs_c,color=colors_c,alpha=.85,edgecolor="none")
                ax_c.axvline(.5,color="#f59e0b",lw=1.5,ls="--"); ax_c.set_xlim(0,1)
                plt.tight_layout(); st.pyplot(fig_c); plt.close(fig_c)

            # ── Mutagenesis ───────────────────────────────────────────────
            if run_mut:
                st.markdown('<div class="section-header"><div class="section-header-line"></div><h3>🔬 In-silico Mutagenesis</h3><div class="section-header-line" style="background:linear-gradient(90deg,transparent,rgba(59,130,246,.3))"></div></div>',unsafe_allow_html=True)
                st.caption("Each position mutated to every other nucleotide. Green = increases promoter probability. Red = decreases it.")
                best_clf=load_dna_baseline().get("XGBoost",load_dna_baseline().get("Random Forest",
                          next(iter(load_dna_baseline().values()),None)))
                if best_clf:
                    with st.spinner("Scanning positions (≈15 s for 80 bp)…"):
                        n_scan=min(len(dna_in.strip().replace(" ","").replace("\n","")),80)
                        deltas=dna_mutagenesis(seq,best_clf,n=n_scan)
                    if HAS_PLOTLY:
                        st.plotly_chart(plotly_mutagenesis(deltas,seq[:n_scan],
                            "Per-nucleotide sensitivity (XGBoost mutagenesis scan)"),
                            use_container_width=True)
                    else:
                        fig_m,ax_m=plt.subplots(figsize=(12,2.5))
                        vmax=max(abs(deltas).max(),.001)
                        cmap=mcolors.LinearSegmentedColormap.from_list("rg",["#ef4444","#1e293b","#10b981"])
                        ax_m.imshow(deltas.reshape(1,-1),aspect="auto",cmap=cmap,vmin=-vmax,vmax=vmax)
                        ax_m.set_xticks(range(len(deltas)))
                        ax_m.set_xticklabels(list(seq[:len(deltas)]),fontsize=6,fontfamily="monospace")
                        ax_m.set_yticks([]); plt.tight_layout(); st.pyplot(fig_m); plt.close(fig_m)
                    top3=np.argsort(np.abs(deltas))[-3:][::-1]
                    st.info("**Most sensitive positions:** " +
                            ", ".join([f"pos {p+1} ({seq[p]}) Δ={deltas[p]:+.3f}" for p in top3]))

            # ── Embedding visualisation ───────────────────────────────────
            st.markdown('<div class="section-header"><div class="section-header-line"></div><h3>🗺️ Embedding Space</h3><div class="section-header-line" style="background:linear-gradient(90deg,transparent,rgba(59,130,246,.3))"></div></div>',unsafe_allow_html=True)
            etabs=st.tabs(["DNABERT-2","NT-500M"])
            for etab,mk,lbl in zip(etabs,["dnabert2","nt"],
                                   ["DNABERT-2 / XGBoost","NT-500M / XGBoost"]):
                with etab:
                    coords,y_bg,_,_=precompute_pca(mk,"dna")
                    res_emb=results.get(lbl,{})
                    nn_idx=None; q_pt=None
                    if coords is not None and "emb" in res_emb:
                        ed=res_emb.get("emb_data") or load_dna_embs().get(mk)
                        if ed:
                            q_pt=project_into_pca(res_emb["emb"],ed["X"])
                            if run_nn:
                                idx_nn,_,_ = nn_search(res_emb["emb"],ed["X"],ed["y"],k=5)
                                nn_idx=idx_nn
                    dna_df=load_dna_seqs()
                    hover=None
                    if dna_df is not None and coords is not None:
                        hover=[f"ID: {dna_df.iloc[j]['region_id'] if 'region_id' in dna_df.columns else j}<br>"
                               f"Label: {'Promoter' if y_bg[j]==1 else 'Non-promoter'}"
                               for j in range(len(y_bg))]
                    if HAS_PLOTLY and coords is not None:
                        st.plotly_chart(
                            plotly_pca(coords,y_bg,q_pt,["Non-promoter","Promoter"],
                                       nn_idx,f"{mk.upper()} PCA — 4,000 training sequences",hover),
                            use_container_width=True)
                    else:
                        st.info("Plotly not installed — install with `pip install plotly`")

            # ── Feature importance ────────────────────────────────────────
            st.markdown('<div class="section-header"><div class="section-header-line"></div><h3>📈 Feature Importance</h3><div class="section-header-line" style="background:linear-gradient(90deg,transparent,rgba(59,130,246,.3))"></div></div>',unsafe_allow_html=True)
            fn_dna=(["frac_A","frac_C","frac_G","frac_T","frac_N","gc","at","gc_skew","at_skew",
                     "cpg_density","cpg_o_e","entropy","max_hp","periodicity","cpg_rich"]
                    +[f"di_{a}{b}" for a in "ACGT" for b in "ACGT"]
                    +[f"tri_{''.join(t)}" for t in itertools.product("ACGT",repeat=3)])
            xgb=load_dna_baseline().get("XGBoost")
            if xgb:
                est=xgb
                if hasattr(xgb,"named_steps"):
                    for s in xgb.named_steps.values():
                        if hasattr(s,"feature_importances_"): est=s; break
                if hasattr(est,"feature_importances_"):
                    imp=est.feature_importances_
                    top=np.argsort(imp)[-20:][::-1]
                    labels=[fn_dna[i] if i<len(fn_dna) else f"f{i}" for i in top]
                    vals=imp[top]
                    if HAS_PLOTLY:
                        fig_fi=go.Figure(go.Bar(x=vals[::-1],y=labels[::-1],orientation="h",
                                                marker_color=ACCENT[::-1]*4,marker_line_width=0))
                        fig_fi.update_layout(**PLOTLY_LAYOUT,title="Top 20 DNA features — XGBoost",
                                             height=480,xaxis_title="Importance")
                        st.plotly_chart(fig_fi,use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PROTEIN CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════
with tab_prot:
    st.markdown("""
    <div class="hero-banner" style="background:linear-gradient(135deg,rgba(139,92,246,.2) 0%,rgba(67,56,202,.12) 50%,rgba(5,11,24,.9) 100%);border-color:rgba(139,92,246,.22)">
        <div class="hero-badge" style="background:rgba(139,92,246,.15);border-color:rgba(139,92,246,.3);color:#c4b5fd">🔬 Protein Analysis</div>
        <h2>Pfam Family Classification</h2>
        <p>Classify any protein sequence into one of 10 Pfam functional families across 5 paradigms.
        Baseline ML uses 176 features (amino acid composition, CTD descriptors, PseAAC, reduced-alphabet dipeptides).
        ESM-2 and ProtBERT use pre-extracted transformer embeddings.
        <strong style="color:#a78bfa"> Works with any protein sequence.</strong></p>
        <div class="info-pills" style="margin-top:12px">
            <span class="info-pill">🔬 2,293 training sequences</span>
            <span class="info-pill">🗂️ 10 Pfam families</span>
            <span class="info-pill">🧬 UniProt Swiss-Prot</span>
            <span class="info-pill">🏆 Best Accuracy 99.1%</span>
        </div>
    </div>""",unsafe_allow_html=True)

    # ── Protein example loader ────────────────────────────────────────────────
    cp1,cp2=st.columns([3,1])
    with cp2:
        ex_pk=st.selectbox("Load example",["— custom —"]+list(PROTEIN_EXAMPLES.keys()),key="prot_ex")
        
        if ex_pk != "— custom —":
            if st.button("⬇ Load into box", key="prot_load_btn"):
                st.session_state["prot_text"] = PROTEIN_EXAMPLES[ex_pk]["seq"]
    with cp1:
        prot_val=PROTEIN_EXAMPLES.get(ex_pk,{})
        # prot_in=st.text_area("Paste protein sequence (single-letter AA codes)",
        #                       value=st.session_state.get("prot_text",""),
        #                       height=120,placeholder="MKTLL…",key="prot_seq")
        prot_in=st.text_area("Paste protein sequence (single-letter AA codes)",
                              height=120,placeholder="MKTLL…",key="prot_text")

    if isinstance(prot_val,dict) and "info" in prot_val and ex_pk!="— custom —":
        st.markdown(f'<div class="seq-info">📖 {prot_val["info"]}</div>',unsafe_allow_html=True)

    if st.button("🔍 Classify & Analyse",type="primary",key="prot_btn"):
        seq_p,err_p=val_prot(prot_in)
        if err_p: st.error(f"❌ {err_p}")
        else:
            st.success(f"✅ {len(seq_p)} aa accepted")
            res_p={}
            with st.spinner("Running all classifiers…"):
                fp=prot_features(seq_p)
                for n,clf in load_protein_baseline().items():
                    try:
                        pr=clf.predict_proba(fp)[0]
                        res_p[f"Baseline / {n}"]={"proba":pr,"pred":int(np.argmax(pr))}
                    except Exception as e: res_p[f"Baseline / {n}"]={"error":str(e)}

            pe=load_prot_embs()
            bp=next((v["pred"] for v in res_p.values() if "pred" in v),0)
            for mk,lbl,fn in [("esm2","ESM-2 / Linear SVM",load_esm2_clf),
                               ("protbert","ProtBERT / Linear SVM",load_protbert_clf)]:
                clf=fn(); ed=pe.get(mk)
                if clf is None or ed is None: continue
                proxy=ed["X"][ed["y"]==bp].mean(0,keepdims=True)
                try:
                    pr=clf.predict_proba(proxy)[0]
                    res_p[lbl]={"proba":pr,"pred":int(np.argmax(pr)),"emb":proxy,"emb_data":ed,"approx":True}
                except Exception as e: res_p[lbl]={"error":str(e)}

            # ── Prediction cards ──────────────────────────────────────────
            st.markdown('<div class="section-header"><div class="section-header-line"></div><h3>🎯 Predictions</h3><div class="section-header-line" style="background:linear-gradient(90deg,transparent,rgba(59,130,246,.3))"></div></div>',unsafe_allow_html=True)
            valid_rp={k:v for k,v in res_p.items() if "proba" in v}
            cols_p=st.columns(min(len(valid_rp),3))
            for i,(name,res) in enumerate(valid_rp.items()):
                pc=res["pred"]; pid=PFAM_CLASSES[pc] if pc<len(PFAM_CLASSES) else "?"
                pn=PFAM_NAMES.get(pid,pid); conf=float(res["proba"][pc])
                sn=name.replace("Baseline / ","").replace(" / Linear SVM","").replace(" / XGBoost","")
                approx='<span style="color:#f59e0b;font-size:10px"> ⚠ approx</span>' if res.get("approx") else ""
                with cols_p[i%3]:
                    st.markdown(f"""<div class="bio-card">
                    <h4>{sn}{approx}</h4>
                    <div class="badge-protein">{pid}</div><br>
                    <span style="color:#7c93b8;font-size:12px">{pn}</span><br>
                    <span class="{conf_cls(conf)}" style="font-size:20px">{conf*100:.1f}%</span>
                    <span style="color:#475569;font-size:11px"> confidence</span>
                    </div>""",unsafe_allow_html=True)

            # ── Pfam info card ────────────────────────────────────────────
            best_pc=next((v["pred"] for v in res_p.values() if "pred" in v),0)
            best_pid=PFAM_CLASSES[best_pc] if best_pc<len(PFAM_CLASSES) else "?"
            st.markdown(f"""<div class="bio-card" style="border-color:rgba(139,92,246,.4);margin-top:10px">
            <h4 style="color:#a78bfa">📖 {best_pid} — {PFAM_NAMES.get(best_pid,'')}</h4>
            <p>{PFAM_DESC.get(best_pid,'No description.')}</p>
            </div>""",unsafe_allow_html=True)

            # ── Top-3 chart ───────────────────────────────────────────────
            st.markdown('<div class="section-header"><div class="section-header-line"></div><h3>📊 Top-3 Families</h3><div class="section-header-line" style="background:linear-gradient(90deg,transparent,rgba(59,130,246,.3))"></div></div>',unsafe_allow_html=True)
            best_res_p=None
            for pref in ["ProtBERT / Linear SVM","ESM-2 / Linear SVM","Baseline / XGBoost"]:
                if pref in res_p and "proba" in res_p[pref]: best_res_p=res_p[pref]; break
            if best_res_p and HAS_PLOTLY:
                pa=best_res_p["proba"]; top3=np.argsort(pa)[-3:][::-1]
                labs=[f"{PFAM_CLASSES[i]} — {PFAM_NAMES.get(PFAM_CLASSES[i],'')[:28]}" for i in top3]
                fig_t3=go.Figure(go.Bar(x=pa[top3],y=labs,orientation="h",
                                        marker_color=[ACCENT[i%len(ACCENT)] for i in top3],
                                        text=[f"{pa[i]*100:.1f}%" for i in top3],textposition="auto",
                                        marker_line_width=0))
                fig_t3.update_layout(**PLOTLY_LAYOUT,title="Top-3 Pfam families",
                                     xaxis=dict(range=[0,1]),height=280)
                st.plotly_chart(fig_t3,use_container_width=True)

            # ── Alanine scan ──────────────────────────────────────────────
            if run_mut:
                st.markdown('<div class="section-header"><div class="section-header-line"></div><h3>🔬 Alanine Scan</h3><div class="section-header-line" style="background:linear-gradient(90deg,transparent,rgba(59,130,246,.3))"></div></div>',unsafe_allow_html=True)
                st.caption("Each residue mutated to Alanine (or Glycine if already Ala). Red = position critical for family identity.")
                bpclf=load_protein_baseline().get("XGBoost",next(iter(load_protein_baseline().values()),None))
                if bpclf:
                    n_sc=min(len(seq_p),60)
                    with st.spinner(f"Scanning {n_sc} residues…"):
                        del_p=ala_scan(seq_p,bpclf,n=n_sc)
                    if HAS_PLOTLY:
                        st.plotly_chart(plotly_mutagenesis(del_p,seq_p[:n_sc],
                            f"Alanine scan — first {n_sc} residues","Δ top-class probability"),
                            use_container_width=True)
                    top3p=np.argsort(np.abs(del_p))[-3:][::-1]
                    st.info("**Most critical residues:** "+
                            ", ".join([f"pos {p+1} ({seq_p[p]}) Δ={del_p[p]:+.3f}" for p in top3p]))

            # ── Embedding plots ───────────────────────────────────────────
            st.markdown('<div class="section-header"><div class="section-header-line"></div><h3>🗺️ Embedding Space</h3><div class="section-header-line" style="background:linear-gradient(90deg,transparent,rgba(59,130,246,.3))"></div></div>',unsafe_allow_html=True)
            ptabs=st.tabs(["ESM-2","ProtBERT"])
            for etab_p,mk,lbl in zip(ptabs,["esm2","protbert"],
                                     ["ESM-2 / Linear SVM","ProtBERT / Linear SVM"]):
                with etab_p:
                    coords_p,y_p,_,_=precompute_pca(mk,"prot")
                    re_p=res_p.get(lbl,{}); nn_p=None; qp=None
                    if coords_p is not None and "emb" in re_p:
                        edp=re_p.get("emb_data") or load_prot_embs().get(mk)
                        if edp:
                            qp=project_into_pca(re_p["emb"],edp["X"])
                            if run_nn:
                                idx_p,_,_=nn_search(re_p["emb"],edp["X"],edp["y"],k=5)
                                nn_p=idx_p
                    prot_df=load_prot_seqs()
                    hoverp=None
                    if prot_df is not None and coords_p is not None:
                        hoverp=[f"Accession: {prot_df.iloc[j]['accession'] if 'accession' in prot_df.columns else j}<br>"
                                f"Family: {PFAM_NAMES.get(prot_df.iloc[j]['family'],prot_df.iloc[j]['family']) if 'family' in prot_df.columns else y_p[j]}"
                                for j in range(len(y_p))]
                    if HAS_PLOTLY and coords_p is not None:
                        st.plotly_chart(
                            plotly_pca(coords_p,y_p,qp,PFAM_CLASSES,nn_p,
                                       f"{mk.upper()} PCA — 2,293 training sequences (10 Pfam families)",hoverp),
                            use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — SEQUENCE SIMILARITY SEARCH
# ══════════════════════════════════════════════════════════════════════════════
with tab_sim:
    st.markdown("""
    <div class="hero-banner" style="background:linear-gradient(135deg,rgba(16,185,129,.15) 0%,rgba(5,150,105,.08) 50%,rgba(5,11,24,.9) 100%);border-color:rgba(16,185,129,.2)">
        <div class="hero-badge" style="background:rgba(16,185,129,.12);border-color:rgba(16,185,129,.28);color:#6ee7b7">🔍 Similarity Search</div>
        <h2>Sequence Similarity Search</h2>
        <p>Find the most similar sequences to your query using cosine similarity in transformer embedding space.
        Works for <strong>any</strong> DNA or protein sequence — novel sequences, clinical variants, orthologues from other species.
        Similar sequences in embedding space share function, not just sequence identity.
        Think BLAST, but in learned representation space.</p>
    </div>""",unsafe_allow_html=True)

    task_sim=st.radio("Task",["🧬 DNA","🔬 Protein"],horizontal=True,key="sim_task")
    k_nn=st.slider("Number of neighbours to retrieve",3,20,10,key="sim_k")
    sim_in=st.text_area("Paste query sequence",height=100,
                          placeholder="Any DNA or protein sequence — novel sequences welcome",
                          key="sim_seq")

    if st.button("🔍 Find similar sequences",type="primary",key="sim_btn"):
        if "DNA" in task_sim:
            seq_s,err_s=val_dna(sim_in)
            if err_s: st.error(f"❌ {err_s}")
            else:
                dna_embs_s=load_dna_embs()
                dna_df_s=load_dna_seqs()
                for mk,mk_label in [("dnabert2","DNABERT-2"),("nt","NT-500M")]:
                    ed=dna_embs_s.get(mk)
                    if ed is None: continue
                    proxy_lbl=int(float(load_dna_baseline().get("XGBoost",
                        next(iter(load_dna_baseline().values()))).predict_proba(dna_features(seq_s))[0,1])>=.5)
                    proxy=ed["X"][ed["y"]==proxy_lbl].mean(0,keepdims=True)
                    idx_s,sims_s,lbls_s=nn_search(proxy,ed["X"],ed["y"],k=k_nn)

                    st.markdown(f'<div class="section-header"><div class="section-header-line"></div><h3>🔍 {mk_label} nearest neighbours</h3><div class="section-header-line" style="background:linear-gradient(90deg,transparent,rgba(59,130,246,.3))"></div></div>',
                                unsafe_allow_html=True)
                    rows=[]
                    for rank,(i,sim,lbl) in enumerate(zip(idx_s,sims_s,lbls_s),1):
                        row={"Rank":rank,"Similarity":f"{sim:.4f}",
                             "Label":"Promoter" if lbl==1 else "Non-promoter"}
                        if dna_df_s is not None and i<len(dna_df_s):
                            r=dna_df_s.iloc[i]
                            if "region_id" in r: row["Region ID"]=r["region_id"]
                            if "sequence" in r: row["Sequence (first 40bp)"]=r["sequence"][:40]+"…"
                        rows.append(row)
                    df_nn=pd.DataFrame(rows)
                    st.dataframe(df_nn,use_container_width=True)

                    coords_s,y_s,_,_=precompute_pca(mk,"dna")
                    if HAS_PLOTLY and coords_s is not None:
                        st.plotly_chart(
                            plotly_pca(coords_s,y_s,None,["Non-promoter","Promoter"],
                                       idx_s,f"{mk_label} — top {k_nn} neighbours highlighted"),
                            use_container_width=True)
        else:
            seq_s,err_s=val_prot(sim_in)
            if err_s: st.error(f"❌ {err_s}")
            else:
                pe_s=load_prot_embs(); prot_df_s=load_prot_seqs()
                for mk,mk_label in [("esm2","ESM-2"),("protbert","ProtBERT")]:
                    ed=pe_s.get(mk)
                    if ed is None: continue
                    bp=int(np.argmax(load_protein_baseline().get("XGBoost",
                        next(iter(load_protein_baseline().values()))).predict_proba(prot_features(seq_s))[0]))
                    proxy=ed["X"][ed["y"]==bp].mean(0,keepdims=True)
                    idx_s,sims_s,lbls_s=nn_search(proxy,ed["X"],ed["y"],k=k_nn)

                    st.markdown(f'<div class="section-header"><div class="section-header-line"></div><h3>🔍 {mk_label} nearest neighbours</h3><div class="section-header-line" style="background:linear-gradient(90deg,transparent,rgba(59,130,246,.3))"></div></div>',
                                unsafe_allow_html=True)
                    rows=[]
                    for rank,(i,sim,lbl) in enumerate(zip(idx_s,sims_s,lbls_s),1):
                        pfid=PFAM_CLASSES[lbl] if lbl<len(PFAM_CLASSES) else str(lbl)
                        row={"Rank":rank,"Similarity":f"{sim:.4f}",
                             "Pfam":pfid,"Family":PFAM_NAMES.get(pfid,pfid)}
                        if prot_df_s is not None and i<len(prot_df_s):
                            r=prot_df_s.iloc[i]
                            if "accession" in r: row["UniProt"]=r["accession"]
                            if "sequence" in r: row["Sequence (first 30aa)"]=r["sequence"][:30]+"…"
                        rows.append(row)
                    df_nn=pd.DataFrame(rows)
                    st.dataframe(df_nn,use_container_width=True)

                    coords_s,y_s,_,_=precompute_pca(mk,"prot")
                    if HAS_PLOTLY and coords_s is not None:
                        st.plotly_chart(
                            plotly_pca(coords_s,y_s,None,PFAM_CLASSES,
                                       idx_s,f"{mk_label} — top {k_nn} neighbours highlighted"),
                            use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — EMBEDDING EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with tab_explore:
    st.markdown("""
    <div class="hero-banner" style="background:linear-gradient(135deg,rgba(245,158,11,.12) 0%,rgba(180,83,9,.08) 50%,rgba(5,11,24,.9) 100%);border-color:rgba(245,158,11,.18)">
        <div class="hero-badge" style="background:rgba(245,158,11,.12);border-color:rgba(245,158,11,.25);color:#fcd34d">🌐 Embedding Explorer</div>
        <h2>Interactive Embedding Space</h2>
        <p>Hover over any of the 4,000 DNA or 2,293 protein training sequences to see its identity, label, and accession.
        The 2D PCA projection reveals the geometric structure of the embedding space.
        Tight, well-separated clusters explain why linear classifiers achieve 99.1% accuracy on ProtBERT embeddings.</p>
    </div>""",unsafe_allow_html=True)

    task_exp=st.radio("Task",["🧬 DNA","🔬 Protein"],horizontal=True,key="exp_task")
    if "DNA" in task_exp:
        mk_exp=st.radio("Embedding model",["DNABERT-2","NT-500M"],horizontal=True,key="exp_mk")
        mk_key="dnabert2" if mk_exp=="DNABERT-2" else "nt"
        coords_e,y_e,_,_=precompute_pca(mk_key,"dna")
        dna_df_e=load_dna_seqs()
        hover_e=None
        if dna_df_e is not None and coords_e is not None:
            hover_e=[f"{'Promoter' if y_e[j]==1 else 'Non-promoter'}<br>"
                     f"ID: {dna_df_e.iloc[j]['region_id'] if 'region_id' in dna_df_e.columns else j}<br>"
                     f"Seq: {dna_df_e.iloc[j]['sequence'][:30] if 'sequence' in dna_df_e.columns else ''}…"
                     for j in range(len(y_e))]
        if HAS_PLOTLY and coords_e is not None:
            st.plotly_chart(
                plotly_pca(coords_e,y_e,None,["Non-promoter","Promoter"],
                           title=f"{mk_exp} — 4,000 DNA sequences (hover to inspect)",
                           hover_texts=hover_e),
                use_container_width=True)
            st.caption(f"**{(y_e==1).sum()}** promoter sequences (blue) · **{(y_e==0).sum()}** non-promoter (orange). "
                       "Hover any point to see its sequence. Zoom and pan freely.")
        else:
            st.info("Install plotly for interactive embedding explorer: `pip install plotly`")
    else:
        mk_exp=st.radio("Embedding model",["ESM-2","ProtBERT"],horizontal=True,key="exp_mk_p")
        mk_key="esm2" if mk_exp=="ESM-2" else "protbert"
        coords_e,y_e,_,_=precompute_pca(mk_key,"prot")
        prot_df_e=load_prot_seqs()
        hover_e=None
        if prot_df_e is not None and coords_e is not None:
            hover_e=[f"{PFAM_NAMES.get(prot_df_e.iloc[j]['family'],prot_df_e.iloc[j]['family']) if 'family' in prot_df_e.columns else PFAM_CLASSES[y_e[j]]}<br>"
                     f"Accession: {prot_df_e.iloc[j]['accession'] if 'accession' in prot_df_e.columns else j}<br>"
                     f"Pfam: {prot_df_e.iloc[j]['family'] if 'family' in prot_df_e.columns else ''}"
                     for j in range(len(y_e))]
        if HAS_PLOTLY and coords_e is not None:
            st.plotly_chart(
                plotly_pca(coords_e,y_e,None,PFAM_CLASSES,
                           title=f"{mk_exp} — 2,293 protein sequences (hover to inspect)",
                           hover_texts=hover_e),
                use_container_width=True)
            st.caption(f"10 Pfam families · {len(y_e)} sequences. "
                       "Tight clusters indicate families that are linearly separable in this embedding space — "
                       "explaining why linear classifiers achieve 99.1% accuracy on ProtBERT embeddings.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — SEQUENCE CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════
with tab_cluster:
    st.markdown("""
    <div class="hero-banner" style="background:linear-gradient(135deg,rgba(236,72,153,.12) 0%,rgba(157,23,77,.08) 50%,rgba(5,11,24,.9) 100%);border-color:rgba(236,72,153,.18)">
        <div class="hero-badge" style="background:rgba(236,72,153,.12);border-color:rgba(236,72,153,.25);color:#f9a8d4">🧪 Clustering</div>
        <h2>Sequence Clustering</h2>
        <p>Paste 2–10 sequences and cluster them in engineered feature space using KMeans.
        Compare wild-type vs mutant, sequences from different species, or sequences from a FASTA file.
        Fully unsupervised — no labels needed. Works for any DNA or protein sequence.</p>
    </div>""",unsafe_allow_html=True)

    task_cl=st.radio("Task",["🧬 DNA","🔬 Protein"],horizontal=True,key="cl_task")
    n_cl=st.slider("Number of sequences",2,10,3,key="cl_n")
    k_cl=st.slider("Number of clusters",2,min(n_cl,5),2,key="cl_k")

    cl_seqs=[]; cl_labels=[]
    cols_cl=st.columns(min(n_cl,3))
    for i in range(n_cl):
        with cols_cl[i%3]:
            lbl=st.text_input(f"Label {i+1}",f"Seq {i+1}",key=f"cl_lbl_{i}")
            seq=st.text_area(f"",height=80,key=f"cl_seq_{i}",
                              placeholder="ATCG…" if "DNA" in task_cl else "MKTLL…",
                              label_visibility="collapsed")
            cl_seqs.append(seq); cl_labels.append(lbl)

    if st.button("🧪 Cluster sequences",type="primary",key="cl_btn"):
        valid_cl=[]
        for lbl,seq in zip(cl_labels,cl_seqs):
            if not seq.strip(): continue
            if "DNA" in task_cl: s,e=val_dna(seq)
            else: s,e=val_prot(seq)
            if e: st.error(f"❌ {lbl}: {e}"); continue
            valid_cl.append((lbl,s))

        if len(valid_cl)<2:
            st.warning("Please enter at least 2 valid sequences.")
        else:
            with st.spinner("Computing features and clustering…"):
                if "DNA" in task_cl:
                    X_cl=np.vstack([dna_features(s) for _,s in valid_cl])
                else:
                    X_cl=np.vstack([prot_features(s) for _,s in valid_cl])
                sc_cl=StandardScaler().fit(X_cl)
                X_sc_cl=sc_cl.transform(X_cl)
                k_act=min(k_cl,len(valid_cl))
                km=KMeans(n_clusters=k_act,random_state=42,n_init=10).fit(X_sc_cl)
                cluster_labels=km.labels_
                pca_cl=PCA(n_components=min(2,len(valid_cl)-1),random_state=42)
                coords_cl=pca_cl.fit_transform(X_sc_cl)

            st.markdown('<div class="section-header"><div class="section-header-line"></div><h3>📊 Clustering result</h3><div class="section-header-line" style="background:linear-gradient(90deg,transparent,rgba(59,130,246,.3))"></div></div>',unsafe_allow_html=True)
            df_cl=pd.DataFrame({
                "Sequence":[l for l,_ in valid_cl],
                "Cluster":[f"Cluster {c+1}" for c in cluster_labels],
                "Length":[len(s) for _,s in valid_cl],
            })
            st.dataframe(df_cl,use_container_width=True)

            if HAS_PLOTLY and coords_cl.shape[1]>=2:
                fig_cl=go.Figure()
                for ci in range(k_act):
                    mask=cluster_labels==ci
                    idxs=np.where(mask)[0]
                    fig_cl.add_trace(go.Scatter(
                        x=coords_cl[mask,0],y=coords_cl[mask,1],
                        mode="markers+text",name=f"Cluster {ci+1}",
                        marker=dict(color=ACCENT[ci],size=16,
                                    line=dict(color="white",width=1.5)),
                        text=[valid_cl[j][0] for j in idxs],
                        textposition="top center",textfont=dict(size=11,color="#e2e8f0"),
                    ))
                fig_cl.update_layout(**PLOTLY_LAYOUT,
                                     title="Sequence clustering (KMeans in feature space, PCA projection)",
                                     height=420)
                st.plotly_chart(fig_cl,use_container_width=True)
            elif coords_cl.shape[1]==1:
                st.info("Only 1 PCA dimension available with 2 sequences — try adding more sequences for a 2D plot.")

            # Pairwise similarity matrix
            if HAS_PLOTLY:
                st.markdown('<div class="section-header"><div class="section-header-line"></div><h3>🔢 Pairwise cosine similarity</h3><div class="section-header-line" style="background:linear-gradient(90deg,transparent,rgba(59,130,246,.3))"></div></div>',unsafe_allow_html=True)
                Xn=X_sc_cl/(np.linalg.norm(X_sc_cl,axis=1,keepdims=True)+1e-9)
                sim_mat=Xn@Xn.T
                seq_names=[l for l,_ in valid_cl]
                fig_heat=go.Figure(go.Heatmap(
                    z=sim_mat,x=seq_names,y=seq_names,
                    colorscale=[[0,"#0d1117"],[0.5,"#1d4ed8"],[1,"#10b981"]],
                    text=[[f"{sim_mat[i,j]:.3f}" for j in range(len(seq_names))]
                          for i in range(len(seq_names))],
                    texttemplate="%{text}",
                    hovertemplate="%{y} vs %{x}: %{z:.4f}<extra></extra>",
                ))
                fig_heat.update_layout(**PLOTLY_LAYOUT,
                                       title="Pairwise cosine similarity (engineered feature space)",
                                       height=380)
                st.plotly_chart(fig_heat,use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — RESULTS DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab_dash:
    st.markdown("""
    <div class="hero-banner" style="background:linear-gradient(135deg,rgba(245,158,11,.12) 0%,rgba(5,11,24,.9) 100%);border-color:rgba(245,158,11,.18)">
        <div class="hero-badge" style="background:rgba(245,158,11,.1);border-color:rgba(245,158,11,.22);color:#fcd34d">📊 Results</div>
        <h2>Full Paradigm Comparison Dashboard</h2>
        <p>Complete evaluation of all 5 paradigms × 2 biological tasks. Best model per paradigm selected
        by primary metric (ROC-AUC for DNA, Accuracy for protein). All results from held-out test sets.</p>
    </div>""",unsafe_allow_html=True)

    dna_c,prot_c=load_comparison()
    summary_path=REPORTS/"final_summary.json"
    if summary_path.exists():
        with open(summary_path) as f: fs=json.load(f)
        m1,m2,m3,m4=st.columns(4)
        bd=fs.get("dna",{}).get("best_model",{})
        bp2=fs.get("protein",{}).get("best_model",{})
        with m1: st.metric("🏆 Best DNA","NT-500M / XGBoost",f"ROC-AUC {bd.get('roc_auc',0.851):.4f}")
        with m2: st.metric("🏆 Best Protein","ProtBERT / LinSVM",f"Acc {bp2.get('accuracy',0.991):.4f}")
        with m3: st.metric("🧬 DNA sequences","4,000","200 bp · GRCh38")
        with m4: st.metric("🔬 Protein sequences","2,293","10 Pfam families")

    st.markdown("---")
    cd,cp2=st.columns(2)

    with cd:
        st.markdown("### 🧬 DNA Results")
        if dna_c is not None:
            hl=[c for c in ["accuracy","f1","roc_auc"] if c in dna_c.columns]
            st.dataframe(dna_c.style.highlight_max(subset=hl,color="#0a2e1a")
                                    .format({c:"{:.4f}" for c in hl}),
                         use_container_width=True)
            if HAS_PLOTLY and "roc_auc" in dna_c.columns:
                fig_dd=go.Figure()
                fig_dd.add_trace(go.Bar(name="Accuracy",x=dna_c["paradigm"],y=dna_c["accuracy"],
                                        marker_color="#3b82f6",marker_line_width=0))
                fig_dd.add_trace(go.Bar(name="ROC-AUC",x=dna_c["paradigm"],y=dna_c["roc_auc"],
                                        marker_color="#f59e0b",marker_line_width=0))
                fig_dd.update_layout(**PLOTLY_LAYOUT,barmode="group",title="DNA paradigm comparison",height=340,)
                fig_dd.update_xaxes(gridcolor="#21262d", zerolinecolor="#30363d")
                fig_dd.update_yaxes(range=[0.6, 0.92], gridcolor="#21262d", zerolinecolor="#30363d")
                st.plotly_chart(fig_dd,use_container_width=True)
        else:
            st.info("Run notebook 21 to generate comparison CSVs.")

    with cp2:
        st.markdown("### 🔬 Protein Results")
        if prot_c is not None:
            hlp=[c for c in ["accuracy","f1_macro"] if c in prot_c.columns]
            st.dataframe(prot_c.style.highlight_max(subset=hlp,color="#0a2e1a")
                                     .format({c:"{:.4f}" for c in hlp}),
                         use_container_width=True)
            if HAS_PLOTLY and "accuracy" in prot_c.columns:
                fig_pp=go.Figure()
                fig_pp.add_trace(go.Bar(x=prot_c["paradigm"],y=prot_c["accuracy"],
                                        name="Accuracy",
                                        marker_color=[ACCENT[i] for i in range(len(prot_c))],
                                        marker_line_width=0))
                if "f1_macro" in prot_c.columns:
                    fig_pp.add_trace(go.Scatter(x=prot_c["paradigm"],y=prot_c["f1_macro"],
                                                name="F1-macro",mode="lines+markers",
                                                line=dict(color="#10b981",width=2),
                                                marker=dict(size=8,color="#10b981")))
                fig_pp.update_layout(**PLOTLY_LAYOUT,title="Protein paradigm comparison",height=340,)
                fig_pp.update_xaxes(gridcolor="#21262d", zerolinecolor="#30363d")
                fig_pp.update_yaxes(range=[0.5, 1.05], gridcolor="#21262d", zerolinecolor="#30363d")
                st.plotly_chart(fig_pp,use_container_width=True)
        else:
            st.info("Run notebook 21 to generate comparison CSVs.")

    st.markdown("---")
    st.markdown("### 🔑 Key Findings")
    kf1,kf2=st.columns(2)
    with kf1:
        st.markdown("""<div class="bio-card">
        <h4>🧬 DNA — Marginal transformer gains</h4>
        <p>NT-500M XGBoost achieves ROC-AUC 0.851, only 1.5 points above the baseline Random Forest (0.837).
        Linear classifiers on transformer embeddings <em>underperform</em> baselines — high-dimensional
        spaces (768–1280 dim) are less separable for this task. CpG density and GC content remain
        the most discriminative features, consistent with the biology of promoter regions.
        Promoter-specific regulatory signals are not fully captured by sequence-only masked LM objectives.</p>
        </div>""",unsafe_allow_html=True)
    with kf2:
        st.markdown("""<div class="bio-card">
        <h4>🔬 Protein — Dramatic transformer gains (+7%)</h4>
        <p>ProtBERT Linear SVM achieves 99.1% accuracy — a 7-point jump over Hybrid (95.6%).
        Linear classifiers are <em>strongest</em> on protein embeddings, opposite of the DNA task.
        Protein family structure is already linearly separable in pretrained LM embedding space.
        PCA confirms tight, distinct clusters for all 10 Pfam families in both ESM-2 and ProtBERT spaces.
        Evolutionary constraints encoded by UniRef100 pretraining align directly with Pfam classification.</p>
        </div>""",unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.markdown("""
    <div class="bio-card" style="margin-bottom:20px;border-color:rgba(59,130,246,.4)">
    <h4 style="font-size:22px">🧬 Bio-Seq LM Explorer</h4>
    <p style="font-size:15px"><strong style="color:#e2e8f0">Design and Evaluation of DNA &amp; Protein
    Language Model Pipelines for Biological Sequence Analysis</strong><br>
    MS Applied Data Science Capstone · University of Florida · Spring 2026</p>
    </div>""",unsafe_allow_html=True)

    a1,a2=st.columns(2)
    with a1:
        st.markdown("""<div class="bio-card">
        <h4>👤 Author & Advisor</h4>
        <p><strong style="color:#e2e8f0">Deepika Sarala Pratapa</strong><br>
        MS Applied Data Science, University of Florida<br>
        dpratapa@ufl.edu<br><br>
        <strong style="color:#e2e8f0">Faculty Advisor</strong><br>
        Dr. Matthew Gitzendanner<br>
        Research Computing, UF Information Technology<br>
        magitz@ufl.edu</p>
        </div>""",unsafe_allow_html=True)

        st.markdown("""<div class="bio-card" style="margin-top:10px">
        <h4>🗄️ Data sources</h4>
        <p><strong style="color:#60a5fa">DNA:</strong> 4,000 sequences (200 bp, GRCh38).
        Promoters from Ensembl regulatory annotations; non-promoters sampled from intergenic regions.
        Balanced (2,000 each class).<br><br>
        <strong style="color:#a78bfa">Protein:</strong> 2,293 sequences from UniProt/Swiss-Prot
        reviewed entries across 10 Pfam families, ≤400 sequences per family.
        Families selected to span diverse functional categories (GPCR, kinase, TF, immune, etc.).</p>
        </div>""",unsafe_allow_html=True)

    with a2:
        st.markdown("""<div class="bio-card">
        <h4>🤖 Models & paradigms</h4>
        <p><strong style="color:#60a5fa">DNA (5 paradigms):</strong><br>
        • Baseline ML — 95 features: GC, CpG density, CpG O/E ratio, Shannon entropy,
          dinucleotide + trinucleotide frequencies, homopolymer runs<br>
        • Sequence CNN — 1D convolutions on one-hot encoded DNA<br>
        • Hybrid — CNN features + engineered features concatenated<br>
        • DNABERT-2 — 117M params, BPE tokenization, 135-species pretraining, ALiBi positions<br>
        • NT-500M — 500M params, single-nucleotide, human GRCh38 pretraining<br><br>
        <strong style="color:#a78bfa">Protein (5 paradigms):</strong><br>
        • Baseline ML — 176 features: AA composition, grouped physicochemical,
          ProtParam biophysical, CTD descriptors, PseAAC, reduced-alphabet dipeptides<br>
        • Sequence CNN — 1D convolutions on one-hot amino acids<br>
        • Hybrid — CNN + engineered features<br>
        • ESM-2 — 35M params, Meta AI, UR50D (~65M sequences) pretraining<br>
        • ProtBERT — 420M params, Rostlab, UniRef100 (~217M sequences) pretraining</p>
        </div>""",unsafe_allow_html=True)

    st.markdown("""<div class="bio-card" style="margin-top:10px">
    <h4>📚 Key references</h4>
    <p>
    [1] Ji et al. (2023). DNABERT-2: Efficient Foundation Model and Benchmark for Multi-Species Genome. <em>arXiv:2306.15006</em><br>
    [2] Dalla-Torre et al. (2023). The Nucleotide Transformer: Building and Evaluating Robust Foundation Models for Human Genomics. <em>bioRxiv</em><br>
    [3] Elnaggar et al. (2021). ProtTrans: Towards Cracking the Language of Lifes Code Through Self-Supervised Learning. <em>IEEE TPAMI 44(10)</em><br>
    [4] Lin et al. (2023). Evolutionary-scale prediction of atomic-level protein structure with a language model. <em>Science 379(6637)</em><br>
    [5] Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python. <em>JMLR 12:2825-2830</em><br>
    [6] Wolf et al. (2020). HuggingFace's Transformers: State-of-the-art NLP. <em>arXiv:1910.03771</em>
    </p>
    </div>""",unsafe_allow_html=True)

    st.markdown("""<div class="bio-card" style="margin-top:10px">
    <h4>🖥️ Infrastructure & reproducibility</h4>
    <p>
    Trained and deployed on <strong style="color:#e2e8f0">UF HiPerGator</strong> HPC cluster
    (NVIDIA A100 GPUs, SLURM scheduler, 40 GB VRAM).<br>
    Environment: Python 3.11 · PyTorch 2.10 · HuggingFace Transformers 5.1 · scikit-learn 1.7
    · Streamlit 1.32+ · Plotly 5.x · conda env <code>biotm3</code>.<br>
    All transformer embeddings pre-extracted once and cached as NumPy arrays for fast demo inference.
    All random seeds fixed (seed=42) for reproducibility.
    Full code available at: <strong style="color:#60a5fa">github.com/deepikapratapa/bioseq-lm-capstone</strong>
    </p>
    </div>""",unsafe_allow_html=True)
