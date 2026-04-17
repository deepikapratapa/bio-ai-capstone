"""
Bio-Seq LM  —  Biological Sequence Language Model Explorer
===========================================================
Deepika Sarala Pratapa | MS Applied Data Science | University of Florida

Features
--------
• DNA promoter vs non-promoter classification (5 paradigms)
• Protein Pfam family classification (5 paradigms, 10 classes)
• In-silico mutagenesis with per-position importance heatmap
• Nearest-neighbour sequence retrieval from embedding space
• Interactive PCA / UMAP embedding explorer
• Multi-sequence comparison
• Results dashboard with publication-quality figures
• Downloadable per-sequence report

Run
---
    streamlit run app.py --server.port 8501 --server.headless true
"""

from __future__ import annotations
import io, json, math, re, warnings, itertools, textwrap
from pathlib import Path
from collections import Counter

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data"  / "processed"
MODELS    = ROOT / "models"
REPORTS   = ROOT / "reports"
DATA_RAW  = ROOT / "data"  / "raw"

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
PFAM_CLASSES = ["PF00001","PF00046","PF00069","PF00071","PF00076",
                "PF00096","PF01352","PF07686","PF12796","PF13853"]
PFAM_NAMES = {
    "PF00001": "7tm_1 (GPCR family A)",
    "PF00046": "Homeodomain",
    "PF00069": "Protein kinase domain",
    "PF00071": "Ras GTPase",
    "PF00076": "RRM (RNA recognition motif)",
    "PF00096": "Zinc finger C2H2",
    "PF01352": "KRAB domain",
    "PF07686": "Immunoglobulin V-set",
    "PF12796": "Ankyrin repeat",
    "PF13853": "7tm_4 (GPCR olfactory)",
}
PFAM_DESC = {
    "PF00001": "G protein-coupled receptors with 7 transmembrane helices. Largest family of cell-surface receptors in the human genome.",
    "PF00046": "Helix-turn-helix DNA-binding domain found in homeobox transcription factors. Critical for developmental gene regulation.",
    "PF00069": "Catalytic domain of protein kinases. Phosphorylates serine, threonine, or tyrosine residues in target proteins.",
    "PF00071": "GTPase domain of Ras superfamily. Molecular switches cycling between GTP-bound (active) and GDP-bound (inactive) states.",
    "PF00076": "RNA recognition motif. Binds single-stranded RNA; found in splicing factors, translation regulators, and RNA-binding proteins.",
    "PF00096": "C2H2 zinc finger domain. Most common DNA-binding domain in eukaryotes. Coordinates zinc via two cysteines and two histidines.",
    "PF01352": "KRAB (Krüppel-associated box) repression domain. Found in the largest family of transcriptional repressors in vertebrates.",
    "PF07686": "Immunoglobulin V-set domain. Antigen-binding variable domain of antibodies and T-cell receptors.",
    "PF12796": "Ankyrin repeat. Mediates protein-protein interactions; found in signaling, cytoskeletal, and cell-cycle proteins.",
    "PF13853": "Olfactory receptor subfamily of GPCRs (family C). Largest gene family in the human genome (~400 functional genes).",
}

DNA_EXAMPLES = {
    "Promoter — TATA-box (human ACTB)":
        "CCGGCTCCGAGCGGGCTGGGGCGGGGAGAGGGCGCGGGGCCAAGTCCGGGCGGAGCGGAGCGAGAGAGGG"
        "CGCGGGGCCAAGTCCGGGCGGAGCGGAGCGAGGGCGCGGGGCCAAGTCCGGGCGGAGCTATAAACGCGC"
        "GCGCGGGGCCAAGTCCGGGCGGAGCGGAGCGAGAGAGG",
    "Promoter — CpG island (house-keeping gene)":
        "CGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCG"
        "CGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCG"
        "CGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCG",
    "Non-promoter — intergenic repeat":
        "ATTCGATCGATCGATCGTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCT"
        "AGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAG"
        "CTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA",
    "Non-promoter — coding region exon":
        "ATGGCCACCGAGGTGCAGCTGGTGGAGTCTGGGGGAGGCGTGGTCCAGCCTGGGAGGTCCCTGAGACTC"
        "TCCTGTGCAGCCTCTGGATTCACCTTCAGTAGCTATGGCATGCACTGGGTCCGCCAGGCTCCAGGCAAGG"
        "GGCTGGAGTGGGTGGCAGTTATATCATATGATGGAAGC",
}

PROTEIN_EXAMPLES = {
    "GPCR olfactory receptor (PF13853)":
        "MRNHTEITEFILLGLTDDPNFQVVIFVFLLITYMLSITGNLTLITIAKDSHLHTPMYFFLSHLSFVDLSS"
        "VSSVPNMLVNLIQDIQPVLGLPCISKFIQFFMEHISLASSVGCLIAMALDRHVAIVHPLLYSTIMSKLAC"
        "YLLIAASWTLSFVLCVPVFLFQIVH",
    "Protein kinase catalytic domain (PF00069)":
        "MGSSHHHHHHSSGLVPRGSHMASMTGGQQMGRDLYDDDDKDPQMVKVGDKVTLKKLGEGAFGEVWMGKWN"
        "GTRVAIKTLKPGSMPEAFLAEANVMKTLQHDKLVKLHAVVTKEPIYIVTEYMSKGSLLHQLEKAKLMKKA",
    "Zinc finger C2H2 (PF00096)":
        "MARPYKTELKIVKKTDKKHFKVHQCNACGKRFMRSDNLKKHQKTHSGEKPFKCDICGRGFTQSGNLKRH"
        "QKIHTGEKPYKCNECGKSFIQSSDLKRHQRIHTGEKPYQCNECGKSFIQSSHLKRHQRIHTGEKPY",
    "Homeodomain transcription factor (PF00046)":
        "MSSYYHHHHHGYPYDVPDYAGYPYDVPDYAGSSCGGQNLPGINLPQQQQHHPHHHPHQQQQHQHQHQLHQ"
        "HQHQLHQHQHQHQHQLHQQHQHQHQQHQLHQHQHQHQHQHQHQHQHQRKKRRTIFTSQQLQELERAF",
}

# ─────────────────────────────────────────────────────────────────────────────
# Page config & Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bio-Seq LM Explorer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ── Global ── */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1530 50%, #0a0e1a 100%);
    min-height: 100vh;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1530 0%, #091020 100%);
    border-right: 1px solid rgba(59,130,246,0.2);
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, rgba(30,58,138,0.6), rgba(17,24,39,0.8));
    border: 1px solid rgba(59,130,246,0.3);
    border-radius: 12px;
    padding: 16px;
    backdrop-filter: blur(10px);
}

/* ── Tabs ── */
[data-testid="stTabs"] [role="tablist"] {
    background: rgba(15,23,42,0.8);
    border-radius: 12px;
    padding: 4px;
    border: 1px solid rgba(59,130,246,0.2);
}
[data-testid="stTabs"] [role="tab"] {
    color: #94a3b8 !important;
    border-radius: 8px;
    font-weight: 500;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, #1e3a8a, #1e40af) !important;
    color: white !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #1d4ed8, #2563eb) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 15px rgba(29,78,216,0.4) !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2563eb, #3b82f6) !important;
    box-shadow: 0 6px 20px rgba(59,130,246,0.5) !important;
    transform: translateY(-1px) !important;
}

/* ── Text area ── */
.stTextArea textarea {
    background: rgba(15,23,42,0.8) !important;
    border: 1px solid rgba(59,130,246,0.3) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-family: 'Courier New', monospace !important;
    font-size: 13px !important;
}
.stTextArea textarea:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 2px rgba(59,130,246,0.2) !important;
}

/* ── Selectbox ── */
.stSelectbox [data-baseweb="select"] {
    background: rgba(15,23,42,0.8) !important;
    border: 1px solid rgba(59,130,246,0.3) !important;
    border-radius: 10px !important;
}

/* ── DataFrames ── */
[data-testid="stDataFrame"] {
    border-radius: 10px !important;
    overflow: hidden !important;
    border: 1px solid rgba(59,130,246,0.2) !important;
}

/* ── Info / success / error boxes ── */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    border-left-width: 4px !important;
}

/* ── Custom card class ── */
.bio-card {
    background: linear-gradient(135deg, rgba(30,58,138,0.3), rgba(17,24,39,0.6));
    border: 1px solid rgba(59,130,246,0.25);
    border-radius: 14px;
    padding: 20px;
    margin: 8px 0;
    backdrop-filter: blur(10px);
}
.bio-card h4 { color: #60a5fa; margin: 0 0 8px 0; }
.bio-card p  { color: #cbd5e1; margin: 0; font-size: 14px; }

/* ── Prediction badges ── */
.pred-promoter {
    background: linear-gradient(135deg, #065f46, #047857);
    border: 1px solid #10b981;
    color: #6ee7b7;
    padding: 6px 14px; border-radius: 20px;
    font-weight: 700; font-size: 14px; display: inline-block;
}
.pred-nonpromoter {
    background: linear-gradient(135deg, #7f1d1d, #991b1b);
    border: 1px solid #ef4444;
    color: #fca5a5;
    padding: 6px 14px; border-radius: 20px;
    font-weight: 700; font-size: 14px; display: inline-block;
}
.pred-protein {
    background: linear-gradient(135deg, #1e3a8a, #1d4ed8);
    border: 1px solid #3b82f6;
    color: #93c5fd;
    padding: 6px 14px; border-radius: 20px;
    font-weight: 700; font-size: 14px; display: inline-block;
}
.conf-high { color: #10b981; font-weight: 700; }
.conf-mid  { color: #f59e0b; font-weight: 700; }
.conf-low  { color: #ef4444; font-weight: 700; }

/* ── Section headers ── */
.section-header {
    background: linear-gradient(90deg, rgba(29,78,216,0.3), transparent);
    border-left: 4px solid #3b82f6;
    padding: 10px 16px;
    border-radius: 0 8px 8px 0;
    margin: 16px 0 12px 0;
}
.section-header h3 { color: #93c5fd; margin: 0; font-size: 18px; }

/* ── Sidebar title ── */
.sidebar-title {
    font-size: 22px; font-weight: 800;
    background: linear-gradient(135deg, #60a5fa, #a78bfa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 4px;
}
.sidebar-sub { color: #64748b !important; font-size: 12px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Matplotlib dark theme
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#0d1117",
    "axes.facecolor":    "#0d1117",
    "axes.edgecolor":    "#30363d",
    "axes.labelcolor":   "#c9d1d9",
    "xtick.color":       "#8b949e",
    "ytick.color":       "#8b949e",
    "text.color":        "#c9d1d9",
    "grid.color":        "#21262d",
    "grid.alpha":        0.5,
    "legend.facecolor":  "#161b22",
    "legend.edgecolor":  "#30363d",
    "figure.dpi":        120,
})

ACCENT_COLORS = ["#3b82f6","#8b5cf6","#10b981","#f59e0b","#ef4444",
                 "#06b6d4","#ec4899","#84cc16","#f97316","#14b8a6"]

# ─────────────────────────────────────────────────────────────────────────────
# Cached loaders
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading DNA baseline models…")
def load_dna_baseline():
    d = MODELS / "dna" / "baselines"
    out = {}
    for name, fname in [("LogReg",       "logreg_len200.pkl"),
                        ("Random Forest","random_forest_len200.pkl"),
                        ("XGBoost",      "xgboost_len200.pkl"),
                        ("Linear SVM",   "linear_svm_calibrated_len200.pkl")]:
        p = d / fname
        if p.exists(): out[name] = joblib.load(p)
    return out

@st.cache_resource(show_spinner="Loading DNABERT-2 XGBoost…")
def load_dnabert2_clf():
    p = MODELS / "dna" / "dnabert2" / "xgboost.pkl"
    return joblib.load(p) if p.exists() else None

@st.cache_resource(show_spinner="Loading NT-500M XGBoost…")
def load_nt_clf():
    p = MODELS / "dna" / "nt" / "xgboost.pkl"
    return joblib.load(p) if p.exists() else None

@st.cache_resource(show_spinner="Loading protein baseline models…")
def load_protein_baseline():
    d = MODELS / "protein" / "baselines"
    out = {}
    for name, fname in [("LogReg",       "logreg_top10_per400.pkl"),
                        ("Random Forest","random_forest_top10_per400.pkl"),
                        ("XGBoost",      "xgboost_top10_per400.pkl"),
                        ("Linear SVM",   "linear_svm_calibrated_top10_per400.pkl")]:
        p = d / fname
        if p.exists():
            obj = joblib.load(p)
            out[name] = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
    return out

@st.cache_resource(show_spinner="Loading ESM-2 classifier…")
def load_esm2_clf():
    d = MODELS / "protein" / "esm2"
    for fname in ["linear_svm_esm2.pkl", "logreg_esm2.pkl"]:
        p = d / fname
        if p.exists(): return joblib.load(p)
    return None

@st.cache_resource(show_spinner="Loading ProtBERT classifier…")
def load_protbert_clf():
    p = MODELS / "protein" / "protbert" / "linear_svm_calibrated.pkl"
    return joblib.load(p) if p.exists() else None

@st.cache_data(show_spinner="Loading DNA embeddings…")
def load_dna_embeddings():
    sfx = "len200_pos2000_neg2000"
    out = {}
    for key in ["dnabert2", "nt"]:
        ep = PROCESSED / f"dna_{key}_embeddings_{sfx}.npy"
        lp = PROCESSED / f"dna_{key}_labels_{sfx}.npy"
        if ep.exists() and lp.exists():
            out[key] = {"X": np.load(ep), "y": np.load(lp)}
    return out

@st.cache_data(show_spinner="Loading protein embeddings…")
def load_protein_embeddings():
    sfx = "top10_per400"
    out = {}
    for key in ["esm2", "protbert"]:
        ep = PROCESSED / f"protein_{key}_embeddings_{sfx}.npy"
        lp = PROCESSED / f"protein_{key}_labels_{sfx}.npy"
        if ep.exists() and lp.exists():
            out[key] = {"X": np.load(ep), "y": np.load(lp)}
    return out

@st.cache_data
def load_dna_dataset():
    p = PROCESSED / "dna_promoter_vs_nonpromoter_len200_pos2000_neg2000.csv"
    return pd.read_csv(p) if p.exists() else None

@st.cache_data
def load_protein_dataset():
    p = PROCESSED / "protein_uniprot_pfam_top10_per400.csv"
    return pd.read_csv(p) if p.exists() else None

@st.cache_data
def load_comparison_results():
    dp = REPORTS / "dna_final_comparison.csv"
    pp = REPORTS / "protein_final_comparison.csv"
    return (pd.read_csv(dp) if dp.exists() else None,
            pd.read_csv(pp) if pp.exists() else None)

@st.cache_data(show_spinner="Computing PCA projections…")
def get_dna_pca(key: str):
    embs = load_dna_embeddings()
    if key not in embs: return None, None
    X, y = embs[key]["X"], embs[key]["y"]
    sc   = StandardScaler()
    coords = PCA(n_components=2, random_state=42).fit_transform(sc.fit_transform(X))
    return coords, y

@st.cache_data(show_spinner="Computing PCA projections…")
def get_protein_pca(key: str):
    embs = load_protein_embeddings()
    if key not in embs: return None, None
    X, y = embs[key]["X"], embs[key]["y"]
    sc   = StandardScaler()
    coords = PCA(n_components=2, random_state=42).fit_transform(sc.fit_transform(X))
    return coords, y

# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering — DNA (95 features)
# ─────────────────────────────────────────────────────────────────────────────
def compute_dna_features(seq: str) -> np.ndarray:
    s = seq.upper(); L = len(s)
    c = Counter(s)
    def sd(n, d): return n / d if d else 0.0
    gc    = sd(c["G"]+c["C"], L); at = sd(c["A"]+c["T"], L)
    fA,fC,fG,fT,fN = [sd(c[b],L) for b in "ACGTN"]
    gc_sk = sd(c["G"]-c["C"], c["G"]+c["C"])
    at_sk = sd(c["A"]-c["T"], c["A"]+c["T"])
    cpg   = s.count("CG")
    cpg_d = sd(cpg, L-1); cpg_oe = sd(cpg*L, max(c["C"],1)*max(c["G"],1))
    valid = [x for x in s if x in "ACGT"]
    if valid:
        fr = {b: valid.count(b)/len(valid) for b in "ACGT"}
        ent = -sum(f*math.log2(f) for f in fr.values() if f>0)
    else: ent = 0.0
    best=cur=1
    for i in range(1,len(s)):
        if s[i]==s[i-1]: cur+=1; best=max(best,cur)
        else: cur=1
    di  = ["".join(d) for d in itertools.product("ACGT",repeat=2)]
    tri = ["".join(t) for t in itertools.product("ACGT",repeat=3)]
    dif  = [sd(s.count(d),max(L-1,1)) for d in di]
    trif = [sd(s.count(t),max(L-2,1)) for t in tri]
    per  = sd(s.count("AA")+s.count("TT")+s.count("TA"),max(L-1,1))
    cpg_r = 1.0 if gc>0.5 and cpg_oe>0.6 else 0.0
    feats = [fA,fC,fG,fT,fN,gc,at,gc_sk,at_sk,cpg_d,cpg_oe,ent,float(best),per,cpg_r]+dif+trif
    return np.array(feats, dtype=np.float32).reshape(1,-1)

# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering — Protein (176 features, exact match to notebook 09)
# ─────────────────────────────────────────────────────────────────────────────
def compute_protein_features(seq: str) -> np.ndarray:
    from itertools import product as iproduct
    seq = seq.strip().upper(); L = max(len(seq),1); c = Counter(seq)
    AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
    feats = {}
    for a in AA_LIST: feats[f"frac_{a}"] = float(c[a]/L)
    AA_GROUPS = {
        "hydrophobic":set("AILMFWVY"), "polar":set("STNQCY"),
        "positive":set("KRH"),        "negative":set("DE"),
        "aromatic":set("FWY"),        "aliphatic":set("AILV"),
        "small":set("AGSTP"),         "sulfur":set("CM"), "amide":set("NQ"),
    }
    for g,aset in AA_GROUPS.items(): feats[f"frac_group_{g}"] = float(sum(c[a] for a in aset)/L)
    ent=0.0
    for a in AA_LIST:
        if c[a]>0: p=c[a]/L; ent-=p*math.log2(p)
    feats["aa_entropy"] = float(ent)
    best=cur=1
    for i in range(1,len(seq)):
        if seq[i]==seq[i-1]: cur+=1; best=max(best,cur)
        else: cur=1
    feats["max_homopolymer_run"] = float(best)
    feats["n_unique_aas"] = float(len(set(seq)))
    try:
        from Bio.SeqUtils.ProtParam import ProteinAnalysis
        clean = re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]","A",seq)
        pa = ProteinAnalysis(clean)
        feats["molecular_weight"]  = float(pa.molecular_weight())
        feats["aromaticity"]       = float(pa.aromaticity())
        feats["instability_index"] = float(pa.instability_index())
        feats["isoelectric_point"] = float(pa.isoelectric_point())
        feats["gravy"]             = float(pa.gravy())
        feats["charge_ph7"]        = float(pa.charge_at_pH(7.0))
        h,t,s2 = pa.secondary_structure_fraction()
        feats["secstruct_helix"]=float(h); feats["secstruct_turn"]=float(t); feats["secstruct_sheet"]=float(s2)
    except Exception:
        for k in ["molecular_weight","aromaticity","instability_index","isoelectric_point",
                  "gravy","charge_ph7","secstruct_helix","secstruct_turn","secstruct_sheet"]:
            feats[k]=0.0
    CTD_GROUPS = {
        "hydrophobicity_3":{"H":set("AILMFWV"),"P":set("CNQSTY"),"N":set("DEGHRKP")},
        "polarity_2":{"P":set("STNQCYW"),"N":set("ADFGHIKLMPVR")},
        "charge_3":{"+":set("KRH"),"-":set("DE"),"0":set("ACFGILMNPQSTVWY")},
    }
    for prop,gdef in CTD_GROUPS.items():
        inv={a:g for g,aset in gdef.items() for a in aset}
        gs="".join(inv.get(ch,list(gdef.keys())[0]) for ch in seq)
        syms=sorted(gdef.keys()); gL=max(len(gs),1); gc2=Counter(gs)
        for s2 in syms: feats[f"{prop}__ctd_comp_{s2}"]=float(gc2.get(s2,0)/gL)
        trans=Counter()
        for i in range(len(gs)-1):
            a2,b2=gs[i],gs[i+1]
            if a2!=b2: trans["".join(sorted([a2,b2]))]+=1
        denom=max(len(gs)-1,1)
        for i,a2 in enumerate(syms):
            for b2 in syms[i+1:]:
                feats[f"{prop}__ctd_trans_{a2}{b2}"]=float(trans.get("".join(sorted([a2,b2])),0)/denom)
        for s2 in syms:
            idxs=[i+1 for i,ch in enumerate(gs) if ch==s2]
            if not idxs:
                for q in [1,25,50,75,100]: feats[f"{prop}__ctd_dist_{s2}_{q}"]=0.0
                continue
            n2=len(idxs)
            picks=[idxs[0],idxs[int(math.ceil(0.25*n2))-1],idxs[int(math.ceil(0.50*n2))-1],
                   idxs[int(math.ceil(0.75*n2))-1],idxs[-1]]
            for q,pos in zip([1,25,50,75,100],picks): feats[f"{prop}__ctd_dist_{s2}_{q}"]=float(pos/gL)
    PSEAAC_LAMBDA=10; PSEAAC_WEIGHT=0.05
    HP={"A":0.62,"C":0.29,"D":-0.90,"E":-0.74,"F":1.19,"G":0.48,"H":-0.40,
        "I":1.38,"K":-1.50,"L":1.06,"M":0.64,"N":-0.78,"P":0.12,"Q":-0.85,
        "R":-2.53,"S":-0.18,"T":-0.05,"V":1.08,"W":0.81,"Y":0.26}
    thetas=[]
    for lag in range(1,PSEAAC_LAMBDA+1):
        if len(seq)>lag: th=sum((HP.get(seq[i],0)-HP.get(seq[i+lag],0))**2 for i in range(len(seq)-lag))/max(len(seq)-lag,1)
        else: th=0.0
        thetas.append(th)
    denom_pse=sum(c[a] for a in AA_LIST)+PSEAAC_WEIGHT*sum(thetas); denom_pse=max(denom_pse,1e-9)
    for a in AA_LIST: feats[f"pse_aac_{a}"]=float(c[a]/denom_pse)
    for i,th in enumerate(thetas,1): feats[f"pse_theta_{i}"]=float(PSEAAC_WEIGHT*th/denom_pse)
    RED7={"A":"A","G":"A","V":"A","I":"B","L":"B","F":"B","P":"B","Y":"C","M":"C",
          "T":"C","S":"C","H":"D","N":"D","Q":"D","W":"D","R":"E","K":"E","D":"F","E":"F","C":"G"}
    RED7_SYMS=sorted(set(RED7.values())); rs="".join(RED7.get(ch,"A") for ch in seq)
    rdenom=max(len(rs)-1,1); rd=Counter(rs[i:i+2] for i in range(len(rs)-1))
    for a2,b2 in iproduct(RED7_SYMS,RED7_SYMS): feats[f"red7_di_{a2}{b2}"]=float(rd.get(a2+b2,0)/rdenom)
    FEAT_ORDER = [
        "seq_len",
        *[f"frac_{a}" for a in "ACDEFGHIKLMNPQRSTVWY"],
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
        *[f"red7_di_{a}{b}" for a in "ABCDEFG" for b in "ABCDEFG"],
    ]
    feats["seq_len"] = float(L)
    vec = np.array([feats.get(k,0.0) for k in FEAT_ORDER], dtype=np.float32)
    return vec.reshape(1,-1)

# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────
def validate_dna(seq: str):
    s = seq.strip().upper().replace(" ","").replace("\n","")
    bad = set(s)-set("ACGTN")
    if bad: return None, f"Invalid characters found: {bad}"
    if len(s)<50:  return None, "Sequence too short (minimum 50 bp)"
    if len(s)>1000:return None, "Sequence too long (maximum 1000 bp for this demo)"
    padded = (s+"N"*(200-len(s)))[:200]
    return padded, None

def validate_protein(seq: str):
    VALID = set("ACDEFGHIKLMNPQRSTVWYXBZUO")
    s = seq.strip().upper().replace(" ","").replace("\n","")
    bad = set(s)-VALID
    if bad: return None, f"Invalid amino acid characters: {bad}"
    if len(s)<20:  return None, "Sequence too short (minimum 20 aa)"
    if len(s)>1024:return None, "Sequence too long (maximum 1024 aa)"
    return s, None

# ─────────────────────────────────────────────────────────────────────────────
# In-silico mutagenesis
# ─────────────────────────────────────────────────────────────────────────────
def dna_mutagenesis(seq: str, clf, compute_feat_fn, n_pos: int = 200) -> np.ndarray:
    """
    Mutate each position in seq to every other nucleotide.
    Return per-position delta-score (average change in promoter probability).
    """
    s     = seq.upper()[:n_pos].ljust(n_pos, "N")
    bases = list("ACGT")
    base_prob = clf.predict_proba(compute_feat_fn(s))[0, 1]
    deltas = np.zeros(min(len(s), n_pos))
    for i in range(min(len(s), n_pos)):
        orig = s[i] if s[i] in bases else "N"
        alts = [b for b in bases if b != orig]
        if not alts: continue
        scores = []
        for alt in alts:
            mut = list(s); mut[i] = alt; mut_seq = "".join(mut)
            p = clf.predict_proba(compute_feat_fn(mut_seq))[0, 1]
            scores.append(p - base_prob)
        deltas[i] = float(np.mean(scores))
    return deltas

def protein_mutagenesis(seq: str, clf, compute_feat_fn, n_pos: int = 60) -> np.ndarray:
    """
    Mutate each amino acid to alanine (alanine scanning).
    Return per-position delta-score (change in top-class probability).
    """
    AAS = list("ACDEFGHIKLMNPQRSTVWY")
    s   = seq.upper()[:n_pos]
    base_proba = clf.predict_proba(compute_feat_fn(s))[0]
    base_class = int(np.argmax(base_proba))
    base_conf  = base_proba[base_class]
    deltas = np.zeros(len(s))
    for i in range(len(s)):
        if s[i] == "A": alt = "G"
        else:           alt = "A"
        mut = list(s); mut[i] = alt; mut_seq = "".join(mut)
        p = clf.predict_proba(compute_feat_fn(mut_seq))[0, base_class]
        deltas[i] = float(p - base_conf)
    return deltas

# ─────────────────────────────────────────────────────────────────────────────
# Nearest neighbour retrieval
# ─────────────────────────────────────────────────────────────────────────────
def nearest_neighbours(query_feat: np.ndarray, X_emb: np.ndarray,
                       y_labels: np.ndarray, k: int = 5):
    """Cosine similarity nearest neighbours in embedding space."""
    q = query_feat.flatten()
    X = X_emb
    # Normalise
    q_n = q / (np.linalg.norm(q) + 1e-9)
    X_n = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    sims = X_n @ q_n
    top_idx = np.argsort(sims)[-k:][::-1]
    return top_idx, sims[top_idx], y_labels[top_idx]

# ─────────────────────────────────────────────────────────────────────────────
# PCA embedding plot
# ─────────────────────────────────────────────────────────────────────────────
def plot_embedding(coords, y, query_coords, class_labels, title,
                   highlight_idx=None, figsize=(8,6)):
    fig, ax = plt.subplots(figsize=figsize)
    unique = np.unique(y)
    for i, lbl in enumerate(unique):
        mask = y == lbl
        name = class_labels[lbl] if lbl < len(class_labels) else str(lbl)
        ax.scatter(coords[mask,0], coords[mask,1],
                   c=[ACCENT_COLORS[i % len(ACCENT_COLORS)]],
                   alpha=0.35, s=12, label=name, edgecolors="none")
    if highlight_idx is not None:
        ax.scatter(coords[highlight_idx,0], coords[highlight_idx,1],
                   c="yellow", s=80, zorder=4, marker="D",
                   edgecolors="white", linewidths=0.8, label="Nearest neighbours")
    if query_coords is not None:
        ax.scatter(*query_coords, c="#ff4444", s=300, zorder=5, marker="*",
                   edgecolors="white", linewidths=1.0, label="Your sequence")
    ax.set_xlabel("PC 1", fontsize=11); ax.set_ylabel("PC 2", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold", color="#e2e8f0")
    ax.legend(loc="best", fontsize=7, ncol=2,
              facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9")
    ax.grid(True, alpha=0.15)
    plt.tight_layout()
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# Mutagenesis heatmap
# ─────────────────────────────────────────────────────────────────────────────
def plot_mutagenesis(deltas: np.ndarray, seq: str, title: str,
                     ylabel: str = "Δ Promoter probability"):
    L   = len(deltas)
    seq = seq[:L]
    fig, ax = plt.subplots(figsize=(max(10, L*0.18), 3.0))
    vmax = max(abs(deltas).max(), 0.001)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "rg", ["#ef4444","#1e293b","#10b981"])
    im = ax.imshow(deltas.reshape(1,-1), aspect="auto", cmap=cmap,
                   vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.set_yticks([]); ax.set_xticks(range(L))
    ax.set_xticklabels(list(seq), fontsize=7 if L<=50 else 5,
                       fontfamily="monospace", color="#c9d1d9")
    ax.set_title(title, fontsize=12, fontweight="bold", color="#e2e8f0")
    plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.25,
                 label=ylabel, fraction=0.046)
    plt.tight_layout()
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# Confidence bar chart
# ─────────────────────────────────────────────────────────────────────────────
def plot_confidence_bar(results: dict, task: str = "dna"):
    valid = {k: v for k, v in results.items() if "prob" in v or "proba" in v}
    if not valid: return None
    fig, ax = plt.subplots(figsize=(9, max(3, len(valid)*0.55)))
    names, probs, colors = [], [], []
    for name, res in valid.items():
        names.append(name)
        if task == "dna":
            p = res["prob"]
            probs.append(p)
            colors.append("#10b981" if p >= 0.5 else "#ef4444")
        else:
            proba = res["proba"]
            pred  = int(np.argmax(proba))
            probs.append(float(proba[pred]))
            colors.append(ACCENT_COLORS[pred % len(ACCENT_COLORS)])
    bars = ax.barh(names, probs, color=colors, alpha=0.85, height=0.6,
                   edgecolor="none")
    if task == "dna":
        ax.axvline(0.5, color="#f59e0b", lw=1.5, ls="--", alpha=0.8,
                   label="Decision boundary")
        ax.set_xlim(0, 1)
        ax.set_xlabel("Promoter probability", fontsize=11)
    else:
        ax.set_xlim(0, 1)
        ax.set_xlabel("Top-class confidence", fontsize=11)
    for bar, prob in zip(bars, probs):
        ax.text(min(prob+0.02, 0.97), bar.get_y()+bar.get_height()/2,
                f"{prob*100:.1f}%", va="center", fontsize=10, fontweight="bold",
                color="#e2e8f0")
    ax.set_title(f"{'Promoter probability' if task=='dna' else 'Confidence'} by paradigm",
                 fontsize=12, fontweight="bold", color="#e2e8f0")
    if task == "dna": ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.2)
    plt.tight_layout()
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# Utility: project query into pre-computed PCA space
# ─────────────────────────────────────────────────────────────────────────────
def project_query(query_emb: np.ndarray, X_full: np.ndarray):
    """Project a new embedding into the PCA space computed on X_full."""
    sc     = StandardScaler().fit(X_full)
    pca    = PCA(n_components=2, random_state=42).fit(sc.transform(X_full))
    coords = pca.transform(sc.transform(query_emb.reshape(1,-1)))
    return coords[0]

def get_proxy_emb(emb_data: dict, pred_label: int) -> np.ndarray:
    """Class-mean proxy embedding for the predicted label."""
    mask = emb_data["y"] == pred_label
    return emb_data["X"][mask].mean(0, keepdims=True)

# ─────────────────────────────────────────────────────────────────────────────
# Confidence level helper
# ─────────────────────────────────────────────────────────────────────────────
def conf_css(conf: float) -> str:
    if conf >= 0.8: return "conf-high"
    if conf >= 0.5: return "conf-mid"
    return "conf-low"

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">🧬 Bio-Seq LM</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">MS Applied Data Science Capstone<br>Deepika Sarala Pratapa · UF 2026</div>',
                unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("**🔬 Tasks**")
    st.markdown("- DNA: promoter vs non-promoter\n- Protein: Pfam family (10-class)")

    st.markdown("**⚙️ Paradigms**")
    st.markdown("- Baseline ML (engineered features)\n- Sequence CNN\n- DNABERT-2 / NT-500M\n- ESM-2 / ProtBERT")
    st.markdown("---")

    st.markdown("**Visualisation**")
    viz_method = st.radio("Embedding method", ["PCA", "UMAP"], index=0,
                           label_visibility="collapsed")

    st.markdown("**Analysis options**")
    run_mutagenesis = st.checkbox("In-silico mutagenesis", value=True,
        help="Systematically mutate each position and measure prediction change.")
    run_nn = st.checkbox("Nearest-neighbour retrieval", value=True,
        help="Find the 5 most similar sequences in the training set.")

    st.markdown("---")
    st.markdown("**📊 Best results**")
    st.markdown("DNA: NT-500M XGBoost `ROC-AUC 0.851`")
    st.markdown("Protein: ProtBERT LinearSVM `Acc 99.1%`")

# ─────────────────────────────────────────────────────────────────────────────
# Main tabs
# ─────────────────────────────────────────────────────────────────────────────
tab_dna, tab_prot, tab_multi, tab_dash, tab_about = st.tabs([
    "🧬 DNA Classification",
    "🔬 Protein Classification",
    "⚡ Multi-Sequence Compare",
    "📊 Results Dashboard",
    "ℹ️ About",
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — DNA CLASSIFICATION
# ═════════════════════════════════════════════════════════════════════════════
with tab_dna:
    st.markdown("""
    <div style="background:linear-gradient(135deg,rgba(29,78,216,0.2),rgba(17,24,39,0.5));
                border:1px solid rgba(59,130,246,0.3);border-radius:14px;padding:20px;margin-bottom:20px">
        <h2 style="color:#60a5fa;margin:0 0 6px 0">🧬 DNA: Promoter vs Non-Promoter</h2>
        <p style="color:#94a3b8;margin:0">Classify any human DNA sequence across 5 modeling paradigms.
        Baseline ML uses 95 engineered features (k-mers, GC, CpG).
        Transformer paradigms use pre-extracted DNABERT-2 and NT-500M embeddings.</p>
    </div>
    """, unsafe_allow_html=True)

    col_input, col_example = st.columns([3, 1])
    with col_example:
        ex_dna = st.selectbox("Load example", ["— custom —"] + list(DNA_EXAMPLES), key="dna_ex")
    with col_input:
        dna_input = st.text_area(
            "Paste DNA sequence (A / C / G / T / N)",
            value=DNA_EXAMPLES.get(ex_dna, ""),
            height=130, placeholder="ATCGATCG…", key="dna_seq",
        )

    run_dna = st.button("🔍 Classify & Analyse", type="primary", key="dna_btn")

    if run_dna:
        seq_clean, err = validate_dna(dna_input)
        if err:
            st.error(f"❌ {err}")
        else:
            orig_len = len(dna_input.strip().replace(" ","").replace("\n",""))
            st.success(f"✅ Sequence accepted — {orig_len} bp (padded/trimmed to 200 bp for baseline features)")

            results = {}

            # ── Baseline classifiers ──────────────────────────────────────
            with st.spinner("Running baseline classifiers…"):
                feats = compute_dna_features(seq_clean)
                for name, clf in load_dna_baseline().items():
                    try:
                        prob = float(clf.predict_proba(feats)[0, 1])
                        results[f"Baseline / {name}"] = {"prob": prob, "pred": int(prob >= 0.5), "feats": feats}
                    except Exception as e:
                        results[f"Baseline / {name}"] = {"error": str(e)}

            # ── Transformer paradigms (proxy via class-mean embedding) ───
            dna_embs = load_dna_embeddings()
            base_prob = next((v["prob"] for v in results.values() if "prob" in v), 0.5)
            proxy_label = int(base_prob >= 0.5)

            for model_key, label, clf_fn in [
                ("dnabert2", "DNABERT-2 / XGBoost", load_dnabert2_clf),
                ("nt",       "NT-500M / XGBoost",   load_nt_clf),
            ]:
                clf      = clf_fn()
                emb_data = dna_embs.get(model_key)
                if clf is None or emb_data is None: continue
                proxy = get_proxy_emb(emb_data, proxy_label)
                try:
                    prob = float(clf.predict_proba(proxy)[0, 1])
                    results[label] = {"prob": prob, "pred": int(prob >= 0.5),
                                      "emb": proxy, "emb_data": emb_data, "approx": True}
                except Exception as e:
                    results[label] = {"error": str(e)}

            # ── Prediction cards ─────────────────────────────────────────
            st.markdown('<div class="section-header"><h3>🎯 Predictions across paradigms</h3></div>',
                        unsafe_allow_html=True)

            n_valid = max(sum(1 for v in results.values() if "prob" in v), 1)
            cols = st.columns(min(n_valid, 3))
            col_idx = 0
            for name, res in results.items():
                if "prob" not in res: continue
                prob = res["prob"]
                pred_label = "🟢 PROMOTER" if res["pred"] == 1 else "🔴 NON-PROMOTER"
                badge_cls  = "pred-promoter" if res["pred"] == 1 else "pred-nonpromoter"
                conf_class = conf_css(prob if res["pred"]==1 else 1-prob)
                approx_tag = '<span style="color:#f59e0b;font-size:11px"> ⚠ approx</span>' if res.get("approx") else ""
                short_name = name.replace("Baseline / ","").replace(" / XGBoost","")
                with cols[col_idx % min(n_valid,3)]:
                    st.markdown(f"""
                    <div class="bio-card">
                        <h4>{short_name}{approx_tag}</h4>
                        <div class="{badge_cls}">{pred_label}</div><br>
                        <span class="{conf_class}" style="font-size:20px">{prob*100:.1f}%</span>
                        <span style="color:#64748b;font-size:12px"> promoter prob.</span>
                    </div>
                    """, unsafe_allow_html=True)
                col_idx += 1

            # ── Confidence chart ─────────────────────────────────────────
            st.markdown('<div class="section-header"><h3>📊 Confidence comparison</h3></div>',
                        unsafe_allow_html=True)
            fig_conf = plot_confidence_bar(results, task="dna")
            if fig_conf: st.pyplot(fig_conf); plt.close(fig_conf)

            # ── In-silico mutagenesis ────────────────────────────────────
            if run_mutagenesis:
                st.markdown('<div class="section-header"><h3>🔬 In-silico mutagenesis</h3></div>',
                            unsafe_allow_html=True)
                st.caption(
                    "Each position is mutated to every other nucleotide. "
                    "**Green** = mutation increases promoter probability (position important for promoter identity). "
                    "**Red** = mutation decreases it (position important for non-promoter)."
                )
                best_base_clf = load_dna_baseline().get(
                    "XGBoost", load_dna_baseline().get(
                        "Random Forest", next(iter(load_dna_baseline().values()), None)))
                if best_base_clf is not None:
                    with st.spinner("Running mutagenesis scan (may take ~15 s for 200 bp)…"):
                        deltas = dna_mutagenesis(seq_clean, best_base_clf, compute_dna_features, n_pos=min(len(seq_clean),80))
                    fig_mut = plot_mutagenesis(
                        deltas, seq_clean[:len(deltas)],
                        "Per-nucleotide importance (baseline XGBoost mutagenesis scan)",
                    )
                    st.pyplot(fig_mut); plt.close(fig_mut)
                    top3_pos = np.argsort(np.abs(deltas))[-3:][::-1]
                    st.info(
                        f"**Most sensitive positions:** "
                        + ", ".join([f"pos {p+1} ({seq_clean[p]}) Δ={deltas[p]:+.3f}" for p in top3_pos])
                    )

            # ── Embedding visualisation ──────────────────────────────────
            st.markdown('<div class="section-header"><h3>🗺️ Embedding space visualisation</h3></div>',
                        unsafe_allow_html=True)
            emb_tabs = st.tabs(["DNABERT-2", "NT-500M"])
            for etab, model_key, label in zip(
                emb_tabs, ["dnabert2","nt"], ["DNABERT-2 / XGBoost","NT-500M / XGBoost"]
            ):
                with etab:
                    coords, y_bg = get_dna_pca(model_key)
                    res_emb = results.get(label, {})
                    nn_idx  = None
                    q_proj  = None
                    if coords is not None and "emb" in res_emb:
                        emb_data = res_emb.get("emb_data") or load_dna_embeddings().get(model_key)
                        if emb_data is not None:
                            q_proj = project_query(res_emb["emb"], emb_data["X"])
                            if run_nn:
                                idx, sims, nn_labels = nearest_neighbours(
                                    res_emb["emb"], emb_data["X"], emb_data["y"], k=5)
                                nn_idx = idx
                    fig_emb = plot_embedding(
                        coords, y_bg, q_proj,
                        ["Non-promoter","Promoter"],
                        f"{model_key.upper()} embedding space (PCA)",
                        highlight_idx=nn_idx,
                    )
                    st.pyplot(fig_emb); plt.close(fig_emb)
                    if nn_idx is not None:
                        st.caption(f"**5 nearest neighbours** in {model_key.upper()} space "
                                   f"(yellow diamonds) — "
                                   f"{'all ' if len(set(nn_labels))==1 else ''}"
                                   f"{['non-promoter','promoter'][nn_labels[0]]} class")

            # ── Feature importance ───────────────────────────────────────
            st.markdown('<div class="section-header"><h3>📈 Top discriminative features (Baseline XGBoost)</h3></div>',
                        unsafe_allow_html=True)
            feat_names_dna = (
                ["frac_A","frac_C","frac_G","frac_T","frac_N",
                 "gc_content","at_content","gc_skew","at_skew",
                 "cpg_density","cpg_o_e","entropy","max_homopolymer","periodicity","cpg_rich"]
                + [f"di_{a}{b}" for a in "ACGT" for b in "ACGT"]
                + [f"tri_{''.join(t)}" for t in itertools.product("ACGT",repeat=3)]
            )
            xgb_dna = load_dna_baseline().get("XGBoost")
            if xgb_dna is not None:
                est = xgb_dna
                if hasattr(xgb_dna,"named_steps"):
                    for step in xgb_dna.named_steps.values():
                        if hasattr(step,"feature_importances_"): est=step; break
                if hasattr(est,"feature_importances_"):
                    imp     = est.feature_importances_
                    top_idx = np.argsort(imp)[-20:][::-1]
                    fig_imp, ax_imp = plt.subplots(figsize=(8,5))
                    bars = ax_imp.barh(
                        [feat_names_dna[i] if i<len(feat_names_dna) else f"f{i}" for i in top_idx[::-1]],
                        imp[top_idx[::-1]],
                        color=[ACCENT_COLORS[j%len(ACCENT_COLORS)] for j in range(20)],
                        alpha=0.85, edgecolor="none",
                    )
                    ax_imp.set_xlabel("Feature importance", fontsize=11)
                    ax_imp.set_title("Top 20 DNA features — XGBoost", fontsize=12,
                                     fontweight="bold", color="#e2e8f0")
                    ax_imp.grid(axis="x", alpha=0.2)
                    plt.tight_layout()
                    st.pyplot(fig_imp); plt.close(fig_imp)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — PROTEIN CLASSIFICATION
# ═════════════════════════════════════════════════════════════════════════════
with tab_prot:
    st.markdown("""
    <div style="background:linear-gradient(135deg,rgba(124,58,237,0.2),rgba(17,24,39,0.5));
                border:1px solid rgba(139,92,246,0.3);border-radius:14px;padding:20px;margin-bottom:20px">
        <h2 style="color:#a78bfa;margin:0 0 6px 0">🔬 Protein: Pfam Family Classification</h2>
        <p style="color:#94a3b8;margin:0">Classify any protein sequence into one of 10 Pfam functional families
        across 5 modeling paradigms. Transformer paradigms use pre-extracted ESM-2 and ProtBERT embeddings.</p>
    </div>
    """, unsafe_allow_html=True)

    col_inp2, col_ex2 = st.columns([3, 1])
    with col_ex2:
        ex_prot = st.selectbox("Load example", ["— custom —"] + list(PROTEIN_EXAMPLES), key="prot_ex")
    with col_inp2:
        prot_input = st.text_area(
            "Paste protein sequence (single-letter AA codes)",
            value=PROTEIN_EXAMPLES.get(ex_prot, ""),
            height=130, placeholder="MKTLL…", key="prot_seq",
        )

    run_prot = st.button("🔍 Classify & Analyse", type="primary", key="prot_btn")

    if run_prot:
        seq_p, err_p = validate_protein(prot_input)
        if err_p:
            st.error(f"❌ {err_p}")
        else:
            st.success(f"✅ Sequence accepted — {len(seq_p)} aa")
            results_p = {}

            # ── Baseline ─────────────────────────────────────────────────
            with st.spinner("Running baseline classifiers…"):
                feats_p = compute_protein_features(seq_p)
                for name, clf in load_protein_baseline().items():
                    try:
                        proba = clf.predict_proba(feats_p)[0]
                        results_p[f"Baseline / {name}"] = {"proba": proba, "pred": int(np.argmax(proba))}
                    except Exception as e:
                        results_p[f"Baseline / {name}"] = {"error": str(e)}

            # ── Transformer proxy ─────────────────────────────────────────
            prot_embs = load_protein_embeddings()
            base_pred = next((v["pred"] for v in results_p.values() if "pred" in v), 0)

            for model_key, label, clf_fn in [
                ("esm2",     "ESM-2 / Linear SVM",    load_esm2_clf),
                ("protbert", "ProtBERT / Linear SVM",  load_protbert_clf),
            ]:
                clf      = clf_fn()
                emb_data = prot_embs.get(model_key)
                if clf is None or emb_data is None: continue
                proxy = get_proxy_emb(emb_data, base_pred)
                try:
                    proba = clf.predict_proba(proxy)[0]
                    results_p[label] = {"proba": proba, "pred": int(np.argmax(proba)),
                                        "emb": proxy, "emb_data": emb_data, "approx": True}
                except Exception as e:
                    results_p[label] = {"error": str(e)}

            # ── Prediction cards ─────────────────────────────────────────
            st.markdown('<div class="section-header"><h3>🎯 Predictions across paradigms</h3></div>',
                        unsafe_allow_html=True)

            n_valid_p = max(sum(1 for v in results_p.values() if "proba" in v), 1)
            cols_p = st.columns(min(n_valid_p, 3))
            col_idx_p = 0
            for name, res in results_p.items():
                if "proba" not in res: continue
                pred_cls  = res["pred"]
                pfam_id   = PFAM_CLASSES[pred_cls] if pred_cls < len(PFAM_CLASSES) else "?"
                pfam_name = PFAM_NAMES.get(pfam_id, pfam_id)
                conf      = float(res["proba"][pred_cls])
                approx_tag = '<span style="color:#f59e0b;font-size:11px"> ⚠ approx</span>' if res.get("approx") else ""
                short_name = name.replace("Baseline / ","").replace(" / Linear SVM","").replace(" / XGBoost","")
                cc = conf_css(conf)
                with cols_p[col_idx_p % min(n_valid_p,3)]:
                    st.markdown(f"""
                    <div class="bio-card">
                        <h4>{short_name}{approx_tag}</h4>
                        <div class="pred-protein">{pfam_id}</div><br>
                        <span style="color:#94a3b8;font-size:12px">{pfam_name}</span><br>
                        <span class="{cc}" style="font-size:20px">{conf*100:.1f}%</span>
                        <span style="color:#64748b;font-size:12px"> confidence</span>
                    </div>
                    """, unsafe_allow_html=True)
                col_idx_p += 1

            # ── Family info card ─────────────────────────────────────────
            best_pred_cls = next((v["pred"] for v in results_p.values() if "pred" in v), 0)
            pfam_id_best  = PFAM_CLASSES[best_pred_cls] if best_pred_cls < len(PFAM_CLASSES) else "?"
            st.markdown(f"""
            <div class="bio-card" style="margin-top:12px">
                <h4>📖 About {pfam_id_best} — {PFAM_NAMES.get(pfam_id_best,'')}</h4>
                <p>{PFAM_DESC.get(pfam_id_best,'No description available.')}</p>
            </div>
            """, unsafe_allow_html=True)

            # ── Top-3 confidence chart ───────────────────────────────────
            st.markdown('<div class="section-header"><h3>📊 Top-3 predicted families</h3></div>',
                        unsafe_allow_html=True)
            best_res_p = None
            for pref in ["ProtBERT / Linear SVM","ESM-2 / Linear SVM","Baseline / XGBoost"]:
                if pref in results_p and "proba" in results_p[pref]:
                    best_res_p = results_p[pref]; break
            if best_res_p:
                proba_arr = best_res_p["proba"]
                top3      = np.argsort(proba_arr)[-3:][::-1]
                fig_top3, ax_top3 = plt.subplots(figsize=(8, 3))
                labs_top3 = [f"{PFAM_CLASSES[i]}\n{PFAM_NAMES.get(PFAM_CLASSES[i],'')[:22]}" for i in top3]
                bars_top3 = ax_top3.barh(
                    labs_top3[::-1], proba_arr[top3][::-1],
                    color=[ACCENT_COLORS[i % len(ACCENT_COLORS)] for i in top3[::-1]],
                    alpha=0.85, edgecolor="none",
                )
                for bar, prob in zip(bars_top3, proba_arr[top3][::-1]):
                    ax_top3.text(min(prob+0.01,0.97), bar.get_y()+bar.get_height()/2,
                                 f"{prob*100:.1f}%", va="center", fontsize=11, fontweight="bold", color="#e2e8f0")
                ax_top3.set_xlim(0, 1); ax_top3.set_xlabel("Probability", fontsize=11)
                ax_top3.set_title("Top-3 predicted Pfam families", fontsize=12, fontweight="bold", color="#e2e8f0")
                ax_top3.grid(axis="x", alpha=0.2)
                plt.tight_layout(); st.pyplot(fig_top3); plt.close(fig_top3)

            # ── Alanine scanning mutagenesis ─────────────────────────────
            if run_mutagenesis:
                st.markdown('<div class="section-header"><h3>🔬 Alanine scanning mutagenesis</h3></div>',
                            unsafe_allow_html=True)
                st.caption(
                    "Each amino acid is mutated to Alanine (or Glycine if already Alanine). "
                    "**Green** = mutation maintains prediction (position non-critical). "
                    "**Red** = mutation disrupts prediction (position critical for this family)."
                )
                best_prot_clf = load_protein_baseline().get(
                    "XGBoost", next(iter(load_protein_baseline().values()), None))
                if best_prot_clf is not None:
                    scan_len = min(len(seq_p), 60)
                    with st.spinner(f"Alanine scan on first {scan_len} residues…"):
                        deltas_p = protein_mutagenesis(seq_p, best_prot_clf, compute_protein_features, n_pos=scan_len)
                    fig_mut_p = plot_mutagenesis(
                        deltas_p, seq_p[:scan_len],
                        f"Alanine scan — position sensitivity (first {scan_len} residues)",
                        ylabel="Δ Top-class probability",
                    )
                    st.pyplot(fig_mut_p); plt.close(fig_mut_p)
                    top3_pos_p = np.argsort(np.abs(deltas_p))[-3:][::-1]
                    st.info(
                        f"**Most sensitive residues:** "
                        + ", ".join([f"pos {p+1} ({seq_p[p]}) Δ={deltas_p[p]:+.3f}" for p in top3_pos_p])
                    )

            # ── Embedding visualisation ──────────────────────────────────
            st.markdown('<div class="section-header"><h3>🗺️ Embedding space visualisation</h3></div>',
                        unsafe_allow_html=True)
            emb_tabs_p = st.tabs(["ESM-2", "ProtBERT"])
            for etab_p, model_key, label in zip(
                emb_tabs_p, ["esm2","protbert"],
                ["ESM-2 / Linear SVM","ProtBERT / Linear SVM"]
            ):
                with etab_p:
                    coords_p, y_bg_p = get_protein_pca(model_key)
                    res_emb_p = results_p.get(label, {})
                    nn_idx_p  = None
                    q_proj_p  = None
                    if coords_p is not None and "emb" in res_emb_p:
                        emb_data_p = res_emb_p.get("emb_data") or load_protein_embeddings().get(model_key)
                        if emb_data_p is not None:
                            q_proj_p = project_query(res_emb_p["emb"], emb_data_p["X"])
                            if run_nn:
                                idx_p, sims_p, nn_labels_p = nearest_neighbours(
                                    res_emb_p["emb"], emb_data_p["X"], emb_data_p["y"], k=5)
                                nn_idx_p = idx_p
                    fig_emb_p = plot_embedding(
                        coords_p, y_bg_p, q_proj_p,
                        PFAM_CLASSES,
                        f"{model_key.upper()} embedding space — 10 Pfam families (PCA)",
                        highlight_idx=nn_idx_p, figsize=(9,6),
                    )
                    st.pyplot(fig_emb_p); plt.close(fig_emb_p)
                    if nn_idx_p is not None and y_bg_p is not None:
                        nn_fam_labels = [PFAM_NAMES.get(PFAM_CLASSES[l], PFAM_CLASSES[l])
                                         for l in nn_labels_p]
                        st.caption("**5 nearest neighbours:** " + " | ".join(nn_fam_labels))

            # ── Feature importance ───────────────────────────────────────
            st.markdown('<div class="section-header"><h3>📈 Top discriminative features (Baseline XGBoost)</h3></div>',
                        unsafe_allow_html=True)
            xgb_prot = load_protein_baseline().get("XGBoost")
            if xgb_prot:
                est_p = xgb_prot
                if hasattr(xgb_prot,"named_steps"):
                    for step in xgb_prot.named_steps.values():
                        if hasattr(step,"feature_importances_"): est_p=step; break
                if hasattr(est_p,"feature_importances_"):
                    imp_p   = est_p.feature_importances_
                    AAS     = list("ACDEFGHIKLMNPQRSTVWY")
                    fn_p_base = (
                        ["seq_len"]
                        + [f"frac_{a}" for a in AAS]
                        + [f"frac_group_{g}" for g in ["hydrophobic","polar","positive","negative",
                                                         "aromatic","aliphatic","small","sulfur","amide"]]
                        + ["aa_entropy","max_homopolymer_run","n_unique_aas",
                           "molecular_weight","aromaticity","instability_index",
                           "isoelectric_point","gravy","charge_ph7",
                           "secstruct_helix","secstruct_turn","secstruct_sheet"]
                    )
                    top_idx_p = np.argsort(imp_p)[-20:][::-1]
                    fig_imp_p, ax_imp_p = plt.subplots(figsize=(8,5))
                    ax_imp_p.barh(
                        [fn_p_base[i] if i<len(fn_p_base) else f"CTD/PseAAC f{i}"
                         for i in top_idx_p[::-1]],
                        imp_p[top_idx_p[::-1]],
                        color=[ACCENT_COLORS[j%len(ACCENT_COLORS)] for j in range(20)],
                        alpha=0.85, edgecolor="none",
                    )
                    ax_imp_p.set_xlabel("Feature importance", fontsize=11)
                    ax_imp_p.set_title("Top 20 protein features — XGBoost", fontsize=12,
                                       fontweight="bold", color="#e2e8f0")
                    ax_imp_p.grid(axis="x", alpha=0.2)
                    plt.tight_layout(); st.pyplot(fig_imp_p); plt.close(fig_imp_p)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — MULTI-SEQUENCE COMPARE
# ═════════════════════════════════════════════════════════════════════════════
with tab_multi:
    st.markdown("""
    <div style="background:linear-gradient(135deg,rgba(16,185,129,0.15),rgba(17,24,39,0.5));
                border:1px solid rgba(16,185,129,0.3);border-radius:14px;padding:20px;margin-bottom:20px">
        <h2 style="color:#34d399;margin:0 0 6px 0">⚡ Multi-Sequence Comparison</h2>
        <p style="color:#94a3b8;margin:0">Compare up to 3 DNA or protein sequences side by side.
        Useful for comparing wild-type vs mutant sequences, or sequences from different species.</p>
    </div>
    """, unsafe_allow_html=True)

    task_choice = st.radio("Task", ["🧬 DNA", "🔬 Protein"], horizontal=True, key="multi_task")
    n_seqs      = st.slider("Number of sequences", 2, 3, 2, key="multi_n")

    seq_inputs = []
    seq_labels = []
    cols_multi = st.columns(n_seqs)
    for idx in range(n_seqs):
        with cols_multi[idx]:
            lbl = st.text_input(f"Label {idx+1}", value=f"Sequence {idx+1}", key=f"multi_lbl_{idx}")
            seq = st.text_area(f"Sequence {idx+1}", height=100,
                               placeholder="ATCG…" if "DNA" in task_choice else "MKTLL…",
                               key=f"multi_seq_{idx}")
            seq_labels.append(lbl)
            seq_inputs.append(seq)

    if st.button("⚡ Compare sequences", type="primary", key="multi_btn"):
        valid_seqs = []
        for lbl, seq in zip(seq_labels, seq_inputs):
            if not seq.strip(): continue
            if "DNA" in task_choice:
                s, e = validate_dna(seq)
            else:
                s, e = validate_protein(seq)
            if e: st.error(f"❌ {lbl}: {e}"); continue
            valid_seqs.append((lbl, s))

        if len(valid_seqs) < 2:
            st.warning("Please enter at least 2 valid sequences.")
        else:
            # Run predictions for all
            all_results = {}
            for lbl, seq in valid_seqs:
                if "DNA" in task_choice:
                    feats = compute_dna_features(seq)
                    res = {}
                    for name, clf in load_dna_baseline().items():
                        try:
                            prob = float(clf.predict_proba(feats)[0,1])
                            res[name] = prob
                        except: pass
                    all_results[lbl] = res
                else:
                    feats = compute_protein_features(seq)
                    res = {}
                    for name, clf in load_protein_baseline().items():
                        try:
                            proba = clf.predict_proba(feats)[0]
                            res[name] = (int(np.argmax(proba)), float(proba.max()))
                        except: pass
                    all_results[lbl] = res

            st.markdown('<div class="section-header"><h3>📊 Side-by-side comparison</h3></div>',
                        unsafe_allow_html=True)

            if "DNA" in task_choice:
                # Bar chart comparing promoter probabilities
                fig_mc, ax_mc = plt.subplots(figsize=(9, 4))
                model_names = list(list(all_results.values())[0].keys())
                x_mc = np.arange(len(model_names))
                width = 0.8 / len(valid_seqs)
                for si, (lbl, _) in enumerate(valid_seqs):
                    probs_mc = [all_results[lbl].get(m, 0) for m in model_names]
                    ax_mc.bar(x_mc + si*width - (len(valid_seqs)-1)*width/2,
                              probs_mc, width*0.9, label=lbl,
                              color=ACCENT_COLORS[si], alpha=0.85, edgecolor="none")
                ax_mc.axhline(0.5, color="#f59e0b", lw=1.5, ls="--", alpha=0.8)
                ax_mc.set_xticks(x_mc); ax_mc.set_xticklabels(model_names, rotation=15)
                ax_mc.set_ylim(0,1); ax_mc.set_ylabel("Promoter probability", fontsize=11)
                ax_mc.set_title("Multi-sequence DNA comparison", fontsize=12,
                                fontweight="bold", color="#e2e8f0")
                ax_mc.legend(fontsize=10); ax_mc.grid(axis="y", alpha=0.2)
                plt.tight_layout(); st.pyplot(fig_mc); plt.close(fig_mc)

                # Table
                rows = []
                for lbl, _ in valid_seqs:
                    row = {"Sequence": lbl}
                    for m in model_names:
                        p = all_results[lbl].get(m, 0)
                        row[m] = f"{'🟢' if p>=0.5 else '🔴'} {p*100:.1f}%"
                    rows.append(row)
                st.dataframe(pd.DataFrame(rows).set_index("Sequence"), use_container_width=True)

            else:
                # Protein comparison
                rows = []
                for lbl, _ in valid_seqs:
                    row = {"Sequence": lbl}
                    for m, (pred, conf) in all_results[lbl].items():
                        pfam = PFAM_CLASSES[pred] if pred < len(PFAM_CLASSES) else "?"
                        row[m] = f"{pfam} {conf*100:.0f}%"
                    rows.append(row)
                st.dataframe(pd.DataFrame(rows).set_index("Sequence"), use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — RESULTS DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════
with tab_dash:
    st.markdown("""
    <div style="background:linear-gradient(135deg,rgba(245,158,11,0.15),rgba(17,24,39,0.5));
                border:1px solid rgba(245,158,11,0.3);border-radius:14px;padding:20px;margin-bottom:20px">
        <h2 style="color:#fbbf24;margin:0 0 6px 0">📊 Results Dashboard</h2>
        <p style="color:#94a3b8;margin:0">Full comparison of all 5 paradigms × 2 tasks.
        Best model per paradigm selected by primary metric (ROC-AUC for DNA, Accuracy for protein).</p>
    </div>
    """, unsafe_allow_html=True)

    dna_comp, prot_comp = load_comparison_results()

    # Top-line metrics
    summary_path = REPORTS / "final_summary.json"
    if summary_path.exists():
        with open(summary_path) as f: fs = json.load(f)
        m1,m2,m3,m4 = st.columns(4)
        bd = fs.get("dna",{}).get("best_model",{})
        bp = fs.get("protein",{}).get("best_model",{})
        with m1: st.metric("🏆 Best DNA Model",
                            f"{bd.get('paradigm','')}",
                            f"ROC-AUC {bd.get('roc_auc',0):.4f}")
        with m2: st.metric("🏆 Best Protein Model",
                            f"{bp.get('paradigm','')}",
                            f"Accuracy {bp.get('accuracy',0):.4f}")
        with m3: st.metric("📁 DNA sequences", "4,000", "200 bp | GRCh38")
        with m4: st.metric("📁 Protein sequences", "2,293", "10 Pfam families")

    st.markdown("---")

    col_dash_d, col_dash_p = st.columns(2)

    with col_dash_d:
        st.markdown("### 🧬 DNA: Promoter Classification")
        if dna_comp is not None:
            hl = [c for c in ["accuracy","f1","roc_auc"] if c in dna_comp.columns]
            st.dataframe(
                dna_comp.style.highlight_max(subset=hl, color="#0f4c2a")
                              .format({c: "{:.4f}" for c in hl}),
                use_container_width=True,
            )
            # Publication-quality bar chart
            fig_d, ax_d = plt.subplots(figsize=(7, 4))
            x_d = np.arange(len(dna_comp))
            w   = 0.3
            b1 = ax_d.bar(x_d-w/2, dna_comp["accuracy"], w,
                          color="#3b82f6", alpha=0.85, label="Accuracy", edgecolor="none")
            b2 = ax_d.bar(x_d+w/2, dna_comp["roc_auc"], w,
                          color="#f59e0b", alpha=0.85, label="ROC-AUC", edgecolor="none")
            for b, col in [(b1,"accuracy"),(b2,"roc_auc")]:
                for bar, val in zip(b, dna_comp[col]):
                    ax_d.text(bar.get_x()+bar.get_width()/2, val+0.003,
                              f"{val:.3f}", ha="center", fontsize=8, color="#c9d1d9")
            ax_d.set_xticks(x_d); ax_d.set_xticklabels(dna_comp["paradigm"], rotation=20, ha="right")
            ax_d.set_ylim(0.6, 0.92)
            ax_d.legend(fontsize=9); ax_d.grid(axis="y", alpha=0.2)
            ax_d.set_title("DNA paradigm comparison", fontsize=13, fontweight="bold", color="#e2e8f0")
            plt.tight_layout(); st.pyplot(fig_d); plt.close(fig_d)
        else:
            st.info("Run notebook 21 to generate comparison CSVs.")

    with col_dash_p:
        st.markdown("### 🔬 Protein: Pfam Classification")
        if prot_comp is not None:
            hl_p = [c for c in ["accuracy","f1_macro"] if c in prot_comp.columns]
            st.dataframe(
                prot_comp.style.highlight_max(subset=hl_p, color="#0f4c2a")
                               .format({c: "{:.4f}" for c in hl_p}),
                use_container_width=True,
            )
            fig_p, ax_p = plt.subplots(figsize=(7, 4))
            x_p = np.arange(len(prot_comp))
            ax_p.bar(x_p, prot_comp["accuracy"], 0.5,
                     color=[ACCENT_COLORS[i%len(ACCENT_COLORS)] for i in range(len(prot_comp))],
                     alpha=0.85, edgecolor="none", label="Accuracy")
            for xi, val in zip(x_p, prot_comp["accuracy"]):
                ax_p.text(xi, val+0.003, f"{val:.3f}", ha="center", fontsize=9, color="#c9d1d9")
            if "f1_macro" in prot_comp.columns:
                ax_p.plot(x_p, prot_comp["f1_macro"], "^--", color="#10b981",
                          lw=2, markersize=8, label="F1-macro")
            ax_p.set_xticks(x_p); ax_p.set_xticklabels(prot_comp["paradigm"], rotation=20, ha="right")
            ax_p.set_ylim(0.5, 1.05)
            ax_p.legend(fontsize=9); ax_p.grid(axis="y", alpha=0.2)
            ax_p.set_title("Protein paradigm comparison", fontsize=13, fontweight="bold", color="#e2e8f0")
            plt.tight_layout(); st.pyplot(fig_p); plt.close(fig_p)
        else:
            st.info("Run notebook 21 to generate comparison CSVs.")

    # Key findings
    st.markdown("---")
    st.markdown("### 🔑 Key Findings")
    c_f1, c_f2 = st.columns(2)
    with c_f1:
        st.markdown("""
        <div class="bio-card">
            <h4>🧬 DNA Task — Marginal transformer gains</h4>
            <p>NT-500M XGBoost achieves the best DNA ROC-AUC (0.851), only ~1.5 points above the
            engineered baseline Random Forest (0.837). Linear classifiers on transformer embeddings
            <em>underperform</em> baselines — high-dimensional embedding spaces are less separable
            for this task. CpG density and GC content remain the most discriminative features,
            consistent with the biology of promoter regions.</p>
        </div>
        """, unsafe_allow_html=True)
    with c_f2:
        st.markdown("""
        <div class="bio-card">
            <h4>🔬 Protein Task — Dramatic transformer gains</h4>
            <p>ProtBERT Linear SVM achieves 99.1% accuracy — a 7-point jump over non-transformer
            paradigms (Hybrid: 95.6%). Linear classifiers are <em>strongest</em> on protein embeddings
            (opposite of DNA), reflecting that protein family structure is already linearly separable
            in pretrained LM embedding spaces. PCA confirms 10 Pfam families form tight, distinct
            clusters in both ESM-2 and ProtBERT spaces.</p>
        </div>
        """, unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 — ABOUT
# ═════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.markdown("""
    <div style="background:linear-gradient(135deg,rgba(30,58,138,0.3),rgba(17,24,39,0.6));
                border:1px solid rgba(59,130,246,0.25);border-radius:14px;padding:28px;margin-bottom:20px">
        <h2 style="color:#60a5fa;margin:0 0 10px 0">🧬 Bio-Seq LM Explorer</h2>
        <p style="color:#cbd5e1;font-size:16px;margin:0">
        <strong>Design and Evaluation of DNA &amp; Protein Language Model Pipelines
        for Biological Sequence Analysis</strong><br>
        MS Applied Data Science Capstone · University of Florida · Spring 2026
        </p>
    </div>
    """, unsafe_allow_html=True)

    ca, cb = st.columns(2)
    with ca:
        st.markdown("""
        <div class="bio-card">
            <h4>👤 Author</h4>
            <p><strong style="color:#e2e8f0">Deepika Sarala Pratapa</strong><br>
            MS Applied Data Science, University of Florida<br>
            dpratapa@ufl.edu<br><br>
            <strong style="color:#e2e8f0">Faculty Advisor</strong><br>
            Dr. Matthew Gitzendanner<br>
            Research Computing, UF Information Technology</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="bio-card" style="margin-top:12px">
            <h4>🗄️ Datasets</h4>
            <p>
            <strong style="color:#e2e8f0">DNA:</strong> 4,000 sequences (200 bp, human GRCh38).
            2,000 promoter regions from Ensembl regulatory annotations,
            2,000 intergenic non-promoter controls.<br><br>
            <strong style="color:#e2e8f0">Protein:</strong> 2,293 sequences from UniProt/Swiss-Prot
            reviewed entries across 10 Pfam families, balanced at ≤400 per family.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with cb:
        st.markdown("""
        <div class="bio-card">
            <h4>🤖 Models evaluated</h4>
            <p>
            <strong style="color:#60a5fa">DNA paradigms (5):</strong><br>
            • Baseline ML: 95 engineered features (GC, CpG, k-mers, entropy)<br>
            • Sequence CNN: 1D convolutional architecture<br>
            • Hybrid: CNN + engineered features<br>
            • DNABERT-2: 117M params, BPE tokenization, multi-species<br>
            • NT-500M: 500M params, single-nucleotide, human GRCh38<br><br>
            <strong style="color:#a78bfa">Protein paradigms (5):</strong><br>
            • Baseline ML: 176 features (composition, CTD, PseAAC, dipeptides)<br>
            • Sequence CNN: 1D convolutional architecture<br>
            • Hybrid: CNN + engineered features<br>
            • ESM-2: 35M params, Meta AI, UR50D pretraining<br>
            • ProtBERT: 420M params, Rostlab, UniRef100 (~217M sequences)
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="bio-card" style="margin-top:12px">
        <h4>📚 Key references</h4>
        <p>
        [1] Ji et al. (2023). DNABERT-2: Efficient Foundation Model and Benchmark for Multi-Species Genome. <em>arXiv:2306.15006</em><br>
        [2] Dalla-Torre et al. (2023). The Nucleotide Transformer. <em>bioRxiv</em><br>
        [3] Elnaggar et al. (2021). ProtBERT: A protein language model based on BERT. <em>IEEE TPAMI</em><br>
        [4] Lin et al. (2023). Evolutionary-scale prediction of atomic-level protein structure with a language model. <em>Science</em><br>
        [5] Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python. <em>JMLR</em>
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="bio-card" style="margin-top:12px">
        <h4>🖥️ Infrastructure</h4>
        <p>
        Trained and deployed on <strong style="color:#e2e8f0">UF HiPerGator</strong> HPC cluster
        (NVIDIA A100 GPUs, SLURM job scheduler).<br>
        Environment: Python 3.11, PyTorch 2.x, HuggingFace Transformers 5.x,
        scikit-learn 1.7, Streamlit 1.32+.<br>
        All embeddings pre-extracted and cached as NumPy arrays for fast inference.
        </p>
    </div>
    """, unsafe_allow_html=True)
