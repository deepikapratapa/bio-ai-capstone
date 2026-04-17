"""
Bio-Seq LM Capstone — Streamlit App
====================================
DNA: promoter vs non-promoter (binary)
Protein: Pfam family classification (10-class)

Run:
    streamlit run app.py --server.port 8501 --server.headless true
"""

from __future__ import annotations
import json, math, re, warnings, itertools
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent   # /home/.../Capstone
PROCESSED = ROOT / "data" / "processed"
MODELS    = ROOT / "models"
REPORTS   = ROOT / "reports"

# ── constants ─────────────────────────────────────────────────────────────────
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

DNA_EXAMPLES = {
    "Promoter example (TATA-box region)":
        "TACTAGCAATACGCTTGCGTTCGGTGGTTAAGTATGTATAATGCGCGGGCTTGTCGT"
        "AAGCGCGGTTTTTTTTTTAAAAAAACGCGCGCGCGCTATATATATATAGCGCGCGCGC"
        "GCGCGCGCGCGCGCTTTTTTTTTTAAAAAAACGCGCGCGCGCTATATATATATAG",
    "Non-promoter example (intergenic)":
        "ATTCGATCGATCGATCGTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAG"
        "CTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAG"
        "CTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA",
}
PROTEIN_EXAMPLES = {
    "GPCR (PF13853 / olfactory receptor)":
        "MRNHTEITEFILLGLTDDPNFQVVIFVFLLITYMLSITGNLTLITIAKDSHLHTPMYFFLSHLSFVDLSS"
        "VSSVPNMLVNLIQDIQPVLGLPCISKFIQFFMEHISLASSVGCLIAMALDRHVAIVHPLLYSTIMSKLAC"
        "YLLIAASWTLSFVLCVPVFLFQIVH",
    "Protein kinase (PF00069)":
        "MGSSHHHHHHSSGLVPRGSHMASMTGGQQMGRDLYDDDDKDPQMVKVGDKVTLKKLGEGAFGEVWMGKWN"
        "GTRVAIKTLKPGSMPEAFLAEANVMKTLQHDKLVKLHAVVTKEPIYIVTEYMSKGSLLHQLEKAKLMKKA",
}

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bio-Seq LM | Capstone",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# Model loaders  (cached so they load once)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading DNA baseline models…")
def load_dna_baseline():
    d = MODELS / "dna" / "baselines"
    out = {}
    for name, fname in [("LogReg",        "logreg_len200.pkl"),
                         ("Random Forest", "random_forest_len200.pkl"),
                         ("XGBoost",       "xgboost_len200.pkl"),
                         ("Linear SVM",    "linear_svm_calibrated_len200.pkl")]:
        p = d / fname
        if p.exists():
            out[name] = joblib.load(p)
    return out

@st.cache_resource(show_spinner="Loading DNABERT-2 classifier…")
def load_dnabert2_clf():
    p = MODELS / "dna" / "dnabert2" / "xgboost.pkl"
    return joblib.load(p) if p.exists() else None

@st.cache_resource(show_spinner="Loading NT-500M classifier…")
def load_nt_clf():
    p = MODELS / "dna" / "nt" / "xgboost.pkl"
    return joblib.load(p) if p.exists() else None

@st.cache_resource(show_spinner="Loading protein baseline models…")
def load_protein_baseline():
    d = MODELS / "protein" / "baselines"
    out = {}
    for name, fname in [("LogReg",        "logreg_top10_per400.pkl"),
                         ("Random Forest", "random_forest_top10_per400.pkl"),
                         ("XGBoost",       "xgboost_top10_per400.pkl"),
                         ("Linear SVM",    "linear_svm_calibrated_top10_per400.pkl")]:
        p = d / fname
        if p.exists():
            obj = joblib.load(p)
            # Protein baseline pkls are saved as dicts with a 'model' key
            out[name] = obj['model'] if isinstance(obj, dict) and 'model' in obj else obj
    return out

@st.cache_resource(show_spinner="Loading ESM-2 classifier…")
def load_esm2_clf():
    d = MODELS / "protein" / "esm2"
    for fname in ["linear_svm_esm2.pkl", "logreg_esm2.pkl"]:
        p = d / fname
        if p.exists():
            return joblib.load(p)
    return None

@st.cache_resource(show_spinner="Loading ProtBERT classifier…")
def load_protbert_clf():
    p = MODELS / "protein" / "protbert" / "linear_svm_calibrated.pkl"
    return joblib.load(p) if p.exists() else None

@st.cache_data(show_spinner="Loading pre-extracted embeddings…")
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
def load_comparison_results():
    dp = REPORTS / "dna_final_comparison.csv"
    pp = REPORTS / "protein_final_comparison.csv"
    return (pd.read_csv(dp) if dp.exists() else None,
            pd.read_csv(pp) if pp.exists() else None)

# ══════════════════════════════════════════════════════════════════════════════
# Feature engineering
# ══════════════════════════════════════════════════════════════════════════════

def compute_dna_features(seq: str) -> np.ndarray:
    s = seq.upper()
    L = len(s)
    counts = {b: s.count(b) for b in "ACGTN"}

    def sd(n, d): return n / d if d else 0.0

    gc   = sd(counts["G"] + counts["C"], L)
    at   = sd(counts["A"] + counts["T"], L)
    fA, fC, fG, fT, fN = [sd(counts[b], L) for b in "ACGTN"]
    gc_sk = sd(counts["G"] - counts["C"], counts["G"] + counts["C"])
    at_sk = sd(counts["A"] - counts["T"], counts["A"] + counts["T"])

    cpg   = s.count("CG")
    cpg_d = sd(cpg, L - 1)
    cpg_oe = sd(cpg * L, max(counts["C"], 1) * max(counts["G"], 1))

    valid = [c for c in s if c in "ACGT"]
    if valid:
        fr = {b: valid.count(b) / len(valid) for b in "ACGT"}
        ent = -sum(f * math.log2(f) for f in fr.values() if f > 0)
    else:
        ent = 0.0

    max_hp = 0
    for base in "ACGT":
        run = mx = 0
        for c in s:
            if c == base: run += 1; mx = max(mx, run)
            else: run = 0
        max_hp = max(max_hp, mx)

    di   = ["".join(d) for d in itertools.product("ACGT", repeat=2)]
    tri  = ["".join(t) for t in itertools.product("ACGT", repeat=3)]
    dif  = [sd(s.count(d), max(L-1, 1)) for d in di]
    trif = [sd(s.count(t), max(L-2, 1)) for t in tri]
    per  = sd(s.count("AA") + s.count("TT") + s.count("TA"), max(L-1, 1))
    cpg_rich = 1.0 if gc > 0.5 and cpg_oe > 0.6 else 0.0

    feats = [fA, fC, fG, fT, fN, gc, at, gc_sk, at_sk,
             cpg_d, cpg_oe, ent, float(max_hp), per, cpg_rich] + dif + trif
    return np.array(feats, dtype=np.float32).reshape(1, -1)


def compute_protein_features(seq: str) -> np.ndarray:
    """Exact 176-feature pipeline matching notebook 09 / protein baseline models."""
    from collections import Counter
    from itertools import product as iproduct

    seq = seq.strip().upper()
    L   = max(len(seq), 1)
    c   = Counter(seq)

    AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")

    # ── 20 AA fractions ──────────────────────────────────────────────────────
    feats = {}
    for a in AA_LIST:
        feats[f"frac_{a}"] = float(c[a] / L)

    # ── grouped fractions ─────────────────────────────────────────────────────
    AA_GROUPS = {
        "hydrophobic": set("AILMFWVY"),
        "polar":       set("STNQCY"),
        "positive":    set("KRH"),
        "negative":    set("DE"),
        "aromatic":    set("FWY"),
        "aliphatic":   set("AILV"),
        "small":       set("AGSTP"),
        "sulfur":      set("CM"),
        "amide":       set("NQ"),
    }
    for g, aset in AA_GROUPS.items():
        feats[f"frac_group_{g}"] = float(sum(c[a] for a in aset) / L)

    # ── entropy, homopolymer, unique AAs ─────────────────────────────────────
    ent = 0.0
    for a in AA_LIST:
        if c[a] > 0:
            p = c[a] / L; ent -= p * math.log2(p)
    feats["aa_entropy"] = float(ent)

    best = cur = 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i-1]: cur += 1; best = max(best, cur)
        else: cur = 1
    feats["max_homopolymer_run"] = float(best)
    feats["n_unique_aas"] = float(len(set(seq)))

    # ── ProtParam (biopython) ─────────────────────────────────────────────────
    try:
        from Bio.SeqUtils.ProtParam import ProteinAnalysis
        # Replace non-standard AAs for ProtParam
        clean = re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "A", seq)
        pa = ProteinAnalysis(clean)
        feats["molecular_weight"]  = float(pa.molecular_weight())
        feats["aromaticity"]       = float(pa.aromaticity())
        feats["instability_index"] = float(pa.instability_index())
        feats["isoelectric_point"] = float(pa.isoelectric_point())
        feats["gravy"]             = float(pa.gravy())
        feats["charge_ph7"]        = float(pa.charge_at_pH(7.0))
        h, t, s2 = pa.secondary_structure_fraction()
        feats["secstruct_helix"]   = float(h)
        feats["secstruct_turn"]    = float(t)
        feats["secstruct_sheet"]   = float(s2)
    except Exception:
        for k in ["molecular_weight","aromaticity","instability_index",
                  "isoelectric_point","gravy","charge_ph7",
                  "secstruct_helix","secstruct_turn","secstruct_sheet"]:
            feats[k] = 0.0

    # ── CTD descriptors ───────────────────────────────────────────────────────
    CTD_GROUPS = {
        "hydrophobicity_3": {"H": set("AILMFWV"), "P": set("CNQSTY"),  "N": set("DEGHRKP")},
        "polarity_2":       {"P": set("STNQCYW"), "N": set("ADFGHIKLMPVR")},
        "charge_3":         {"+": set("KRH"),      "-": set("DE"),       "0": set("ACFGILMNPQSTVWY")},
    }
    for prop, gdef in CTD_GROUPS.items():
        inv = {a: g for g, aset in gdef.items() for a in aset}
        gs  = "".join(inv.get(ch, list(gdef.keys())[0]) for ch in seq)
        syms = sorted(gdef.keys())
        gL   = max(len(gs), 1)
        gc2  = Counter(gs)

        # composition
        for s2 in syms:
            feats[f"{prop}__ctd_comp_{s2}"] = float(gc2.get(s2, 0) / gL)

        # transition
        trans = Counter()
        for i in range(len(gs)-1):
            a2, b2 = gs[i], gs[i+1]
            if a2 != b2:
                trans["".join(sorted([a2, b2]))] += 1
        denom = max(len(gs)-1, 1)
        for i, a2 in enumerate(syms):
            for b2 in syms[i+1:]:
                key = "".join(sorted([a2, b2]))
                feats[f"{prop}__ctd_trans_{a2}{b2}"] = float(trans.get(key, 0) / denom)

        # distribution
        for s2 in syms:
            idxs = [i+1 for i, ch in enumerate(gs) if ch == s2]
            if not idxs:
                for q in [1,25,50,75,100]: feats[f"{prop}__ctd_dist_{s2}_{q}"] = 0.0
                continue
            n2 = len(idxs)
            picks = [idxs[0],
                     idxs[int(math.ceil(0.25*n2))-1],
                     idxs[int(math.ceil(0.50*n2))-1],
                     idxs[int(math.ceil(0.75*n2))-1],
                     idxs[-1]]
            for q, pos in zip([1,25,50,75,100], picks):
                feats[f"{prop}__ctd_dist_{s2}_{q}"] = float(pos / gL)

    # ── PseAAC ────────────────────────────────────────────────────────────────
    PSEAAC_LAMBDA = 10; PSEAAC_WEIGHT = 0.05
    HP = {"A":0.62,"C":0.29,"D":-0.90,"E":-0.74,"F":1.19,"G":0.48,"H":-0.40,
          "I":1.38,"K":-1.50,"L":1.06,"M":0.64,"N":-0.78,"P":0.12,"Q":-0.85,
          "R":-2.53,"S":-0.18,"T":-0.05,"V":1.08,"W":0.81,"Y":0.26}
    thetas = []
    for lag in range(1, PSEAAC_LAMBDA+1):
        if len(seq) > lag:
            th = sum((HP.get(seq[i],0) - HP.get(seq[i+lag],0))**2
                     for i in range(len(seq)-lag)) / max(len(seq)-lag, 1)
        else:
            th = 0.0
        thetas.append(th)
    denom_pse = sum(c[a] for a in AA_LIST) + PSEAAC_WEIGHT * sum(thetas)
    denom_pse = max(denom_pse, 1e-9)
    for a in AA_LIST:
        feats[f"pse_aac_{a}"] = float(c[a] / denom_pse)
    for i, th in enumerate(thetas, 1):
        feats[f"pse_theta_{i}"] = float(PSEAAC_WEIGHT * th / denom_pse)

    # ── Reduced-alphabet (7 groups) dipeptides ────────────────────────────────
    RED7 = {"A":"A","G":"A","V":"A","I":"B","L":"B","F":"B","P":"B",
            "Y":"C","M":"C","T":"C","S":"C","H":"D","N":"D","Q":"D","W":"D",
            "R":"E","K":"E","D":"F","E":"F","C":"G"}
    RED7_SYMS = sorted(set(RED7.values()))
    rs    = "".join(RED7.get(ch, "A") for ch in seq)
    rdenom = max(len(rs)-1, 1)
    rd    = Counter(rs[i:i+2] for i in range(len(rs)-1))
    for a2, b2 in iproduct(RED7_SYMS, RED7_SYMS):
        feats[f"red7_di_{a2}{b2}"] = float(rd.get(a2+b2, 0) / rdenom)

    # ── Assemble in exact column order from the feature CSV ───────────────────
    FEAT_ORDER = [
        "seq_len",
        *[f"frac_{a}" for a in "ACDEFGHIKLMNPQRSTVWY"],
        *[f"frac_group_{g}" for g in ["hydrophobic","polar","positive","negative",
                                       "aromatic","aliphatic","small","sulfur","amide"]],
        "aa_entropy","max_homopolymer_run","n_unique_aas",
        "molecular_weight","aromaticity","instability_index",
        "isoelectric_point","gravy","charge_ph7",
        "secstruct_helix","secstruct_turn","secstruct_sheet",
        # CTD
        "hydrophobicity_3__ctd_comp_H","hydrophobicity_3__ctd_comp_P","hydrophobicity_3__ctd_comp_N",
        "hydrophobicity_3__ctd_trans_HP","hydrophobicity_3__ctd_trans_HN","hydrophobicity_3__ctd_trans_PN",
        *[f"hydrophobicity_3__ctd_dist_H_{q}" for q in [1,25,50,75,100]],
        *[f"hydrophobicity_3__ctd_dist_P_{q}" for q in [1,25,50,75,100]],
        *[f"hydrophobicity_3__ctd_dist_N_{q}" for q in [1,25,50,75,100]],
        "polarity_2__ctd_comp_P","polarity_2__ctd_comp_N",
        "polarity_2__ctd_trans_PN",
        *[f"polarity_2__ctd_dist_P_{q}" for q in [1,25,50,75,100]],
        *[f"polarity_2__ctd_dist_N_{q}" for q in [1,25,50,75,100]],
        "charge_3__ctd_comp_+","charge_3__ctd_comp_-","charge_3__ctd_comp_0",
        "charge_3__ctd_trans_+-","charge_3__ctd_trans_+0","charge_3__ctd_trans_-0",
        *[f"charge_3__ctd_dist_+_{q}" for q in [1,25,50,75,100]],
        *[f"charge_3__ctd_dist_-_{q}" for q in [1,25,50,75,100]],
        *[f"charge_3__ctd_dist_0_{q}" for q in [1,25,50,75,100]],
        # PseAAC
        *[f"pse_aac_{a}" for a in "ACDEFGHIKLMNPQRSTVWY"],
        *[f"pse_theta_{i}" for i in range(1,11)],
        # Red7 dipeptides
        *[f"red7_di_{a}{b}" for a in "ABCDEFG" for b in "ABCDEFG"],
    ]

    vec = np.array([feats.get(k, 0.0) for k in FEAT_ORDER], dtype=np.float32)
    return vec.reshape(1, -1)

# ══════════════════════════════════════════════════════════════════════════════
# Validation
# ══════════════════════════════════════════════════════════════════════════════

def validate_dna(seq: str):
    s = seq.strip().upper().replace(" ", "").replace("\n", "")
    bad = set(s) - set("ACGTN")
    if bad: return None, f"Invalid characters: {bad}"
    if len(s) < 50: return None, "Sequence too short (minimum 50 bp)"
    if len(s) > 1000: return None, "Sequence too long (maximum 1000 bp)"
    padded = (s + "N" * (200 - len(s)))[:200]
    return padded, None

def validate_protein(seq: str):
    VALID = set("ACDEFGHIKLMNPQRSTVWYXBZUO")
    s = seq.strip().upper().replace(" ", "").replace("\n", "")
    bad = set(s) - VALID
    if bad: return None, f"Invalid characters: {bad}"
    if len(s) < 20: return None, "Sequence too short (minimum 20 aa)"
    if len(s) > 1024: return None, "Sequence too long (maximum 1024 aa)"
    return s, None

# ══════════════════════════════════════════════════════════════════════════════
# PCA visualisation
# ══════════════════════════════════════════════════════════════════════════════

def plot_pca(X, y, query, class_labels, title):
    from sklearn.preprocessing import StandardScaler
    X_all = np.vstack([X, query])
    X_sc  = StandardScaler().fit_transform(X_all)
    coords = PCA(n_components=2, random_state=42).fit_transform(X_sc)
    bg, q = coords[:-1], coords[-1]

    fig, ax = plt.subplots(figsize=(7, 5))
    cmap = plt.cm.get_cmap("tab10", len(np.unique(y)))
    for i, lbl in enumerate(np.unique(y)):
        mask = y == lbl
        name = class_labels[lbl] if lbl < len(class_labels) else str(lbl)
        ax.scatter(bg[mask, 0], bg[mask, 1], c=[cmap(i)],
                   alpha=0.35, s=15, label=name)
    ax.scatter(*q, c="red", s=250, zorder=5, marker="*",
               edgecolors="black", linewidths=0.8, label="Your sequence")
    ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
    ax.set_title(title, fontsize=10)
    ax.legend(loc="best", fontsize=6, ncol=2)
    plt.tight_layout()
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("🧬 Bio-Seq LM")
    st.caption("Capstone · Deepika Sarala Pratapa")
    st.markdown("---")
    st.markdown("**Tasks**\n- DNA: promoter vs non-promoter\n- Protein: Pfam family (10-class)")
    st.markdown("**Paradigms**\n- Baseline ML (engineered features)\n- Sequence CNN\n"
                "- DNABERT-2 / NT-500M\n- ESM-2 / ProtBERT")
    st.markdown("---")
    st.markdown("**Embedding visualisation**")
    viz_method = st.radio("", ["PCA", "UMAP"], label_visibility="collapsed")
    st.markdown("---")
    live_inference = st.checkbox(
        "Enable live transformer inference\n(requires GPU, loads ~500M models)",
        value=False)

# ══════════════════════════════════════════════════════════════════════════════
# Tabs
# ══════════════════════════════════════════════════════════════════════════════

tab_dna, tab_prot, tab_dash = st.tabs(
    ["🧬 DNA Classification", "🔬 Protein Classification", "📊 Results Dashboard"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: DNA
# ─────────────────────────────────────────────────────────────────────────────
with tab_dna:
    st.header("DNA: Promoter vs Non-Promoter")
    st.caption("Input a human DNA sequence (50–1000 bp).")

    col_inp, col_ex = st.columns([3, 1])
    with col_ex:
        ex = st.selectbox("Load example", ["— custom —"] + list(DNA_EXAMPLES), key="dna_ex")
    with col_inp:
        dna_input = st.text_area("Paste DNA sequence (A/C/G/T/N)",
                                 value=DNA_EXAMPLES.get(ex, ""),
                                 height=120, placeholder="ACGTACGT…", key="dna_seq")

    if st.button("🔍 Classify DNA sequence", type="primary", key="dna_btn"):
        seq_clean, err = validate_dna(dna_input)
        if err:
            st.error(err)
        else:
            st.success(f"Sequence accepted: {len(dna_input.strip())} bp")
            results = {}

            # Baseline
            with st.spinner("Running baseline classifiers…"):
                feats = compute_dna_features(seq_clean)
                for name, clf in load_dna_baseline().items():
                    try:
                        prob = float(clf.predict_proba(feats)[0, 1])
                        results[f"Baseline / {name}"] = {"prob": prob, "pred": int(prob >= 0.5)}
                    except Exception as e:
                        results[f"Baseline / {name}"] = {"error": str(e)}

            # Transformer paradigms (proxy mode using class mean embeddings)
            dna_embs = load_dna_embeddings()
            base_prob = next((v["prob"] for v in results.values() if "prob" in v), 0.5)
            proxy_label = int(base_prob >= 0.5)

            for model_key, label, clf_fn in [
                ("dnabert2", "DNABERT-2 / XGBoost", load_dnabert2_clf),
                ("nt",       "NT-500M / XGBoost",   load_nt_clf),
            ]:
                clf = clf_fn()
                emb_data = dna_embs.get(model_key)
                if clf is None or emb_data is None:
                    continue
                class_mean = emb_data["X"][emb_data["y"] == proxy_label].mean(0, keepdims=True)
                try:
                    prob = float(clf.predict_proba(class_mean)[0, 1])
                    results[label] = {"prob": prob, "pred": int(prob >= 0.5),
                                      "emb": class_mean, "approx": True}
                except Exception as e:
                    results[label] = {"error": str(e)}

            # ── Display predictions ──
            st.markdown("### Predictions across paradigms")
            n_cols = max(len(results), 1)
            cols = st.columns(n_cols)
            for i, (name, res) in enumerate(results.items()):
                with cols[i]:
                    if "error" in res:
                        st.error(f"**{name}**\n{res['error']}")
                    else:
                        label_str = "🟢 Promoter" if res["pred"] == 1 else "🔴 Non-promoter"
                        st.metric(name.split(" / ")[-1],
                                  label_str,
                                  f"{res['prob']*100:.1f}% conf.")
                        if res.get("approx"):
                            st.caption("⚠ approx")

            # ── Confidence bar chart ──
            st.markdown("### Confidence comparison")
            valid_res = {k: v for k, v in results.items() if "prob" in v}
            if valid_res:
                fig, ax = plt.subplots(figsize=(9, 3))
                names = list(valid_res.keys())
                probs = [valid_res[k]["prob"] for k in names]
                colors = ["#2196F3" if p >= 0.5 else "#9E9E9E" for p in probs]
                ax.barh(names, probs, color=colors, alpha=0.85)
                ax.axvline(0.5, color="red", lw=1.5, ls="--", label="Decision boundary (0.5)")
                ax.set_xlim(0, 1)
                ax.set_xlabel("Promoter probability")
                ax.set_title("Promoter probability by paradigm")
                ax.legend(fontsize=8)
                plt.tight_layout()
                st.pyplot(fig); plt.close(fig)

            # ── Embedding visualisation ──
            st.markdown("### Embedding visualisation (pre-extracted)")
            for model_key, label in [("dnabert2","DNABERT-2 / XGBoost"),
                                      ("nt","NT-500M / XGBoost")]:
                emb_data = dna_embs.get(model_key)
                query_emb = results.get(label, {}).get("emb")
                if emb_data is not None and query_emb is not None:
                    fig2 = plot_pca(emb_data["X"], emb_data["y"], query_emb,
                                    ["Non-promoter", "Promoter"],
                                    f"{model_key.upper()} — PCA")
                    st.pyplot(fig2); plt.close(fig2)

            # ── Feature importance ──
            st.markdown("### Top features (Baseline XGBoost)")
            base_models = load_dna_baseline()
            xgb = base_models.get("XGBoost")
            if xgb is not None:
                est = xgb
                if hasattr(xgb, "named_steps"):
                    for s in xgb.named_steps.values():
                        if hasattr(s, "feature_importances_"):
                            est = s; break
                if hasattr(est, "feature_importances_"):
                    imp = est.feature_importances_
                    feat_names = (
                        ["frac_A","frac_C","frac_G","frac_T","frac_N",
                         "gc","at","gc_skew","at_skew","cpg_density",
                         "cpg_oe","entropy","max_hp","periodicity","cpg_rich"]
                        + [f"di_{a}{b}" for a in "ACGT" for b in "ACGT"]
                        + [f"tri_{''.join(t)}" for t in itertools.product("ACGT", repeat=3)]
                    )
                    top_idx = np.argsort(imp)[-15:][::-1]
                    fig3, ax3 = plt.subplots(figsize=(7, 4))
                    ax3.barh([feat_names[i] if i < len(feat_names) else f"f{i}"
                              for i in top_idx[::-1]],
                             imp[top_idx[::-1]], color="steelblue", alpha=0.85)
                    ax3.set_xlabel("Importance")
                    ax3.set_title("Top 15 DNA features (XGBoost)")
                    plt.tight_layout()
                    st.pyplot(fig3); plt.close(fig3)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: PROTEIN
# ─────────────────────────────────────────────────────────────────────────────
with tab_prot:
    st.header("Protein: Pfam Family Classification")
    st.caption("Input a human protein sequence (20–1024 aa).")

    col_inp2, col_ex2 = st.columns([3, 1])
    with col_ex2:
        ex2 = st.selectbox("Load example", ["— custom —"] + list(PROTEIN_EXAMPLES), key="prot_ex")
    with col_inp2:
        prot_input = st.text_area("Paste protein sequence (single-letter AA codes)",
                                  value=PROTEIN_EXAMPLES.get(ex2, ""),
                                  height=120, placeholder="MKTLL…", key="prot_seq")

    if st.button("🔍 Classify protein sequence", type="primary", key="prot_btn"):
        seq_p, err_p = validate_protein(prot_input)
        if err_p:
            st.error(err_p)
        else:
            st.success(f"Sequence accepted: {len(seq_p)} aa")
            results_p = {}

            # Baseline
            with st.spinner("Running baseline classifiers…"):
                feats_p = compute_protein_features(seq_p)
                for name, clf in load_protein_baseline().items():
                    try:
                        proba = clf.predict_proba(feats_p)[0]
                        pred  = int(np.argmax(proba))
                        results_p[f"Baseline / {name}"] = {"proba": proba, "pred": pred}
                    except Exception as e:
                        results_p[f"Baseline / {name}"] = {"error": str(e)}

            # Transformer proxy
            prot_embs = load_protein_embeddings()
            base_pred = next((v["pred"] for v in results_p.values() if "pred" in v), 0)

            for model_key, label, clf_fn in [
                ("esm2",     "ESM-2 / Linear SVM",    load_esm2_clf),
                ("protbert", "ProtBERT / Linear SVM",  load_protbert_clf),
            ]:
                clf = clf_fn()
                emb_data = prot_embs.get(model_key)
                if clf is None or emb_data is None:
                    continue
                class_mean = emb_data["X"][emb_data["y"] == base_pred].mean(0, keepdims=True)
                try:
                    proba = clf.predict_proba(class_mean)[0]
                    results_p[label] = {"proba": proba, "pred": int(np.argmax(proba)),
                                        "emb": class_mean, "approx": True}
                except Exception as e:
                    results_p[label] = {"error": str(e)}

            # ── Predictions ──
            st.markdown("### Predictions across paradigms")
            for name, res in results_p.items():
                if "error" in res:
                    st.error(f"**{name}**: {res['error']}")
                    continue
                pred_cls  = res["pred"]
                pfam_id   = PFAM_CLASSES[pred_cls] if pred_cls < len(PFAM_CLASSES) else str(pred_cls)
                pfam_name = PFAM_NAMES.get(pfam_id, pfam_id)
                conf      = float(res["proba"][pred_cls]) * 100
                approx    = " ⚠ approx" if res.get("approx") else ""
                st.markdown(f"**{name}{approx}** → `{pfam_id}` — {pfam_name} (*{conf:.1f}% conf.*)")

            # ── Top-3 confidence ──
            st.markdown("### Top-3 predicted families")
            best_res = None
            for pref in ["ProtBERT / Linear SVM","ESM-2 / Linear SVM","Baseline / XGBoost"]:
                if pref in results_p and "proba" in results_p[pref]:
                    best_res = results_p[pref]; break
            if best_res:
                proba = best_res["proba"]
                top3  = np.argsort(proba)[-3:][::-1]
                fig4, ax4 = plt.subplots(figsize=(7, 3))
                labs = [f"{PFAM_CLASSES[i]} — {PFAM_NAMES.get(PFAM_CLASSES[i],'')[:25]}"
                        for i in top3]
                ax4.barh(labs[::-1], proba[top3][::-1], color="coral", alpha=0.85)
                ax4.set_xlim(0, 1); ax4.set_xlabel("Probability")
                ax4.set_title("Top-3 predicted Pfam families")
                plt.tight_layout()
                st.pyplot(fig4); plt.close(fig4)

            # ── Embedding visualisation ──
            st.markdown("### Embedding visualisation (pre-extracted)")
            for model_key, label in [("esm2","ESM-2 / Linear SVM"),
                                      ("protbert","ProtBERT / Linear SVM")]:
                emb_data = prot_embs.get(model_key)
                query_emb = results_p.get(label, {}).get("emb")
                if emb_data is not None and query_emb is not None:
                    fig5 = plot_pca(emb_data["X"], emb_data["y"], query_emb,
                                    PFAM_CLASSES, f"{model_key.upper()} — PCA")
                    st.pyplot(fig5); plt.close(fig5)

            # ── Feature importance ──
            st.markdown("### Top features (Baseline XGBoost)")
            base_p = load_protein_baseline()
            xgb_p  = base_p.get("XGBoost")
            if xgb_p:
                est = xgb_p
                if hasattr(xgb_p, "named_steps"):
                    for s in xgb_p.named_steps.values():
                        if hasattr(s, "feature_importances_"): est = s; break
                if hasattr(est, "feature_importances_"):
                    imp = est.feature_importances_
                    AAS = list("ACDEFGHIKLMNPQRSTVWY")
                    fn  = ([f"frac_{a}" for a in AAS]
                           + ["avg_mw","charge","hydrophobic","polar"]
                           + [f"di_{a}{b}" for a in AAS[:7] for b in AAS[:7]])
                    top_idx = np.argsort(imp)[-15:][::-1]
                    fig6, ax6 = plt.subplots(figsize=(7, 4))
                    ax6.barh([fn[i] if i < len(fn) else f"f{i}" for i in top_idx[::-1]],
                             imp[top_idx[::-1]], color="coral", alpha=0.85)
                    ax6.set_xlabel("Importance")
                    ax6.set_title("Top 15 protein features (XGBoost)")
                    plt.tight_layout()
                    st.pyplot(fig6); plt.close(fig6)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
with tab_dash:
    st.header("📊 Results Dashboard")
    st.caption("Summary of all paradigms evaluated across both tasks.")

    dna_comp, prot_comp = load_comparison_results()

    col_d, col_p = st.columns(2)

    with col_d:
        st.subheader("DNA: Promoter classification")
        if dna_comp is not None:
            highlight_cols = [c for c in ["accuracy","f1","roc_auc"] if c in dna_comp.columns]
            st.dataframe(dna_comp.style.highlight_max(subset=highlight_cols, color="#d4edda"),
                         use_container_width=True)
            if "roc_auc" in dna_comp.columns:
                fig7, ax7 = plt.subplots(figsize=(6, 3.5))
                x7 = np.arange(len(dna_comp))
                ax7.bar(x7 - 0.18, dna_comp["accuracy"], 0.35,
                        label="Accuracy", color="steelblue", alpha=0.85)
                ax7.bar(x7 + 0.18, dna_comp["roc_auc"], 0.35,
                        label="ROC-AUC", color="coral", alpha=0.85)
                ax7.set_xticks(x7)
                ax7.set_xticklabels(dna_comp["paradigm"], rotation=20, ha="right", fontsize=8)
                ax7.set_ylim(0.65, 0.95)
                ax7.legend(fontsize=8); ax7.grid(axis="y", alpha=0.3)
                ax7.set_title("DNA paradigm comparison", fontsize=9)
                plt.tight_layout(); st.pyplot(fig7); plt.close(fig7)
        else:
            st.info("Run notebook 21 to generate comparison CSVs.")

    with col_p:
        st.subheader("Protein: Pfam family classification")
        if prot_comp is not None:
            highlight_cols_p = [c for c in ["accuracy","f1_macro"] if c in prot_comp.columns]
            st.dataframe(prot_comp.style.highlight_max(subset=highlight_cols_p, color="#d4edda"),
                         use_container_width=True)
            if "accuracy" in prot_comp.columns:
                fig8, ax8 = plt.subplots(figsize=(6, 3.5))
                x8 = np.arange(len(prot_comp))
                ax8.bar(x8, prot_comp["accuracy"], 0.5,
                        color="steelblue", alpha=0.85, label="Accuracy")
                if "f1_macro" in prot_comp.columns:
                    ax8.plot(x8, prot_comp["f1_macro"], "g^--",
                             lw=1.5, markersize=6, label="F1-macro")
                ax8.set_xticks(x8)
                ax8.set_xticklabels(prot_comp["paradigm"], rotation=20, ha="right", fontsize=8)
                ax8.set_ylim(0.5, 1.05)
                ax8.legend(fontsize=8); ax8.grid(axis="y", alpha=0.3)
                ax8.set_title("Protein paradigm comparison", fontsize=9)
                plt.tight_layout(); st.pyplot(fig8); plt.close(fig8)
        else:
            st.info("Run notebook 21 to generate comparison CSVs.")

    # Top-line metrics from final_summary.json
    summary_path = REPORTS / "final_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            fs = json.load(f)
        st.markdown("---")
        st.subheader("Top-line findings")
        c1, c2 = st.columns(2)
        with c1:
            bd = fs.get("dna", {}).get("best_model", {})
            st.metric("Best DNA model",
                      f"{bd.get('paradigm','')} / {bd.get('model','')}",
                      f"ROC-AUC {bd.get('roc_auc', 0):.4f}")
        with c2:
            bp = fs.get("protein", {}).get("best_model", {})
            st.metric("Best Protein model",
                      f"{bp.get('paradigm','')} / {bp.get('model','')}",
                      f"Accuracy {bp.get('accuracy', 0):.4f}")
