"""Group cortical-depth tSNR, fixed matched GLM Top200, subject-colour curves.
python -m pip install numpy pandas h5py matplotlib
python group_laminar_tsnr_TOP200_subject_colours.py
Read-only inputs. New timestamped outputs. No GLM refit or contrast plotting.
"""
from pathlib import Path
from datetime import datetime
import json
import re
import traceback
import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ======================== EDIT PATHS HERE =========================
SUBJECT_ROOTS = {
    "7485": Path(r"D:\NORDIC_Subject1_project\Laminarprofile"),
    "7468": Path(r"E:\layerfmri\LAYER_s7468_warpednordic\codes\Laminarprofile\Laminar"),
    "7469": Path(r"E:\layerfmri\LAYER_s7469_warpenordic\codes\laminar profile"),
    "7482": Path(r"E:\layerfmri\LAYER_s7482_Warpenordic\codes\Laminar profile"),
    "7484": Path(r"E:\layerfmri\LAYER_s7484_warpenordic\codes\Laminarprofile"),
}
GLM_DIRS = {
    s: root / "2. GLM 75%" / "GLM" for s, root in SUBJECT_ROOTS.items()
}
GLM_DIRS["7485"] = SUBJECT_ROOTS["7485"] / "2. GLM 75%" / "GLM" / "GLM_results_fixed_SPM_weights"
OUTPUT_ROOT = Path(r"C:\Users\L E N O V O\OneDrive\Desktop\Group  Analysis\laminar profile\1\TOP200_subject_colours")
# Optional exact overrides: SAMPLING_OVERRIDES[("7485", "Nordic")] = Path(r"D:\...\file.h5")
SAMPLING_OVERRIDES = {}
# =================================================================
ANALYSIS_MODE = "TOP200"  # "TOP200" or "ALL_ROI"
SELECTION_REFERENCE = "Without Nordic"  # "Without Nordic" or "Nordic"; fixed across BOTH conditions
TOP_N = 200
TIME_BLOCK = 32
MAX_INVALID_FRACTION = 0.05
DPI = 300
SIDES = ("left", "right")
CONDITIONS = ("Without Nordic", "Nordic")
TOKENS = {"Without Nordic": "without_NORDIC", "Nordic": "with_NORDIC"}
STYLES = {"Without Nordic": "--", "Nordic": "-"}
SUBJECT_COLOURS = {
    "7485": "#4477AA", "7468": "#EE6677", "7469": "#228833",
    "7482": "#AA3377", "7484": "#CCBB44",
}


def as_text(x):
    return x.decode("utf-8", errors="replace") if isinstance(x, bytes) else str(x)


def sampling_path(s, c):
    if (s, c) in SAMPLING_OVERRIDES:
        return Path(SAMPLING_OVERRIDES[(s, c)])
    subdir = "With" if c == "Nordic" else "Without"
    return SUBJECT_ROOTS[s] / "1. Sampling" / subdir / f"pIns_laminar_sampling_{s}_{TOKENS[c]}.h5"


def selection_path(s):
    return GLM_DIRS[s] / f"pIns_GLM_{TOKENS[SELECTION_REFERENCE]}_deep75_top200_layerwise.h5"


def get_depths(h, ndepth, warnings):
    if "depth_fractions" in h:
        depths = np.asarray(h["depth_fractions"][:], dtype=float)
    else:
        if ndepth != 20:
            raise ValueError("Missing depth vector and depth count is not 20")
        depths = np.linspace(0, 1, 20)
        warnings.append("Depth fractions inferred from supplied 20-depth sampling scripts; dataset absent")
    if depths.shape != (ndepth,) or not np.isfinite(depths).all() or np.any(np.diff(depths) <= 0):
        raise ValueError("Invalid depth grid")
    if not np.isclose(depths[0], 0) or not np.isclose(depths[-1], 1):
        raise ValueError("Expected full depth range 0 white to 1 pial")
    return depths


def inspect_inputs(s, handles):
    warnings, info = [], {"subject": s, "conditions": {}, "hemispheres": {}}
    ref_depths, ref_runs, ref_indices = None, None, None
    for c, h in handles.items():
        if as_text(h.attrs.get("subject", "")) != s or as_text(h.attrs.get("condition", "")) != TOKENS[c]:
            raise ValueError(f"{s}/{c}: subject or condition metadata mismatch")
        if "complete" in h.attrs and not bool(h.attrs["complete"]):
            raise ValueError(f"{s}/{c}: complete=False")
        if "complete" not in h.attrs:
            warnings.append(f"{c}: complete marker absent, direct data checks required")
        direction = as_text(h.attrs.get("sampling_direction", "")).lower()
        if "white" not in direction or "pial" not in direction:
            raise ValueError("Depth direction metadata missing")
        raw_runs = np.asarray(h["run_lengths"][:])
        if raw_runs.ndim != 1 or len(raw_runs) == 0 or not np.all(np.isfinite(raw_runs)) or not np.all(raw_runs == raw_runs.astype(int)):
            raise ValueError("Invalid run lengths")
        runs = raw_runs.astype(int)
        if np.any(runs < 2):
            raise ValueError("Runs must contain at least two time points")
        paths = [as_text(x) for x in h["volume_files"][:]]
        matches = [re.search(r"_(\d+)_Warped-to-Anat\.nii(?:\.gz)?$", x, re.I) for x in paths]
        if any(m is None for m in matches):
            raise ValueError("Unrecognized volume filename; inspect temporal ordering")
        indices = np.array([int(m.group(1)) for m in matches])
        if len(indices) != runs.sum():
            raise ValueError("Volume list / run length mismatch")
        edges = np.r_[0, np.cumsum(runs)]
        for a, b in zip(edges[:-1], edges[1:]):
            if np.any(np.diff(indices[a:b]) != 1):
                raise ValueError("Nonsequential scan indices within run")
        if "written_volumes" in h and not np.all(h["written_volumes"][:]):
            raise ValueError("Unwritten volumes")
        if "n_written_volumes" in h.attrs and int(h.attrs["n_written_volumes"]) != runs.sum():
            raise ValueError("Written volume count mismatch")
        for side in SIDES:
            ds = h[f"{side}/layeractivation"]
            if ds.ndim != 3 or ds.shape[0] < 1 or ds.shape[2] != runs.sum():
                raise ValueError(f"{s}/{c}/{side}: dataset shape mismatch")
            depths = get_depths(h, ds.shape[1], warnings)
            if ref_depths is None:
                ref_depths, ref_runs, ref_indices = depths, runs, indices
            if depths.shape != ref_depths.shape or not np.allclose(depths, ref_depths, atol=1e-10, rtol=0) or not np.array_equal(runs, ref_runs) or not np.array_equal(indices, ref_indices):
                raise ValueError("Paired depth grid / runs / numeric time ordering mismatch")
        info["conditions"][c] = {"file": str(sampling_path(s, c)), "first_scan": paths[0], "last_scan": paths[-1],
            "t1_reference": as_text(h.attrs.get("t1_reference", "unknown")), "sampling_direction": direction}
    a, b = (handles[c] for c in CONDITIONS)
    for side in SIDES:
        ds_a, ds_b = a[f"{side}/layeractivation"], b[f"{side}/layeractivation"]
        if ds_a.shape != ds_b.shape:
            raise ValueError(f"{side}: paired shapes differ")
        va, vb = np.asarray(a[f"{side}/surface_vertex_indices"][:]), np.asarray(b[f"{side}/surface_vertex_indices"][:])
        ncol = ds_a.shape[0]
        if va.shape != (ncol,) or vb.shape != (ncol,) or not np.array_equal(va, vb) or len(np.unique(va)) != ncol:
            raise ValueError(f"{side}: vertex identity/order mismatch")
        key = f"{side}/sampling_coordinates_vox"
        status = "unavailable in one or both inputs"
        if key in a and key in b:
            ca, cb = a[key][:], b[key][:]
            if ca.shape != (ncol, len(ref_depths), 3) or cb.shape != ca.shape or not np.isfinite(ca).all() or not np.isfinite(cb).all() or not np.allclose(ca, cb, rtol=0, atol=1e-5):
                raise ValueError(f"{side}: sampling coordinate mismatch")
            status = "matched"
        else:
            warnings.append(f"{side}: coordinate comparison unavailable; paired vertex identity checked")
        info["hemispheres"][side] = {"all_roi_columns": ncol, "coordinate_check": status}
    warnings.append("Numeric volume order checked; acquisition/run provenance is not independently certified")
    info.update(warnings=warnings, run_lengths=ref_runs.tolist(), depths=ref_depths.tolist())
    return ref_depths, ref_runs, info


def validate_selection(raw, stored_vertices, glm_vertices, sampling_vertices, tpain, tcognition, tconjunction):
    raw = np.asarray(raw)
    if raw.ndim != 1 or not np.all(np.isfinite(raw)) or not np.all(raw == raw.astype(int)):
        raise ValueError("Selected column indices must be a finite integer vector")
    idx = raw.astype(np.int64)
    ncol = len(sampling_vertices)
    if len(idx) != TOP_N or len(np.unique(idx)) != TOP_N or np.any(idx < 0) or np.any(idx >= ncol):
        raise ValueError(f"Expected exactly {TOP_N} unique valid stored column indices; found {len(idx)}")
    if not np.array_equal(glm_vertices, sampling_vertices):
        raise ValueError("Selection GLM vertex IDs differ from current sampling file")
    if not np.array_equal(stored_vertices, sampling_vertices[idx]):
        raise ValueError("Stored selected vertex IDs do not match stored column indices")
    for x in (tpain, tcognition, tconjunction):
        if x.shape != (ncol,) or not np.isfinite(x).all():
            raise ValueError("Invalid column-statistic vector in selection GLM")
    if not np.allclose(tconjunction, np.minimum(tpain, tcognition), atol=1e-7, rtol=1e-7):
        raise ValueError("Stored conjunction is not min(Tpain,Tcognition)")
    positive = (tpain > 0) & (tcognition > 0)
    if not positive[idx].all():
        raise ValueError("Some stored selected columns do not have positive conjunction")
    unselected = positive.copy()
    unselected[idx] = False
    if unselected.any() and tconjunction[unselected].max() > tconjunction[idx].min() + 1e-7:
        raise ValueError("Stored selection is not Top200 positive conjunction")
    return np.sort(idx)


def load_selection(s, side, sample, runs, depths):
    vertices = np.asarray(sample[f"{side}/surface_vertex_indices"][:])
    if ANALYSIS_MODE == "ALL_ROI":
        return np.arange(len(vertices)), {"mode": "ALL_ROI", "columns": len(vertices)}, []
    path = selection_path(s)
    with h5py.File(path, "r") as g:
        if "complete" in g.attrs and not bool(g.attrs["complete"]):
            raise ValueError(f"Selection GLM incomplete: {path}")
        expected = TOKENS[SELECTION_REFERENCE]
        if as_text(g.attrs.get("condition", "")) != expected:
            raise ValueError("Selection reference condition mismatch")
        if "subject" in g.attrs and as_text(g.attrs["subject"]) != s:
            raise ValueError("Selection GLM subject metadata mismatch")
        if not np.array_equal(g["run_lengths"][:], runs):
            raise ValueError("Selection GLM / sampling run lengths mismatch")
        if not np.allclose(g[f"{side}/layer_depth"][:], depths, atol=1e-10, rtol=0):
            raise ValueError("Selection GLM / sampling depth grid mismatch")
        raw = g[f"{side}/selected_column_indices_zero_based"][:]
        selected_vertices = g[f"{side}/selected_surface_vertex_indices"][:]
        gv = g[f"{side}/surface_vertex_indices"][:]
        tp = g[f"{side}/column_t_pain"][:]
        tc = g[f"{side}/column_t_cognition"][:]
        tj = g[f"{side}/column_t_conjunction"][:]
        idx = validate_selection(raw, selected_vertices, gv, vertices, tp, tc, tj)
        info = {"mode": "TOP200", "reference": SELECTION_REFERENCE, "file": str(path),
            "columns": len(idx), "criterion": "Stored Top200 positive min(Tpain,Tcognition), deep75 GLM",
            "spm_source": as_text(g.attrs.get("spm_source", "unknown")),
            "estimation": as_text(g.attrs.get("estimation", "unknown")),
            "selection_is_independent": False}
        rank_map = {int(column): rank for rank, column in enumerate(np.asarray(raw), 1)}
        audit = [{"subject": s, "hemisphere": side, "reference": SELECTION_REFERENCE,
            "stored_order": rank_map[int(i)], "column_index_zero_based": int(i),
            "surface_vertex_id": int(vertices[i]), "t_pain": float(tp[i]),
            "t_cognition": float(tc[i]), "t_conjunction": float(tj[i]), "selection_file": str(path)} for i in idx]
    return idx, info, audit


def stream_moments(dataset, start, stop, indices):
    mean = np.zeros((len(indices), dataset.shape[1]), dtype=np.float64)
    m2 = np.zeros_like(mean)
    valid = np.ones(mean.shape, dtype=bool)
    n = 0
    for begin in range(start, stop, TIME_BLOCK):
        end = min(stop, begin + TIME_BLOCK)
        data = np.asarray(dataset[indices, :, begin:end], dtype=np.float64)
        finite = np.isfinite(data)
        if np.any(np.all(finite & (data == 0), axis=(0, 1))) or np.any(np.all(~finite, axis=(0, 1))):
            raise ValueError(f"All selected samples are zero/non-finite at a scan in [{begin},{end})")
        valid &= finite.all(axis=2)
        data[~finite] = 0
        k = data.shape[2]
        bm = data.mean(axis=2)
        centered = data - bm[:, :, None]
        b2 = np.einsum("ijk,ijk->ij", centered, centered)
        delta = bm - mean
        m2 += b2 + delta*delta*(n*k/(n+k))
        mean += delta*(k/(n+k))
        n += k
    if n < 2:
        raise ValueError("Insufficient time points")
    sd = np.sqrt(np.maximum(m2, 0)/(n-1))
    valid &= np.isfinite(mean) & np.isfinite(sd) & (mean > 0) & (sd > 0)
    return mean, sd, valid


def summarize(run_df):
    keys = ["subject", "hemisphere", "condition", "depth_index", "depth_fraction"]
    subject_df = run_df.groupby(keys, as_index=False).agg(tsnr=("tsnr", "mean"),
        mean_signal=("mean_signal", "mean"), temporal_sd=("temporal_sd", "mean"), runs=("run", "nunique"))
    group_df = subject_df.groupby(keys[1:], as_index=False).agg(mean=("tsnr", "mean"), sd=("tsnr", "std"), n=("subject", "nunique"))
    group_df["sem"] = group_df.sd / np.sqrt(group_df.n)
    diff = subject_df.pivot(index=["subject", "hemisphere", "depth_index", "depth_fraction"], columns="condition", values="tsnr").reset_index()
    if diff[list(CONDITIONS)].isna().any().any():
        raise ValueError("Unpaired subject profiles")
    diff["difference"] = diff["Nordic"] - diff["Without Nordic"]
    return subject_df, group_df, diff


def selection_label():
    return f"Top{TOP_N} {SELECTION_REFERENCE}-selected columns, matched in both conditions" if ANALYSIS_MODE == "TOP200" else "All anatomical ROI columns, paired between conditions"


def legends(group=True):
    handles = [Line2D([0], [0], color=SUBJECT_COLOURS[s], lw=2, label=s) for s in SUBJECT_ROOTS]
    handles += [Line2D([0], [0], color="gray", ls=STYLES[c], lw=2, label=c) for c in CONDITIONS]
    if group:
        handles.append(Line2D([0], [0], color="black", lw=3, label="Group mean"))
    return handles


def style_depth(ax, ylabel="tSNR"):
    ax.set(xlim=(0, 1), xlabel="Cortical depth (white matter to pial)", ylabel=ylabel)
    ax.set_xticks([0, .25, .5, .75, 1])
    ax.grid(alpha=.17)
    ax.spines[["top", "right"]].set_visible(False)


def save(fig, folder, stem):
    for extension in ("png", "pdf", "svg"):
        fig.savefig(folder / f"{stem}.{extension}", dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def draw_figures(sub, group, diff, folder):
    ids = list(SUBJECT_ROOTS)
    title = selection_label()
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True, constrained_layout=True)
    for ax, side in zip(axes, SIDES):
        for c in CONDITIONS:
            d = group.query("hemisphere == @side and condition == @c").sort_values("depth_fraction")
            x, y, sd = d.depth_fraction.to_numpy(), d["mean"].to_numpy(), d.sd.to_numpy()
            ax.fill_between(x, y-sd, y+sd, color="gray", alpha=.08, zorder=1)
            for s in ids:
                q = sub.query("hemisphere == @side and condition == @c and subject == @s").sort_values("depth_fraction")
                ax.plot(q.depth_fraction, q.tsnr, color=SUBJECT_COLOURS[s], ls=STYLES[c], lw=1.8, alpha=.9, zorder=2)
            ax.plot(x, y, color="black", ls=STYLES[c], lw=3, zorder=3)
        style_depth(ax)
        ax.set_title(side.title()+" posterior insula")
    axes[1].legend(handles=legends(), loc="upper left", bbox_to_anchor=(1.02,1), frameon=False, fontsize=9)
    fig.suptitle("Group tSNR-depth | n=5 | subject colours\n"+title+"\nGray bands: mean +/- between-subject SD; solid Nordic / dashed Without Nordic", fontsize=12)
    save(fig, folder, "Figure_01_Group_tSNR_subject_colours")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True, constrained_layout=True)
    for row, c in enumerate(CONDITIONS):
        for col, side in enumerate(SIDES):
            ax = axes[row, col]
            for s in ids:
                q = sub.query("hemisphere == @side and condition == @c and subject == @s").sort_values("depth_fraction")
                ax.plot(q.depth_fraction, q.tsnr, color=SUBJECT_COLOURS[s], ls=STYLES[c], lw=2, label=s)
            d = group.query("hemisphere == @side and condition == @c").sort_values("depth_fraction")
            ax.plot(d.depth_fraction, d["mean"], color="black", ls=STYLES[c], lw=3, label="Group mean")
            style_depth(ax)
            ax.set_title(side.title()+" | "+c)
    axes[0,1].legend(frameon=False, bbox_to_anchor=(1.02,1), loc="upper left")
    fig.suptitle("Conditions separated for readability\n"+title, fontsize=13)
    save(fig, folder, "Figure_01B_Conditions_separated")

    fig, axes = plt.subplots(1, 2, figsize=(13,5), sharey=True, constrained_layout=True)
    for ax, side in zip(axes, SIDES):
        for s in ids:
            d = diff.query("hemisphere == @side and subject == @s").sort_values("depth_fraction")
            ax.plot(d.depth_fraction, d.difference, lw=2, color=SUBJECT_COLOURS[s], label=s)
        d = diff.query("hemisphere == @side").groupby("depth_fraction").difference.agg(["mean", "std"])
        x, y, sd = d.index.to_numpy(), d["mean"].to_numpy(), d["std"].to_numpy()
        ax.fill_between(x,y-sd,y+sd,color="gray",alpha=.16)
        ax.plot(x,y,color="black",lw=3,label="Group mean")
        ax.axhline(0,color="gray",ls="--",lw=1)
        style_depth(ax,"tSNR difference (Nordic - Without Nordic)")
        ax.set_title(side.title()+" posterior insula")
    axes[1].legend(frameon=False,loc="upper left",bbox_to_anchor=(1.02,1))
    fig.suptitle("Paired depth differences | mean +/- between-subject SD\n"+title,fontsize=12)
    save(fig,folder,"Figure_02_Paired_depth_differences")

    fig, axes = plt.subplots(5,2,figsize=(11,14),sharex=True,constrained_layout=True)
    for row,s in enumerate(ids):
        for col,side in enumerate(SIDES):
            ax=axes[row,col]
            for c in CONDITIONS:
                d=sub.query("subject == @s and hemisphere == @side and condition == @c").sort_values("depth_fraction")
                ax.plot(d.depth_fraction,d.tsnr,color=SUBJECT_COLOURS[s],ls=STYLES[c],lw=2.2,label=c)
            style_depth(ax)
            ax.set_title(f"Subject {s} | {side.title()}")
            if row==0:
                ax.legend(frameon=False,fontsize=8)
    fig.suptitle("Individual tSNR-depth profiles\n"+title,fontsize=12)
    save(fig,folder,"Figure_03_Individual_subject_profiles")

    averaged=sub.groupby(["subject","hemisphere","condition"],as_index=False).tsnr.mean()
    averaged.to_csv(folder.parent/"subject_depth_averaged_tSNR.csv",index=False)
    fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True,constrained_layout=True)
    offsets=np.linspace(-.10,.10,len(ids))
    for ax,side in zip(axes,SIDES):
        for i,s in enumerate(ids):
            d=averaged.query("subject == @s and hemisphere == @side").set_index("condition").reindex(CONDITIONS)
            ax.plot(np.array([0,1])+offsets[i],d.tsnr,"o-",color=SUBJECT_COLOURS[s],lw=1.7,ms=7,label=s)
        d=averaged.query("hemisphere == @side").groupby("condition").tsnr.agg(["mean","std"]).reindex(CONDITIONS)
        ax.errorbar([0,1],d["mean"],yerr=d["std"],fmt="D",color="black",capsize=5,ms=7,label="Mean +/- SD",zorder=4)
        ax.set_xticks([0,1],CONDITIONS)
        ax.set(xlim=(-.3,1.3),ylabel="Depth-averaged tSNR",title=side.title()+" posterior insula")
        ax.grid(axis="y",alpha=.17)
        ax.spines[["top","right"]].set_visible(False)
    axes[1].legend(frameon=False,loc="upper left",bbox_to_anchor=(1.02,1))
    fig.suptitle("Paired subject summary | equal average across sampled depths\n"+title+"\nHorizontal offsets separate points; tSNR values are unchanged",fontsize=11)
    save(fig,folder,"Figure_04_Paired_subject_summary")

    fig,axes=plt.subplots(1,2,figsize=(12,4.8),constrained_layout=True)
    limit=max(float(np.abs(diff.difference).max()),1e-9)
    for ax,side in zip(axes,SIDES):
        d=diff.query("hemisphere == @side").pivot(index="subject",columns="depth_fraction",values="difference").reindex(ids)
        im=ax.imshow(d.to_numpy(),aspect="auto",interpolation="nearest",cmap="RdBu_r",vmin=-limit,vmax=limit)
        ix=np.unique(np.linspace(0,len(d.columns)-1,5).round().astype(int))
        ax.set_xticks(ix,[f"{d.columns[i]:.2f}" for i in ix])
        ax.set_yticks(range(len(ids)),ids)
        for tick,s in zip(ax.get_yticklabels(),ids):
            tick.set_color(SUBJECT_COLOURS[s])
        ax.set(xlabel="Cortical depth",ylabel="Subject",title=side.title()+" posterior insula")
    fig.colorbar(im,ax=list(axes),label="tSNR difference (Nordic - Without Nordic)",shrink=.85)
    fig.suptitle("Individual depth-difference heatmap\n"+title,fontsize=12)
    save(fig,folder,"Figure_05_Difference_heatmap")

    fig,axes=plt.subplots(2,2,figsize=(13,9),sharex=True,constrained_layout=True)
    for row,metric in enumerate(["mean_signal","temporal_sd"]):
        for col,side in enumerate(SIDES):
            ax=axes[row,col]
            for c in CONDITIONS:
                for s in ids:
                    d=sub.query("subject == @s and hemisphere == @side and condition == @c").sort_values("depth_fraction")
                    ax.plot(d.depth_fraction,d[metric],color=SUBJECT_COLOURS[s],ls=STYLES[c],lw=1.5,alpha=.85)
            style_depth(ax,"Mean signal (input units)" if row==0 else "Temporal SD (input units)")
            ax.set_title(side.title()+" | "+metric.replace("_"," "))
    axes[0,1].legend(handles=legends(False),loc="upper left",bbox_to_anchor=(1.02,1),frameon=False,fontsize=8)
    fig.suptitle("Signal and temporal-variability QC | subject colours\n"+title+"\nInput units may differ between subjects; these are not activation estimates",fontsize=11)
    save(fig,folder,"Figure_06_Signal_and_SD_QC")


def main():
    if ANALYSIS_MODE not in ("TOP200","ALL_ROI") or SELECTION_REFERENCE not in CONDITIONS:
        raise ValueError("Invalid analysis mode or selection reference")
    if len(SUBJECT_ROOTS)!=5 or any(s not in SUBJECT_COLOURS for s in SUBJECT_ROOTS):
        raise ValueError("Expected the five numeric subject IDs with defined colours, not Subject1..5")
    for s in SUBJECT_ROOTS:
        for c in CONDITIONS:
            p=sampling_path(s,c)
            if not p.is_file():
                raise FileNotFoundError(f"Missing {s}/{c}: {p}")
        if ANALYSIS_MODE=="TOP200" and not selection_path(s).is_file():
            raise FileNotFoundError(f"Missing selection GLM for {s}: {selection_path(s)}")
    folder=OUTPUT_ROOT/datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    folder.mkdir(parents=True,exist_ok=False)
    figs=folder/"figures"; figs.mkdir()
    rows,qc,selection_rows,provenance=[],[],[],[]
    common_depths=None
    try:
        for s in SUBJECT_ROOTS:
            print(f"\nSUBJECT {s}",flush=True)
            with h5py.File(sampling_path(s,CONDITIONS[0]),"r") as a,h5py.File(sampling_path(s,CONDITIONS[1]),"r") as b:
                handles=dict(zip(CONDITIONS,[a,b]))
                depths,runs,info=inspect_inputs(s,handles)
                provenance.append(info)
                if common_depths is None:
                    common_depths=depths
                if depths.shape!=common_depths.shape or not np.allclose(depths,common_depths,atol=1e-10,rtol=0):
                    raise ValueError("Depth grids differ between subjects")
                edges=np.r_[0,np.cumsum(runs)]
                for side in SIDES:
                    idx,sel_info,audit=load_selection(s,side,handles[SELECTION_REFERENCE],runs,depths)
                    selection_rows.extend(audit)
                    info["hemispheres"][side]["selection"]=sel_info
                    print(f"  {side}: {len(idx)} selected columns; same set for both conditions",flush=True)
                    for run,(start,stop) in enumerate(zip(edges[:-1],edges[1:]),1):
                        stats={c:stream_moments(handles[c][f"{side}/layeractivation"],int(start),int(stop),idx) for c in CONDITIONS}
                        valid=stats[CONDITIONS[0]][2]&stats[CONDITIONS[1]][2]
                        counts=valid.sum(axis=0)
                        invalid_fraction=1-counts/len(idx)
                        if np.any(counts==0) or np.any(invalid_fraction>MAX_INVALID_FRACTION):
                            raise ValueError(f"{s}/{side}/run{run}: too many invalid selected column-depth series (max {invalid_fraction.max():.3f})")
                        if np.any(counts<len(idx)):
                            info["warnings"].append(f"{side}/run{run}: effective paired valid columns {counts.min()}..{counts.max()} of {len(idx)}; see QC CSV")
                        for j,depth in enumerate(depths):
                            keep=valid[:,j]
                            qc.append({"subject":s,"hemisphere":side,"run":run,"depth_index":j,"depth_fraction":float(depth),
                                "selected_columns":len(idx),"paired_valid_columns":int(counts[j]),"excluded_fraction":float(invalid_fraction[j])})
                            for c in CONDITIONS:
                                mean,sd,_=stats[c]
                                rows.append({"subject":s,"hemisphere":side,"condition":c,"run":run,"depth_index":j,
                                    "depth_fraction":float(depth),"tsnr":float(np.mean(mean[keep,j]/sd[keep,j])),
                                    "mean_signal":float(np.mean(mean[keep,j])),"temporal_sd":float(np.mean(sd[keep,j])),
                                    "selected_columns":len(idx),"paired_valid_columns":int(counts[j])})
                        print(f"    run {run}: calculated",flush=True)
                        del stats
        run_df=pd.DataFrame(rows)
        sub,group,diff=summarize(run_df)
        if not (group.n==5).all():
            raise ValueError("Group profiles must contain all five subjects")
        diff_group=diff.groupby(["hemisphere","depth_index","depth_fraction"],as_index=False).agg(
            mean_difference=("difference","mean"),sd_difference=("difference","std"),n=("subject","nunique"))
        tables={"tSNR_by_run":run_df,"tSNR_by_subject":sub,"tSNR_group_summary":group,
            "paired_depth_differences":diff,"paired_difference_group_summary":diff_group,"paired_validity_QC":pd.DataFrame(qc)}
        if selection_rows:
            tables["Top200_selection_audit"]=pd.DataFrame(selection_rows)
        for name,df in tables.items():
            df.to_csv(folder/f"{name}.csv",index=False)
        metadata={"analysis_mode":ANALYSIS_MODE,"selection_reference":SELECTION_REFERENCE,"description":selection_label(),
            "selection_is_independent":False if ANALYSIS_MODE=="TOP200" else None,
            "tSNR_definition":"Columnwise temporal mean / temporal sample SD (ddof=1), then column mean, then equal run mean, then equal subject mean",
            "prior_smoothing":"Not verified; no smoothing added, do not claim smoothed 4 mm",
            "new_first_level_GLM":False,"contrast_estimates_plotted":False,"inferential_tests":False,
            "detrending_or_filtering_added":False,"group_band":"Between-subject sample SD, not confidence interval",
            "subject_colours":SUBJECT_COLOURS,"condition_styles":STYLES,"subjects":provenance}
        (folder/"provenance_and_QC.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
        methods=("DESCRIPTIVE MATCHED-COLUMN tSNR-DEPTH ANALYSIS\n\n"+selection_label()+"\n"
            "The stored GLM selection is used, not a new ranking of tSNR or a per-depth Top200 selection.\n"
            "TOP200 mode validates positive min(Tpain,Tcognition) ranking and vertex mapping.\n"
            "The selected set is frozen across conditions, depths and runs within subject/hemisphere.\n"
            "Selection from the same dataset is not independent; matched sites do not eliminate selection bias.\n"
            "Existing All-ROI results remain a useful separate non-activation-selected comparison.\n"
            "Time-course mean divided by temporal sample SD, ddof=1. No added detrending/filtering/smoothing.\n"
            "Invalid column-depth series excluded symmetrically within run; actual counts are in QC CSV.\n"
            "Equal columns within run/depth, equal runs within subject, equal subjects at group level.\n"
            "Plots are descriptive, with between-subject SD and no p-values or significance claims.\n"
            "Twenty sampled cortical depths are not twenty independently resolved histological layers.\n"
            "Task, drift and physiological variability remain in temporal SD.\n"
            "Input smoothing and anatomical registration are not independently certified here.\n"
            "No new GLM, SPM estimation, beta/contrast estimation or laminar activation analysis is performed.\n")
        (folder/"METHODS.txt").write_text(methods,encoding="utf-8")
        print("\nDrawing seven figures...",flush=True)
        draw_figures(sub,group,diff,figs)
        (folder/"COMPLETE.json").write_text(json.dumps({"complete":True,"n_subjects":5,"n_figures":7,"mode":ANALYSIS_MODE}),encoding="utf-8")
        print(f"\nDONE. Results: {folder}",flush=True)
    except Exception:
        (folder/"FAILED.txt").write_text(traceback.format_exc(),encoding="utf-8")
        raise


if __name__=="__main__":
    main()
