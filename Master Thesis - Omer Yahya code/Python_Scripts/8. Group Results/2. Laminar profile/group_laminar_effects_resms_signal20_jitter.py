"""Complete supplementary saved-GLM effects, ResMS and full 20-depth signal plots.
Uses matched Without Nordic-selected Top200. Read-only inputs, timestamped outputs.
Install: python -m pip install numpy pandas h5py matplotlib
Run: python group_laminar_effects_resms_signal20_jitter.py
No new GLM, filtering, smoothing, detrending, residual reconstruction or inference.
"""
from pathlib import Path
from datetime import datetime
import json
import traceback
import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ============================ EDIT PATHS ============================
ROOTS={
    "7485":Path(r"D:\NORDIC_Subject1_project\Laminarprofile"),
    "7468":Path(r"E:\layerfmri\LAYER_s7468_warpednordic\codes\Laminarprofile\Laminar"),
    "7469":Path(r"E:\layerfmri\LAYER_s7469_warpenordic\codes\laminar profile"),
    "7482":Path(r"E:\layerfmri\LAYER_s7482_Warpenordic\codes\Laminar profile"),
    "7484":Path(r"E:\layerfmri\LAYER_s7484_warpenordic\codes\Laminarprofile"),
}
GLM_DIRS={s:root/"2. GLM 75%"/"GLM" for s,root in ROOTS.items()}
GLM_DIRS["7485"]=ROOTS["7485"]/"2. GLM 75%"/"GLM"/"GLM_results_fixed_SPM_weights"
OUTPUT_ROOT=Path(r"C:\Users\L E N O V O\OneDrive\Desktop\Group  Analysis\laminar profile\1\Effects_ResMS_Signal20_Jitter")
FILE_OVERRIDES={}
# Example: FILE_OVERRIDES[("7485","Nordic")]=Path(r"D:\...\existing_GLM.h5")
# ===================================================================
TEMPERATURE_MAPPING_CONFIRMED=False
PLOT_TIMECOURSES=True
TIMECOURSE_RUN=1  # 1,2,3; or None for separate figures for all runs (30 figures).
TOP_N=200
DPI=300
TRACE_DPI=180
PROFILE_GROUP="matched_without_selection"
SIDES=("left","right")
CONDITIONS=("Without Nordic","Nordic")
TOKENS={"Without Nordic":"without_NORDIC","Nordic":"with_NORDIC"}
STYLES={"Without Nordic":"--","Nordic":"-"}
COLOURS={"7485":"#4477AA","7468":"#EE6677","7469":"#228833","7482":"#AA3377","7484":"#CCBB44"}
METRICS=("pain_effect","cognition_effect","resms","root_resms","mean_signal")
SCOPE="Stored depth-level GLM | Without Nordic-selected Top200, matched in both conditions"


def text(x):
    return x.decode("utf-8",errors="replace") if isinstance(x,bytes) else str(x)


def glm_path(s,c):
    return Path(FILE_OVERRIDES[(s,c)]) if (s,c) in FILE_OVERRIDES else GLM_DIRS[s]/f"pIns_GLM_{TOKENS[c]}_deep75_top200_layerwise.h5"


def metric_title(m):
    return {
        "pain_effect":"Temperature contrast (High - Low)" if TEMPERATURE_MAPPING_CONFIRMED else "Pain contrast (High pain - Low pain)",
        "cognition_effect":"Cognition contrast (Low cognition - High cognition)",
        "resms":"Model residual mean square (ResMS)",
        "root_resms":"Square root of model ResMS",
        "mean_signal":"Mean sampled signal across 20 depths",
    }[m]


def unit(m):
    if m.endswith("effect"):
        return "Contrast estimate (input intensity units)"
    return {"resms":"ResMS (input units squared)","root_resms":"Root ResMS (input units)",
            "mean_signal":"Mean sampled signal (input units)"}[m]


def jitter(x,subject_index,n_subjects=5,width=.12,condition_index=None,condition_gap=0):
    """Deterministic horizontal DISPLAY offsets. Never change y or numeric data."""
    if not 0<=subject_index<n_subjects:
        raise ValueError("Invalid subject index for jitter")
    offset=np.linspace(-width,width,n_subjects)[subject_index] if n_subjects>1 else 0
    if condition_index is not None:
        offset+=(condition_index-.5)*condition_gap
    return np.asarray(x,dtype=float)+offset


def integers(x,name):
    x=np.asarray(x)
    if x.ndim!=1 or not np.isfinite(x).all() or not np.all(x==x.astype(np.int64)):
        raise ValueError(f"Invalid integer vector: {name}")
    return x.astype(np.int64)


def vector(group,key,n):
    x=np.asarray(group[key][:],dtype=float)
    if x.shape!=(n,) or not np.isfinite(x).all():
        raise ValueError(f"Invalid profile: {group.name}/{key}")
    return x


def matched_selection(profile,all_vertices,reference_idx,reference_vertices):
    idx=integers(profile["selected_column_indices_zero_based"][:],"matched indices")
    vertices=integers(profile["selected_surface_vertex_indices"][:],"matched vertex IDs")
    if len(idx)!=TOP_N or len(np.unique(idx))!=TOP_N or np.any(idx<0) or np.any(idx>=len(all_vertices)):
        raise ValueError("Matched selection must contain exactly 200 valid unique columns")
    if not np.array_equal(vertices,all_vertices[idx]):
        raise ValueError("Selected column-to-vertex mapping mismatch")
    if not np.array_equal(np.sort(idx),np.sort(reference_idx)) or not np.array_equal(np.sort(vertices),np.sort(reference_vertices)):
        raise ValueError("Profiles do not use the same Without Nordic-selected Top200")
    return idx,vertices


def signal_means_by_run(ts,runs):
    ts=np.asarray(ts,dtype=float)
    if ts.shape!=(20,int(np.sum(runs))) or not np.isfinite(ts).all():
        raise ValueError("Expected finite matched-layer time series: 20 x sum(run_lengths)")
    edges=np.r_[0,np.cumsum(runs)]
    per_run=np.stack([ts[:,a:b].mean(axis=1) for a,b in zip(edges[:-1],edges[1:])])
    return per_run.mean(axis=0),per_run


def read_inputs():
    rows,signal_run_rows,selection_rows,provenance=[],[],[],[]
    traces={}
    common_depths=None
    common_scaling=None
    for s in ROOTS:
        print(f"\nSUBJECT {s}",flush=True)
        with h5py.File(glm_path(s,CONDITIONS[0]),"r") as a,h5py.File(glm_path(s,CONDITIONS[1]),"r") as b:
            handles=dict(zip(CONDITIONS,[a,b]))
            reference_runs=None
            for c,g in handles.items():
                if text(g.attrs.get("condition",""))!=TOKENS[c]:
                    raise ValueError(f"{s}/{c}: condition metadata mismatch")
                if "subject" in g.attrs and text(g.attrs["subject"])!=s:
                    raise ValueError("Subject metadata mismatch")
                if not bool(g.attrs.get("complete",False)):
                    raise ValueError("GLM completion marker absent/false")
                runs=integers(g["run_lengths"][:],"run_lengths")
                if len(runs)!=3 or np.any(runs<2):
                    raise ValueError("Expected three runs for these saved contrast estimates")
                if reference_runs is None:
                    reference_runs=runs
                if not np.array_equal(runs,reference_runs):
                    raise ValueError("Paired run lengths differ")
                scaling=text(g.attrs.get("effect_scaling",""))
                pain=text(g.attrs.get("pain_direction",""))
                cognition=text(g.attrs.get("cognition_direction",""))
                if "high pain - low pain" not in pain.lower() or "low cognition - high cognition" not in cognition.lower():
                    raise ValueError("Unrecognized contrast directions")
                if not scaling:
                    raise ValueError("Missing contrast scaling metadata")
                if common_scaling is None:
                    common_scaling=scaling
                if scaling!=common_scaling:
                    raise ValueError("Contrast scaling differs across files")
                provenance.append({"subject":s,"condition":c,"file":str(glm_path(s,c)),
                    "spm_source":text(g.attrs.get("spm_source","unknown")),
                    "estimation":text(g.attrs.get("estimation","unknown")),
                    "effect_scaling":scaling,"pain_direction":pain,"cognition_direction":cognition,
                    "run_lengths":runs.tolist()})
            for side in SIDES:
                av=integers(a[f"{side}/surface_vertex_indices"][:],"baseline vertices")
                bv=integers(b[f"{side}/surface_vertex_indices"][:],"Nordic vertices")
                if not np.array_equal(av,bv) or len(np.unique(av))!=len(av):
                    raise ValueError("Paired GLM vertices differ or are duplicated")
                ri=integers(a[f"{side}/selected_column_indices_zero_based"][:],"reference indices")
                rv=integers(a[f"{side}/selected_surface_vertex_indices"][:],"reference vertex IDs")
                if len(ri)!=TOP_N or len(np.unique(ri))!=TOP_N or np.any(ri<0) or np.any(ri>=len(av)) or not np.array_equal(rv,av[ri]):
                    raise ValueError("Invalid reference Top200 selection")
                for c,g in handles.items():
                    key=f"{side}/{PROFILE_GROUP}"
                    if key not in g:
                        raise KeyError(f"Missing {key}; no independent-selection fallback")
                    profile=g[key]
                    idx,vertices=matched_selection(profile,av,ri,rv)
                    depths=vector(profile,"layer_depth",20)
                    if np.any(np.diff(depths)<=0) or not np.isclose(depths[0],0) or not np.isclose(depths[-1],1):
                        raise ValueError("Invalid white-to-pial depth grid")
                    if common_depths is None:
                        common_depths=depths
                    if not np.allclose(depths,common_depths,atol=1e-10,rtol=0):
                        raise ValueError("Depth grids differ between models")
                    pain=vector(profile,"layer_effect_pain",20)
                    cognition=vector(profile,"layer_effect_cognition",20)
                    resms=vector(profile,"layer_resms",20)
                    if np.any(resms<0):
                        raise ValueError("Negative ResMS")
                    for name,values in [("layer_effect_pain_spm_sum",pain),("layer_effect_cognition_spm_sum",cognition)]:
                        if name in profile and not np.allclose(vector(profile,name,20),6*values,atol=1e-6,rtol=1e-6):
                            raise ValueError("Stored contrast sum/mean scaling mismatch")
                    ts=np.asarray(profile["layer_mean_timeseries"][:],dtype=float)
                    signal,per_run=signal_means_by_run(ts,reference_runs)
                    traces[(s,side,c)]={"timeseries":ts,"run_lengths":reference_runs.copy(),"depths":depths.copy()}
                    values={"pain_effect":pain,"cognition_effect":cognition,"resms":resms,
                            "root_resms":np.sqrt(resms),"mean_signal":signal}
                    for m,v in values.items():
                        for j,d in enumerate(depths):
                            rows.append({"subject":s,"hemisphere":side,"condition":c,"metric":m,
                                "depth_index":j,"layer_number":j+1,"depth_fraction":float(d),"value":float(v[j]),"selected_columns":len(idx)})
                    for run in range(len(reference_runs)):
                        for j,d in enumerate(depths):
                            signal_run_rows.append({"subject":s,"hemisphere":side,"condition":c,"run":run+1,
                                "layer_number":j+1,"depth_fraction":float(d),"mean_signal":float(per_run[run,j])})
                    selection_rows.extend({"subject":s,"hemisphere":side,"condition":c,
                        "column_index_zero_based":int(i),"surface_vertex_id":int(v)} for i,v in zip(idx,vertices))
                print(f"  {side}: effects, ResMS and all 20 matched-layer time series read",flush=True)
    return pd.DataFrame(rows),pd.DataFrame(signal_run_rows),pd.DataFrame(selection_rows),provenance,traces


def summaries(df):
    group=df.groupby(["hemisphere","condition","metric","depth_index","layer_number","depth_fraction"],as_index=False).agg(
        mean=("value","mean"),sd=("value","std"),n=("subject","nunique"))
    if not (group.n==5).all():
        raise ValueError("All group cells must contain five subjects")
    diff=df.pivot(index=["subject","hemisphere","metric","depth_index","layer_number","depth_fraction"],columns="condition",values="value").reset_index()
    if diff[list(CONDITIONS)].isna().any().any():
        raise ValueError("Incomplete paired profiles")
    diff["difference"]=diff["Nordic"]-diff["Without Nordic"]
    return group,diff


def legend_handles():
    return [Line2D([0],[0],color=COLOURS[s],lw=2,label=s) for s in ROOTS]+[
        Line2D([0],[0],color="gray",ls=STYLES[c],lw=2,label=c) for c in CONDITIONS]+[
        Line2D([0],[0],color="black",lw=3,label="Group mean")]


def style(ax,label):
    ax.set(xlim=(0,1),xlabel="Cortical depth (white matter to pial)",ylabel=label)
    ax.set_xticks([0,.25,.5,.75,1]);ax.grid(alpha=.16)
    ax.spines[["top","right"]].set_visible(False)


def save(fig,folder,name,dpi=DPI):
    for ext in ("png","pdf","svg"):
        fig.savefig(folder/f"{name}.{ext}",dpi=dpi,bbox_inches="tight",facecolor="white")
    plt.close(fig)


def overview(df,group,metrics,folder,name,heading):
    fig,axes=plt.subplots(len(metrics),2,figsize=(14,4.2*len(metrics)),squeeze=False,constrained_layout=True)
    for row,m in enumerate(metrics):
        for col,side in enumerate(SIDES):
            ax=axes[row,col]
            for c in CONDITIONS:
                for s in ROOTS:
                    d=df.query("subject == @s and hemisphere == @side and condition == @c and metric == @m").sort_values("depth_fraction")
                    ax.plot(d.depth_fraction,d.value,color=COLOURS[s],ls=STYLES[c],lw=1.6,alpha=.9)
                d=group.query("hemisphere == @side and condition == @c and metric == @m").sort_values("depth_fraction")
                x,y,sd=d.depth_fraction.to_numpy(),d["mean"].to_numpy(),d.sd.to_numpy()
                ax.fill_between(x,y-sd,y+sd,color="gray",alpha=.08)
                ax.plot(x,y,color="black",ls=STYLES[c],lw=3)
            if m.endswith("effect"):
                ax.axhline(0,color="gray",lw=.8)
            style(ax,unit(m));ax.set_title(side.title()+" | "+metric_title(m))
    axes[0,1].legend(handles=legend_handles(),frameon=False,loc="upper left",bbox_to_anchor=(1.02,1),fontsize=8)
    fig.suptitle(heading+"\n"+SCOPE+"\nBands: between-subject SD; descriptive summaries",fontsize=12)
    save(fig,folder,name)


def individual(df,m,folder,name):
    fig,axes=plt.subplots(5,2,figsize=(12,14),sharex=True,constrained_layout=True)
    for row,s in enumerate(ROOTS):
        for col,side in enumerate(SIDES):
            ax=axes[row,col]
            for c in CONDITIONS:
                d=df.query("subject == @s and hemisphere == @side and condition == @c and metric == @m").sort_values("depth_fraction")
                ax.plot(d.depth_fraction,d.value,color=COLOURS[s],ls=STYLES[c],lw=2,label=c)
            if m.endswith("effect"):
                ax.axhline(0,color="gray",lw=.8)
            style(ax,unit(m));ax.set_title(f"Subject {s} | {side.title()}")
            if row==0:
                ax.legend(frameon=False,fontsize=8)
    fig.suptitle(metric_title(m)+" | individual profiles\n"+SCOPE,fontsize=12)
    save(fig,folder,name)


def differences(diff,folder):
    metrics=("pain_effect","cognition_effect","resms","mean_signal")
    fig,axes=plt.subplots(4,2,figsize=(14,16),constrained_layout=True)
    for row,m in enumerate(metrics):
        for col,side in enumerate(SIDES):
            ax=axes[row,col]
            for s in ROOTS:
                d=diff.query("subject == @s and hemisphere == @side and metric == @m").sort_values("depth_fraction")
                ax.plot(d.depth_fraction,d.difference,color=COLOURS[s],lw=2,label=s)
            d=diff.query("hemisphere == @side and metric == @m").groupby("depth_fraction").difference.agg(["mean","std"])
            x,y,sd=d.index.to_numpy(),d["mean"].to_numpy(),d["std"].to_numpy()
            ax.fill_between(x,y-sd,y+sd,color="gray",alpha=.12);ax.plot(x,y,color="black",lw=3,label="Group mean")
            ax.axhline(0,color="gray",ls="--",lw=1)
            style(ax,"Nordic - Without Nordic\n"+unit(m));ax.set_title(side.title()+" | "+metric_title(m))
    axes[0,1].legend(frameon=False,loc="upper left",bbox_to_anchor=(1.02,1))
    fig.suptitle("Paired depth differences\n"+SCOPE,fontsize=12)
    save(fig,folder,"Figure_06_Paired_depth_differences")


def paired_summary(df,folder):
    averaged=df.groupby(["subject","hemisphere","condition","metric"],as_index=False).value.mean()
    averaged.to_csv(folder.parent/"subject_depth_averaged_summary.csv",index=False)
    fig,axes=plt.subplots(4,2,figsize=(13,16),constrained_layout=True)
    for row,m in enumerate(("pain_effect","cognition_effect","resms","mean_signal")):
        for col,side in enumerate(SIDES):
            ax=axes[row,col]
            for i,s in enumerate(ROOTS):
                d=averaged.query("subject == @s and hemisphere == @side and metric == @m").set_index("condition").reindex(CONDITIONS)
                ax.plot(jitter([0,1],i,width=.11),d.value,"o-",color=COLOURS[s],lw=1.8,ms=6,label=s)
            d=averaged.query("hemisphere == @side and metric == @m").groupby("condition").value.agg(["mean","std"]).reindex(CONDITIONS)
            ax.errorbar([0,1],d["mean"],yerr=d["std"],fmt="D",color="black",capsize=5,ms=7,label="Mean +/- SD")
            if m.endswith("effect"):
                ax.axhline(0,color="gray",lw=.8)
            ax.set_xticks([0,1],CONDITIONS);ax.set(xlim=(-.3,1.3),ylabel=unit(m),title=side.title()+" | "+metric_title(m))
            ax.grid(axis="y",alpha=.16);ax.spines[["top","right"]].set_visible(False)
    axes[0,1].legend(frameon=False,loc="upper left",bbox_to_anchor=(1.02,1))
    fig.suptitle("Paired subject summary | horizontal jitter only; values unchanged\n"+SCOPE,fontsize=12)
    save(fig,folder,"Figure_07_Paired_subject_summary_jitter")


def full_signal_depth(df,group,folder):
    m="mean_signal"
    fig,axes=plt.subplots(1,2,figsize=(16,5.5),sharey=True,constrained_layout=True)
    for ax,side in zip(axes,SIDES):
        for ci,c in enumerate(CONDITIONS):
            for i,s in enumerate(ROOTS):
                d=df.query("subject == @s and hemisphere == @side and condition == @c and metric == @m").sort_values("layer_number")
                x=d.layer_number.to_numpy();y=d.value.to_numpy()
                ax.plot(x,y,color=COLOURS[s],ls=STYLES[c],lw=1.7,alpha=.85)
                shown_x=jitter(x,i,width=.14,condition_index=ci,condition_gap=.08)
                ax.scatter(shown_x,y,s=16,edgecolors=COLOURS[s],facecolors="white" if ci==0 else COLOURS[s],linewidths=.8,zorder=4)
            d=group.query("hemisphere == @side and condition == @c and metric == @m").sort_values("layer_number")
            x,y,sd=d.layer_number.to_numpy(),d["mean"].to_numpy(),d.sd.to_numpy()
            ax.fill_between(x,y-sd,y+sd,color="gray",alpha=.08);ax.plot(x,y,color="black",ls=STYLES[c],lw=3)
        ax.set(xlim=(.65,20.35),xlabel="Sampling depth index: 1 = white matter, 20 = pial",ylabel=unit(m),title=side.title()+" posterior insula")
        ax.set_xticks(np.arange(1,21));ax.tick_params(axis="x",labelsize=8)
        ax.grid(alpha=.16);ax.spines[["top","right"]].set_visible(False)
    axes[1].legend(handles=legend_handles(),frameon=False,loc="upper left",bbox_to_anchor=(1.02,1),fontsize=8)
    fig.suptitle("Mean signal from first to last sampled depth (all 20)\n"+SCOPE+"\nEqual mean of run means; marker x-jitter only, curves stay at true depth indices",fontsize=12)
    save(fig,folder,"Figure_08_Full20_mean_signal_depth_jitter")


def timecourse_figures(traces,folder):
    count=0
    for s in ROOTS:
        for side in SIDES:
            runs=traces[(s,side,CONDITIONS[0])]["run_lengths"]
            chosen=range(1,len(runs)+1) if TIMECOURSE_RUN is None else [TIMECOURSE_RUN]
            edges=np.r_[0,np.cumsum(runs)]
            for run in chosen:
                if not isinstance(run,int) or not 1<=run<=len(runs):
                    raise ValueError("TIMECOURSE_RUN must be a valid integer run number or None")
                start,stop=edges[run-1],edges[run]
                x=np.arange(1,stop-start+1)
                fig,axes=plt.subplots(5,4,figsize=(17,13),sharex=True,sharey=True,constrained_layout=True)
                for j,ax in enumerate(axes.ravel()):
                    for c in CONDITIONS:
                        y=traces[(s,side,c)]["timeseries"][j,start:stop]
                        ax.plot(x,y,color=COLOURS[s],ls=STYLES[c],lw=.8,alpha=.85)
                    d=traces[(s,side,CONDITIONS[0])]["depths"][j]
                    ax.set_title(f"Depth {j+1}/20 | fraction={d:.2f}",fontsize=9)
                    ax.grid(alpha=.12);ax.spines[["top","right"]].set_visible(False)
                    if j>=16:
                        ax.set_xlabel("Volume within run")
                    if j%4==0:
                        ax.set_ylabel("Signal (input units)")
                fig.legend(handles=[Line2D([0],[0],color=COLOURS[s],ls=STYLES[c],label=c) for c in CONDITIONS],
                    loc="upper center",bbox_to_anchor=(.5,1.02),ncol=2,frameon=False)
                fig.suptitle(f"Subject {s} | {side.title()} pIns | Run {run} | all 20 depth time courses\n"
                    "Stored matched Top200 mean signal; common y-axis; no temporal jitter, normalization or offset",fontsize=12)
                save(fig,folder,f"Signal_timecourses_s{s}_{side}_run{run}_all20depths",dpi=TRACE_DPI)
                count+=1
                print(f"  Time-course figure: {s}/{side}/run{run}",flush=True)
    return count


def main():
    if len(ROOTS)!=5 or any(s not in COLOURS for s in ROOTS):
        raise ValueError("Use five numeric subject IDs, not Subject1..5")
    for s in ROOTS:
        for c in CONDITIONS:
            path=glm_path(s,c)
            if not path.is_file():
                raise FileNotFoundError(f"Missing {s}/{c}: {path}")
    folder=OUTPUT_ROOT/datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    folder.mkdir(parents=True,exist_ok=False)
    figures=folder/"figures";figures.mkdir()
    try:
        df,run_signal,audit,provenance,traces=read_inputs()
        group,diff=summaries(df)
        for name,table in [("subject_profiles",df),("group_profiles",group),("paired_depth_differences",diff),
                           ("signal_by_run_depth",run_signal),("matched_selection_audit",audit)]:
            table.to_csv(folder/f"{name}.csv",index=False)
        arrays={f"s{s}_{side}_{TOKENS[c]}":v["timeseries"] for (s,side,c),v in traces.items()}
        for s in ROOTS:
            arrays[f"s{s}_run_lengths"]=traces[(s,"left",CONDITIONS[0])]["run_lengths"]
        arrays["depth_fractions"]=traces[(next(iter(ROOTS)),"left",CONDITIONS[0])]["depths"]
        np.savez_compressed(folder/"matched_layer_signal_timeseries.npz",**arrays)
        metadata={"scope":SCOPE,"source_models":provenance,"temperature_mapping_confirmed":TEMPERATURE_MAPPING_CONFIRMED,
            "new_GLM":False,"new_group_GLM":False,"inferential_tests":False,"input_smoothing_verified":False,
            "raw_residual_time_series_generated":False,"residual_output":"Stored ResMS only; root ResMS is not a raw residual trace",
            "signal_source":"matched_without_selection/layer_mean_timeseries, stored Top200-averaged sampled signal",
            "signal_depth_mean":"Mean across time within each run, then equal average across runs, then equal average of subjects",
            "signal_timecourse_run":TIMECOURSE_RUN,"x_axis_timecourses":"Volume within run; no unverified TR assumed",
            "jitter":"Deterministic horizontal marker offsets only; no y change, no temporal jitter; line depths unchanged",
            "selection_independent":False,"condition_styles":STYLES,"subject_colours":COLOURS}
        (folder/"provenance_and_scope.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
        methods=("SAVED GLM EFFECTS, RESMS AND TWENTY-DEPTH SIGNAL VISUALIZATION\n\n"+SCOPE+"\n"
            "Pain contrast: High pain - Low pain; cognition: Low cognition - High cognition.\n"
            "Temperature label disabled unless exact stimulus-contrast mapping is confirmed.\n"
            "Saved contrast mean estimates are checked against SPM contrast sums / 6.\n"
            "Existing frozen-SPM-weight GLM results are reused; no fresh first-level estimation.\n"
            "Matched Top200 selection is checked within each subject/hemisphere. Same-data selection is not independent.\n"
            "ResMS and root ResMS are stored model residual summaries, not residual traces or voxel residual images.\n"
            "Mean signal curves use stored matched layer mean time series: equal run means, then equal subject means.\n"
            "Signal curves and contrast estimates are different quantities; neither is automatically percent signal change.\n"
            "Depth labels 1..20 correspond to normalized positions 0..1, not twenty independently resolved histological layers.\n"
            "No additional filtering, detrending, smoothing, centering or normalization is performed.\n"
            "Jitter changes horizontal marker locations only. Values, saved numeric data, line depths and time axes are unchanged.\n"
            "Time courses are displayed per run to avoid connecting unrelated run boundaries.\n"
            "Default trace plots show run 1 only at all depths; mean signal-depth curves include all runs.\n"
            "Input smoothing and anatomical registration are not independently certified.\n"
            "Group bands and error bars are between-subject sample SD, not confidence intervals. No statistical tests.\n")
        (folder/"METHODS_AND_LIMITATIONS.txt").write_text(methods,encoding="utf-8")
        plt.rcParams.update({"font.size":9,"axes.titlesize":11,"pdf.fonttype":42,"ps.fonttype":42})
        print("\nDrawing eight main figures...",flush=True)
        overview(df,group,("pain_effect","cognition_effect"),figures,"Figure_01_Contrast_estimates_depth","Pain and cognition contrast estimates across cortical depth")
        overview(df,group,("resms","root_resms"),figures,"Figure_02_Residual_variance_depth","Saved model residual-variance summaries")
        individual(df,"pain_effect",figures,"Figure_03_Individual_pain_effect")
        individual(df,"cognition_effect",figures,"Figure_04_Individual_cognition_effect")
        individual(df,"resms",figures,"Figure_05_Individual_ResMS")
        differences(diff,figures)
        paired_summary(df,figures)
        full_signal_depth(df,group,figures)
        trace_count=0
        if PLOT_TIMECOURSES:
            trace_folder=folder/"timecourses_all20depths";trace_folder.mkdir()
            trace_count=timecourse_figures(traces,trace_folder)
        (folder/"COMPLETE.json").write_text(json.dumps({"complete":True,"n_subjects":5,
            "main_figures":8,"timecourse_figures":trace_count,"new_GLM":False}),encoding="utf-8")
        print(f"\nDONE. Results: {folder}",flush=True)
        print(f"Created 8 main figures + {trace_count} time-course figures; each in PNG/PDF/SVG.",flush=True)
    except Exception:
        (folder/"FAILED.txt").write_text(traceback.format_exc(),encoding="utf-8")
        raise


if __name__=="__main__":
    main()
