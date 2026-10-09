"""7485 inflated pIns + EXISTING NORDIC Top200; depth-resolved tSNR.
Read-only inputs. No GLM, re-selection, registration, smoothing, or resampling.
Primary tSNR: mean_time/SD_time (ddof=1) PER column/depth/run, then average
columns and runs. Also exports ROI-mean-time-course tSNR as a DIFFERENT metric.
Raw time courses; no detrending or task regression. Descriptive single-subject QC.
"""
from pathlib import Path
from io import BytesIO
import sys
import json
import time
import numpy as np
import pandas as pd
import h5py
import nibabel as nib
from nibabel.freesurfer.io import read_geometry, read_morph_data
from scipy import sparse
from scipy.sparse.csgraph import connected_components
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from nilearn import plotting

# FULL PATHS: change here only.
SUBJECT='7485'
SURF_DIR=Path(r'D:\NORDIC_Subject1_project\7485_extracted\7485\surf')
WITHOUT_H5=Path(r'D:\NORDIC_Subject1_project\Laminarprofile\1. Sampling\Without\pIns_laminar_sampling_7485_without_NORDIC.h5')
WITH_H5=Path(r'D:\NORDIC_Subject1_project\Laminarprofile\1. Sampling\With\pIns_laminar_sampling_7485_with_NORDIC.h5')
ANALYSIS_ROOT=Path(r'D:\NORDIC_Subject1_project\Laminarprofile\Comparison\NORDIC_selected_matched_columns_7485')
# Optional exact full Excel path. Empty = latest COMPLETED analysis in ANALYSIS_ROOT.
SELECTION_XLSX =r"D:\NORDIC_Subject1_project\Laminarprofile\4. QC Comparison\2\20261005_232408\QC_and_NORDIC_matched_profiles.xlsx"
OUT_ROOT=Path(r'D:\NORDIC_Subject1_project\Laminarprofile\TSNR Depth\2\2')

SIDES=('left','right')
CONDITIONS=('without_NORDIC','with_NORDIC')
FILES={'without_NORDIC':WITHOUT_H5,'with_NORDIC':WITH_H5}
EXPECTED_RUN_LENGTHS=np.array([407,407,407])
BLOCK=32
DPI=300
ROI_COLOR='#1B8A89'
TOP_COLOR='#E76F51'
PRE_COLOR='#0072B2'
POST_COLOR='#D55E00'
SURFACE_VIEW='lateral'
# H5s in this project use concatenated lh+rh IDs, not ambiguous auto-detection.
H5_VERTEX_INDEXING='global_bilateral'


def ensure(ok,message):
    if not ok: raise ValueError(message)


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}",flush=True)


def choose_excel():
    if SELECTION_XLSX:
        path=Path(SELECTION_XLSX)
        ensure(path.is_file(),f'Excel missing: {path}')
        return path
    candidates=[]
    for path in ANALYSIS_ROOT.rglob('QC_and_NORDIC_matched_profiles.xlsx'):
        marker=path.parent/'analysis_complete.json'
        if not marker.is_file(): continue
        metadata=json.loads(marker.read_text(encoding='utf-8'))
        if metadata.get('complete') and str(metadata.get('subject'))==SUBJECT:
            candidates.append(path)
    ensure(bool(candidates),'No COMPLETED analysis Excel found. Set SELECTION_XLSX to the exact full path of your reviewed Excel.')
    selected=max(candidates,key=lambda p:p.stat().st_mtime)
    log(f'Using selection from latest completed analysis:\n{selected}')
    return selected


def local_ids(ids,side,n_lh,n_rh):
    ids=np.asarray(ids,dtype=int)
    if H5_VERTEX_INDEXING=='global_bilateral':
        local=ids if side=='left' else ids-n_lh
    elif H5_VERTEX_INDEXING=='local_per_hemisphere':
        local=ids
    else: raise ValueError('Invalid H5_VERTEX_INDEXING setting')
    limit=n_lh if side=='left' else n_rh
    ensure(np.all((local>=0)&(local<limit)),f'{side}: IDs outside expected hemisphere; inspect indexing')
    return local


def edges_of(faces):
    return np.unique(np.sort(np.vstack((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]])),axis=1),axis=0)


def component_sizes(edges,ids):
    lookup={int(v):i for i,v in enumerate(ids)}
    chosen=edges[np.isin(edges[:,0],ids)&np.isin(edges[:,1],ids)]
    a=np.array([lookup[int(v)] for v in chosen[:,0]],dtype=int)
    b=np.array([lookup[int(v)] for v in chosen[:,1]],dtype=int)
    graph=sparse.coo_matrix((np.ones(len(a)*2),(np.r_[a,b],np.r_[b,a])),shape=(len(ids),len(ids))).tocsr()
    n,labels=connected_components(graph,directed=False)
    return n,np.sort(np.bincount(labels))[::-1]


def prepare_inputs(selection):
    surface={}
    for side,hemi in (('left','lh'),('right','rh')):
        vertices,faces=read_geometry(str(SURF_DIR/f'{hemi}.inflated'))
        sulc=read_morph_data(str(SURF_DIR/f'{hemi}.sulc'))
        ensure(len(sulc)==len(vertices),'Sulc/surface vertex count mismatch')
        surface[side]={'vertices':vertices,'faces':faces,'sulc':sulc}
    nlh=len(surface['left']['vertices']); nrh=len(surface['right']['vertices'])
    ids={}; selected_indices={}; selection_rows=[]; qc=[]
    with h5py.File(WITH_H5,'r') as post,h5py.File(WITHOUT_H5,'r') as pre:
        for label,f in (('with_NORDIC',post),('without_NORDIC',pre)):
            ensure(str(f.attrs.get('subject',''))==SUBJECT,f'{label}: subject metadata mismatch')
            ensure(str(f.attrs.get('condition',''))==label,f'{label}: condition metadata mismatch')
            ensure(np.array_equal(f['run_lengths'][:],EXPECTED_RUN_LENGTHS),'Run lengths must be 407/407/407')
            if 'complete' in f.attrs: ensure(bool(f.attrs['complete']),f'{label}: H5 incomplete')
        for side in SIDES:
            gids=post[side]['surface_vertex_indices'][:].astype(int)
            ensure(np.array_equal(gids,pre[side]['surface_vertex_indices'][:]),f'{side}: pre/post IDs or order differ')
            ensure(len(np.unique(gids))==len(gids),f'{side}: duplicate IDs')
            for f in (pre,post): ensure(f[side]['layeractivation'].shape==(len(gids),20,1221),'Invalid layeractivation dimensions')
            sub=selection[selection.hemisphere==side].sort_values('rank').copy()
            ensure(len(sub)==200 and sub.surface_vertex_id.nunique()==200,f'{side}: need exactly 200 unique selected vertex IDs')
            ensure(np.array_equal(sub['rank'].to_numpy(),np.arange(1,201)),f'{side}: invalid ranks')
            ensure((sub.t_pain_NORDIC>0).all() and (sub.t_cognition_NORDIC>0).all(),f'{side}: selected T values not both positive')
            ensure(np.allclose(sub.conjunction_min_t_NORDIC,np.minimum(sub.t_pain_NORDIC,sub.t_cognition_NORDIC)),'Conjunction score mismatch')
            lookup={int(v):i for i,v in enumerate(gids)}
            ensure(set(sub.surface_vertex_id.astype(int)).issubset(lookup),'Selected IDs not contained in sampling ROI')
            index=np.array([lookup[int(v)] for v in sub.surface_vertex_id],dtype=int)
            ensure(np.array_equal(index,sub.local_index_zero_based.astype(int).to_numpy()),'Excel local indices do not match current H5 ordering')
            selected_indices[side]=index; ids[side]=gids
            local=local_ids(gids,side,nlh,nrh); top_local=local[index]
            surface[side]['roi_ids']=local; surface[side]['top_ids']=top_local
            edge=edges_of(surface[side]['faces'])
            ncomp,sizes=component_sizes(edge,top_local)
            flag=np.isin(gids,sub.surface_vertex_id.astype(int))
            selection_rows.append(pd.DataFrame({'subject':SUBJECT,'hemisphere':side,'local_column_index':np.arange(len(gids)),'surface_vertex_id':gids,'local_surface_vertex_id':local,'selected_NORDIC_top200':flag}))
            qc.append({'subject':SUBJECT,'hemisphere':side,'full_mesh_vertices':len(surface[side]['vertices']),'ROI_columns':len(gids),'Top200_columns':len(index),'fraction_of_ROI_vertices':len(index)/len(gids),'fraction_of_full_mesh_vertices':len(index)/len(surface[side]['vertices']),'Top200_connected_components':ncomp,'largest_component_vertices':int(sizes[0]),'isolated_selected_vertices':int(np.count_nonzero(sizes==1))})
            log(f'{side}: ROI={len(gids)}, Top200=200 ({100*200/len(gids):.2f}% of ROI vertices), connected components={ncomp}')
    return surface,ids,selected_indices,pd.concat(selection_rows,ignore_index=True),pd.DataFrame(qc)


def render_surface(side,data):
    labels=np.zeros(len(data['vertices']),dtype=int)
    labels[data['roi_ids']]=1; labels[data['top_ids']]=2
    fig=plt.figure(figsize=(6.4,5.4),facecolor='white')
    ax=fig.add_subplot(111,projection='3d')
    plotting.plot_surf_roi(surf_mesh=(data['vertices'],data['faces']),roi_map=labels,bg_map=data['sulc'],hemi=side,view=SURFACE_VIEW,engine='matplotlib',cmap=ListedColormap(['#DDDDDD',ROI_COLOR,TOP_COLOR]),colorbar=False,avg_method='median',threshold=.5,vmin=0,vmax=2,bg_on_data=False,axes=ax,figure=fig)
    # Exact selected vertex markers: categorical face voting can hide sparse vertices.
    v=data['vertices'][data['top_ids']]
    center=data['vertices'].mean(axis=0); direction=v-center
    norm=np.linalg.norm(direction,axis=1,keepdims=True)
    pts=v+0.5*direction/np.maximum(norm,1e-8)
    ax.scatter(pts[:,0],pts[:,1],pts[:,2],s=6,c=TOP_COLOR,edgecolors='#5A291A',linewidths=.15,depthshade=False)
    buffer=BytesIO(); fig.savefig(buffer,format='png',dpi=DPI,bbox_inches='tight',pad_inches=.01,facecolor='white'); plt.close(fig)
    buffer.seek(0); image=plt.imread(buffer)
    nonwhite=np.any(image[:,:,:3]<.97,axis=2)
    yy,xx=np.where(nonwhite)
    if len(xx): image=image[max(0,yy.min()-10):min(image.shape[0],yy.max()+11),max(0,xx.min()-10):min(image.shape[1],xx.max()+11)]
    return image


def schematic(ax):
    ax.axhspan(-.015,14/19,facecolor='#EFEFEF')
    ax.plot([.5,.5],[0,1],color='#333333',lw=1.4)
    ax.scatter(np.full(20,.5),np.linspace(0,1,20),c=np.linspace(0,1,20),cmap='viridis',s=32,edgecolors='#333333',lw=.3)
    ax.text(.41,0,'White/GM\nd = 0; index 0',ha='right',va='center',fontsize=9)
    ax.text(.41,1,'Pial\nd = 1; index 19',ha='right',va='center',fontsize=9)
    ax.annotate('',xy=(.76,1),xytext=(.76,0),arrowprops={'arrowstyle':'->','lw':1.4})
    ax.text(.83,.5,'Deep to superficial',rotation=90,va='center',fontsize=9)
    ax.text(.5,-.17,'20 equidistant samples\nGray band: depths 0-14 (deep75)\nNot histological layers',ha='center',va='center',fontsize=9)
    ax.set(xlim=(-.3,1),ylim=(-.25,1.13)); ax.axis('off')


def surface_figure(surface,qc,out):
    imgs={side:render_surface(side,surface[side]) for side in SIDES}
    fig=plt.figure(figsize=(14,6),layout='constrained')
    grid=fig.add_gridspec(1,3,width_ratios=(1,1,.65))
    for col,side in enumerate(SIDES):
        ax=fig.add_subplot(grid[0,col]); ax.imshow(imgs[side]); ax.axis('off')
        row=qc[qc.hemisphere==side].iloc[0]
        ax.set_title(f'{"A" if side=="left" else "B"}  {side.capitalize()} posterior insula',fontsize=13)
        ax.text(.5,-.025,f'{int(row.ROI_columns)} ROI columns | 200 NORDIC-selected\n{100*row.fraction_of_ROI_vertices:.1f}% of ROI vertices',transform=ax.transAxes,ha='center',fontsize=10)
    ax=fig.add_subplot(grid[0,2]); schematic(ax); ax.set_title('C  Cortical-depth sampling',fontsize=12)
    fig.suptitle(f'Subject {SUBJECT}: full pIns ROI and NORDIC-selected Top200',fontsize=16)
    fig.legend(handles=[Patch(facecolor=ROI_COLOR,label='ROI outside Top200'),Patch(facecolor=TOP_COLOR,label='NORDIC-selected Top200'),Patch(facecolor='#AAAAAA',label='Inflated cortical anatomy')],loc='lower center',ncol=3,bbox_to_anchor=(.5,-.035),frameon=False)
    for ext in ('png','pdf'): fig.savefig(out/f'Figure_01_inflated_ROI_and_NORDIC_Top200.{ext}',dpi=DPI,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def tsnr_runs(values,lengths):
    # values: columns x 20 depths x time. Run-specific tSNR before any averaging.
    result=[]; start=0
    for length in lengths:
        segment=values[:,:,start:start+length]
        mu=segment.mean(axis=2); sd=segment.std(axis=2,ddof=1)
        ensure(np.isfinite(mu).all() and np.isfinite(sd).all() and np.all(sd>0),'Zero/non-finite temporal SD: cannot calculate valid tSNR')
        result.append(mu/sd); start+=length
    return np.stack(result,axis=2)


def calculate_tsnr(ids,selected,out):
    rows=[]; homogeneity=[]; mean_ts_rows=[]
    csvpath=out/'tSNR_individual_columns_by_depth_run.csv'
    write_header=True
    for condition,path in FILES.items():
        with h5py.File(path,'r') as f:
            for side in SIDES:
                ds=f[side]['layeractivation']; n=len(ids[side]); values_tsnr=np.empty((n,20,3)); total=np.zeros((20,1221)); selected_total=np.zeros_like(total)
                for start in range(0,n,BLOCK):
                    stop=min(start+BLOCK,n); block=np.asarray(ds[start:stop,:,:],dtype=float)
                    ensure(np.isfinite(block).all(),f'{condition}/{side}: NaN/Inf in sampling')
                    values_tsnr[start:stop]=tsnr_runs(block,EXPECTED_RUN_LENGTHS)
                    total+=block.sum(axis=0)
                    flag=np.isin(np.arange(start,stop),selected[side])
                    selected_total+=block[flag].sum(axis=0)
                all_mean=total/n; top_mean=selected_total/len(selected[side])
                for name,index,mean in (('all_ROI',np.arange(n),all_mean),('NORDIC_top200_matched',selected[side],top_mean)):
                    t=values_tsnr[index]
                    column_runmean=t.mean(axis=2)
                    for d in range(20):
                        z=column_runmean[:,d]
                        q25,med,q75=np.percentile(z,[25,50,75])
                        homogeneity.append({'subject':SUBJECT,'condition':condition,'hemisphere':side,'selection':name,'depth_index':d,'normalized_depth':d/19,'n_columns':len(index),'mean_column_tSNR_across_runs':z.mean(),'median_column_tSNR_across_runs':med,'q25':q25,'q75':q75,'sd_between_columns':z.std(ddof=1),'CV_between_columns':z.std(ddof=1)/z.mean()})
                        for run in range(3):
                            zrun=t[:,d,run]
                            segment=mean[d,run*407:(run+1)*407]
                            sdmean=segment.std(ddof=1)
                            ensure(sdmean>0,'ROI-mean time-course SD is zero')
                            rows.append({'subject':SUBJECT,'condition':condition,'hemisphere':side,'selection':name,'run':run+1,'depth_index':d,'normalized_depth':d/19,'n_columns':len(index),'mean_of_column_tSNR':zrun.mean(),'median_of_column_tSNR':np.median(zrun),'sd_between_columns_tSNR':zrun.std(ddof=1),'tSNR_of_ROI_mean_timecourse':segment.mean()/sdmean})
                    for run in range(3):
                        mean_ts_rows.append(pd.DataFrame({'subject':SUBJECT,'condition':condition,'hemisphere':side,'selection':name,'run':run+1,'timepoint_global':np.arange(run*407,(run+1)*407),**{f'depth_{d}_raw_mean':mean[d,run*407:(run+1)*407] for d in range(20)}}))
                cc,dd,rr=np.indices(values_tsnr.shape)
                detail=pd.DataFrame({'subject':SUBJECT,'condition':condition,'hemisphere':side,'local_column_index':cc.ravel(),'surface_vertex_id':np.broadcast_to(ids[side][:,None,None],values_tsnr.shape).ravel(),'depth_index':dd.ravel(),'run':rr.ravel()+1,'selected_NORDIC_top200':np.isin(cc.ravel(),selected[side]),'tSNR':values_tsnr.ravel()})
                detail.to_csv(csvpath,mode='w' if write_header else 'a',header=write_header,index=False); write_header=False
                log(f'{condition}/{side}: tSNR computed per column/depth/run')
    run_df=pd.DataFrame(rows); homogeneous_df=pd.DataFrame(homogeneity)
    subject_df=run_df.groupby(['subject','condition','hemisphere','selection','depth_index','normalized_depth','n_columns'],as_index=False)[['mean_of_column_tSNR','median_of_column_tSNR','tSNR_of_ROI_mean_timecourse']].mean()
    subject_df['n_runs']=3
    pd.concat(mean_ts_rows,ignore_index=True).to_csv(out/'ROImean_timeseries_by_depth.csv',index=False)
    return run_df,subject_df,homogeneous_df


def plot_tsnr(subject_df,homogeneous,out):
    for metric,tag,ylabel in (('mean_of_column_tSNR','columnwise','Mean column-wise tSNR (mean of 3 runs)'),('tSNR_of_ROI_mean_timecourse','ROImean_timecourse','tSNR of ROI-mean time course (different metric)')):
        fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
        for r,side in enumerate(SIDES):
            for c,selection in enumerate(('all_ROI','NORDIC_top200_matched')):
                ax=axes[r,c]
                for condition,label,color in (('without_NORDIC','Without NORDIC',PRE_COLOR),('with_NORDIC','With NORDIC',POST_COLOR)):
                    sub=subject_df[(subject_df.condition==condition)&(subject_df.hemisphere==side)&(subject_df.selection==selection)].sort_values('depth_index')
                    ax.plot(sub.normalized_depth,sub[metric],'-o',ms=3,color=color,label=label)
                ax.set(title=f'{side}: {selection}',xlabel='Depth (0=white; 1=pial)',ylabel=ylabel); ax.grid(alpha=.2); ax.legend()
        fig.suptitle(f'Subject {SUBJECT}: depth-resolved tSNR | raw time courses, no detrending',fontsize=14)
        for ext in ('png','pdf'): fig.savefig(out/f'Figure_02_tSNR_depth_{tag}.{ext}',dpi=DPI,bbox_inches='tight',facecolor='white')
        plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for ax,side in zip(axes,SIDES):
        for condition,label,color in (('without_NORDIC','Without NORDIC',PRE_COLOR),('with_NORDIC','With NORDIC',POST_COLOR)):
            sub=homogeneous[(homogeneous.condition==condition)&(homogeneous.hemisphere==side)&(homogeneous.selection=='NORDIC_top200_matched')].sort_values('depth_index')
            ax.plot(sub.normalized_depth,sub.CV_between_columns,'-o',ms=3,color=color,label=label)
        ax.set(title=f'{side}: selected Top200',xlabel='Depth (0=white; 1=pial)',ylabel='Between-column CV of run-mean tSNR'); ax.legend(); ax.grid(alpha=.2)
    fig.suptitle('Descriptive tSNR dispersion: lower CV does not prove functional homogeneity')
    fig.savefig(out/'Figure_03_Top200_tSNR_dispersion.png',dpi=DPI,bbox_inches='tight',facecolor='white'); plt.close(fig)


def main():
    for path in (WITH_H5,WITHOUT_H5): ensure(path.is_file(),f'Missing input H5: {path}')
    for h in ('lh','rh'):
        for name in ('inflated','sulc'): ensure((SURF_DIR/f'{h}.{name}').is_file(),f'Missing surface {h}.{name}')
    source=choose_excel()
    selection=pd.read_excel(source,sheet_name='NORDIC_top200_indices')
    ensure(set(selection.subject.astype(str))=={SUBJECT},'Selection Excel subject is not 7485')
    surface,ids,indices,selection_map,qc=prepare_inputs(selection)
    out=OUT_ROOT/time.strftime('%Y%m%d_%H%M%S'); out.mkdir(parents=True,exist_ok=False)
    (out/'selection_source.txt').write_text(str(source),encoding='utf-8')
    # Compute every table first, then create all figures.
    runs,subject_table,homogeneity=calculate_tsnr(ids,indices,out)
    runs.to_csv(out/'tSNR_depth_by_run.csv',index=False)
    subject_table.to_csv(out/'tSNR_depth_subject_summary_for_group.csv',index=False)
    homogeneity.to_csv(out/'tSNR_depth_column_dispersion.csv',index=False)
    selection_map.to_csv(out/'ROI_and_Top200_vertex_labels.csv',index=False)
    with pd.ExcelWriter(out/'Inflated_Top200_and_depth_tSNR.xlsx',engine='openpyxl') as writer:
        qc.to_excel(writer,sheet_name='surface_QC',index=False)
        selection.to_excel(writer,sheet_name='NORDIC_top200_indices',index=False)
        runs.to_excel(writer,sheet_name='tSNR_by_run',index=False)
        subject_table.to_excel(writer,sheet_name='tSNR_subject_mean',index=False)
        homogeneity.to_excel(writer,sheet_name='column_dispersion',index=False)
        pd.DataFrame({'note':['tSNR calculated as time mean / temporal SD (ddof=1) per column/depth/run before averaging','Raw sampled time courses: no detrending, no filtering, no task regression','Primary mean column tSNR differs from tSNR of averaged ROI time course; both exported separately','Within-subject three-run average; no inferential group analysis','Same NORDIC-derived columns in With/Without; full-data selection is not independent','Top200 vertex fraction is not cortical-area fraction','Inflated surfaces are visualization only; sampling occurred between white and pial surfaces','Connected components describe spatial clustering; tSNR CV describes dispersion, neither proves functional homogeneity','Surface triangles use median labels; orange point markers show exact selected vertices','No new GLM or fresh rWLS performed; existing indices reused']}).to_excel(writer,sheet_name='definitions',index=False)
    surface_figure(surface,qc,out)
    plot_tsnr(subject_table,homogeneity,out)
    for fname in ('Figure_01_inflated_ROI_and_NORDIC_Top200.png','Figure_01_inflated_ROI_and_NORDIC_Top200.pdf','Figure_02_tSNR_depth_columnwise.png','Inflated_Top200_and_depth_tSNR.xlsx'):
        ensure((out/fname).is_file() and (out/fname).stat().st_size>0,f'Missing output: {fname}')
    (out/'complete.json').write_text(json.dumps({'complete':True,'subject':SUBJECT,'selection_excel':str(source),'with_H5':str(WITH_H5),'without_H5':str(WITHOUT_H5),'tSNR_definition':'Raw time mean / sample SD per column/depth/run; average columns then 3 runs','surface_QC':qc.to_dict(orient='records')},indent=2),encoding='utf-8')
    log('DONE: inflated ROI/Top200 + depth tSNR + Excel/CSV created. Source files unchanged.')
    log(f'Output folder:\n{out}')


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        log(f'ERROR: {error}')
        log('No successful completion declared. Input files remain unchanged.')
        sys.exit(1)
