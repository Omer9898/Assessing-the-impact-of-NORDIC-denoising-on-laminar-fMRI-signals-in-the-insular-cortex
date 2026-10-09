"""7485: audit sampling, then NORDIC-selected matched Top200 and All-ROI GLMs.
Read-only source images/H5/SPM. No resampling/reconstruction/fresh rWLS.
Uses original validated 7485 Model (contrasts 8 and -4, fixed SPM matrices).
Selection is from NORDIC raw deep75; figures are effects, NOT corrected significance.
Requires numpy scipy nibabel h5py pandas openpyxl matplotlib.
"""
from pathlib import Path
import sys
import re
import json
import time
import numpy as np
import h5py
import nibabel as nib
import pandas as pd
from nibabel.freesurfer.io import read_geometry
from scipy.ndimage import map_coordinates
from scipy.io import loadmat
from scipy import sparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Edit these FULL paths only. This model is specifically validated for 7485.
WITHOUT_H5=Path(r'D:\NORDIC_Subject1_project\Laminarprofile\1. Sampling\Without\pIns_laminar_sampling_7485_without_NORDIC.h5')
WITH_H5=Path(r'D:\NORDIC_Subject1_project\Laminarprofile\1. Sampling\With\pIns_laminar_sampling_7485_with_NORDIC.h5')
SPM_FILE=Path(r'D:\Downloads\s7485\rwls_stats_compcor_UNsmoothed_hpf180\SPM.mat')
T1_FILE=Path(r'D:\NORDIC_Subject1_project\7485_extracted\UNI_MoCo_MPRAGEised_biascorrected.nii')
FS_SUBJECT=Path(r'D:\NORDIC_Subject1_project\7485_extracted\7485')
OUT_ROOT=Path(r'D:\NORDIC_Subject1_project\Laminarprofile\4. QC Comparison\2')
SUBJECT='7485'
SIDES=('left','right')
CONDITIONS=('without_NORDIC','with_NORDIC')
RUN_LENGTHS=np.array([407,407,407])
DEPTH=np.linspace(0,1,20)
TOP_N=200
BLOCK=32
FILES={'without_NORDIC':WITHOUT_H5,'with_NORDIC':WITH_H5}
# Reproduce current sampling from original images at first volume of each run.
# NO automatic coordinate correction or .partial deletion.
REPLAY_VOLUMES=(0,407,814)


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}",flush=True)

def get(obj, name):
    return obj[name] if isinstance(obj, dict) else getattr(obj, name)

def items(obj):
    return list(obj.ravel()) if isinstance(obj, np.ndarray) else [obj]

def dense(obj):
    return np.asarray(obj.toarray() if sparse.issparse(obj) else obj, dtype=float)

def text(obj):
    if isinstance(obj, bytes):
        return obj.decode()
    if isinstance(obj, np.ndarray) and obj.dtype.kind in 'US':
        return ''.join(obj.ravel().tolist())
    return str(obj)

class Model:

    def __init__(self):
        s = loadmat(SPM_FILE, struct_as_record=False, squeeze_me=True)['SPM']
        x = get(s, 'xX')
        self.X = dense(get(x, 'X'))
        self.W = get(x, 'W')
        self.P = dense(get(x, 'pKX'))
        self.B = dense(get(x, 'Bcov'))
        self.trRV = float(get(x, 'trRV'))
        self.filters = []
        self.names = [text(v) for v in items(get(x, 'name'))]
        n, p = self.X.shape
        if (n, p) != (1221, 57) or self.W.shape != (n, n) or self.P.shape != (p, n) or (self.B.shape != (p, p)):
            raise ValueError('SPM dimensions differ from validated subject-7485 report')
        if self.trRV <= 0:
            raise ValueError('Invalid trRV')
        covered = np.zeros(n, dtype=int)
        for k in items(get(x, 'K')):
            rows = np.asarray(get(k, 'row'), dtype=int).ravel() - 1
            x0 = dense(get(k, 'X0'))
            if x0.ndim == 1:
                x0 = x0[:, None]
            if x0.shape[0] != rows.size or np.any(rows < 0) or np.any(rows >= n):
                raise ValueError('Invalid SPM filter rows/X0')
            covered[rows] += 1
            self.filters.append((rows, x0))
        if not np.all(covered == 1):
            raise ValueError('Filter rows do not cover each scan exactly once')
        if [len(rows) for rows, _ in self.filters] != RUN_LENGTHS.tolist():
            raise ValueError('SPM run lengths do not match H5')
        self.A = self.filter(self.W @ self.X)
        actual = dense(get(get(x, 'xKXs'), 'X'))
        if not np.allclose(self.A, actual, rtol=1e-06, atol=1e-08):
            raise ValueError('Reconstructed K*W*X differs from saved SPM.xX.xKXs.X')
        self.C = {}
        contrasts = items(get(s, 'xCon'))
        for key, number, sign in (('pain', 8, 1), ('cognition', 4, -1)):
            con = contrasts[number - 1]
            if text(get(con, 'STAT')).strip() != 'T':
                raise ValueError(f'Contrast {number} is not T')
            raw = dense(get(con, 'c')).reshape(-1)
            expected = np.zeros(p)
            for i, name in enumerate(self.names):
                if 'pain_' not in name:
                    continue
                if key == 'pain':
                    expected[i] = 1 if '_high_pain*' in name else -1 if '_low_pain*' in name else 0
                else:
                    expected[i] = 1 if 'pain_high_cogn_' in name else -1 if 'pain_low_cogn_' in name else 0
            if not np.array_equal(raw, expected) or raw.size != p:
                raise ValueError(f'Contrast {number} weights differ from inspected report')
            self.C[key] = sign * raw
        self.anticipation = {}
        for level in ('high', 'low'):
            indices = [i for i, name in enumerate(self.names) if f'anticipation_{level}_cognition*bf(1)' in name]
            if len(indices) != 3:
                raise ValueError(f'Expected three anticipation_{level} regressors')
            self.anticipation[level] = indices
        self.scan_keys = []
        xy = get(s, 'xY')
        for v in items(get(xy, 'VY')):
            self.scan_keys.append(self.file_key(text(get(v, 'fname'))))
        log(f'SPM loaded: {n} scans x {p} regressors; cognition=-contrast4; pain=contrast8')

    @staticmethod
    def file_key(name):
        match = re.search('(\\d+)_Warped-to-Anat', name)
        if not match:
            raise ValueError(f'Cannot identify scan index: {name}')
        run = re.search('mag_POCS_r(\\d+)_', name)
        return (int(match.group(1)), int(run.group(1)) if run else None)

    def filter(self, y):
        out = np.asarray(y, dtype=float).copy()
        for rows, x0 in self.filters:
            out[rows] -= x0 @ (x0.T @ out[rows])
        return out

    def fit(self, y):
        y = np.asarray(y, dtype=float)
        if y.ndim != 2 or y.shape[0] != self.X.shape[0] or (not np.isfinite(y).all()):
            raise ValueError('Invalid/non-finite GLM input')
        ky = self.filter(self.W @ y)
        beta = self.P @ ky
        residual = ky - self.A @ beta
        resms = (residual * residual).sum(axis=0) / self.trRV
        if np.any(resms <= 0) or not np.isfinite(resms).all():
            raise ValueError('Invalid residual variance')
        stats = {}
        for name, c in self.C.items():
            effect = c @ beta
            variance_factor = float(c @ self.B @ c)
            if variance_factor <= 0:
                raise ValueError('Invalid contrast covariance')
            stats[name] = {'effect_spm_sum': effect, 'effect_mean': effect / 6.0, 't': effect / np.sqrt(resms * variance_factor)}
        return (beta, resms, stats)


def ensure(ok,message):
    if not ok: raise ValueError(message)


def coords_from_surfaces(ref):
    white=np.vstack([read_geometry(str(FS_SUBJECT/'surf'/f'{h}.white'))[0] for h in ('lh','rh')])
    pial=np.vstack([read_geometry(str(FS_SUBJECT/'surf'/f'{h}.pial'))[0] for h in ('lh','rh')])
    ensure(white.shape==pial.shape,'White/pial dimensions differ')
    w=nib.affines.apply_affine(np.linalg.inv(ref.affine),white)
    p=nib.affines.apply_affine(np.linalg.inv(ref.affine),pial)
    orig=FS_SUBJECT/'mri'/'orig.mgz'
    spatial_note='Direct RAS assumption; anatomical visual QC remains required.'
    if orig.is_file():
        fs=nib.load(str(orig))
        transform=fs.affine@np.linalg.inv(fs.header.get_vox2ras_tkr())
        scanwhite=nib.affines.apply_affine(transform,white)
        delta=float(np.max(np.abs(nib.affines.apply_affine(np.linalg.inv(ref.affine),scanwhite)-w)))
        ensure(delta<=0.01,f'Direct mapping and tkRAS conversion differ by {delta:.4f} voxels. Stop: review anatomical transform before GLM.')
        spatial_note=f'tkRAS/direct difference <=0.01 voxel (max {delta:.6g}). Assumes orig.mgz/T1 share world space; NOT independent registration certification.'
    return w,p,spatial_note


def audit_sampling(model,output):
    ref=nib.load(str(T1_FILE))
    ensure(len(ref.shape)==3,'T1 must be 3D')
    white,pial,spatial_note=coords_from_surfaces(ref)
    report={'subject':SUBJECT,'scope':'Sampling reproduction/internal consistency ONLY; anatomical correctness needs visual inspection. No fresh rWLS.','spatial_note':spatial_note,'conditions':{},'warnings':[],'qc_pass':False}
    summaries=[]; ids_by_condition={}; means={}; deep={}; coords={}
    try:
        for condition,path in FILES.items():
            log(f'Audit {condition}')
            with h5py.File(path,'r') as f:
                ensure(text(f.attrs.get('subject',''))==SUBJECT,f'{condition}: subject metadata is not 7485')
                ensure(text(f.attrs.get('condition',''))==condition,f'{condition}: incorrect condition metadata')
                if 'complete' in f.attrs: ensure(bool(f.attrs['complete']),f'{condition}: incomplete H5')
                if 'written_volumes' in f: ensure(f['written_volumes'][:].all(),f'{condition}: unwritten volumes')
                ensure(np.array_equal(f['run_lengths'][:],RUN_LENGTHS),'Run lengths must be 407/407/407')
                names=[text(v) for v in f['volume_files'][:]]
                ensure(len(names)==1221,'Expected 1221 source volumes')
                for i,name in enumerate(names):
                    key=model.file_key(name)
                    ensure(key[0]==model.scan_keys[i][0] and (key[1] is None or key[1]==i//407+1),f'{condition}: source/SPM scan ordering mismatch at {i}')
                    ensure(model.scan_keys[i][1] is None or model.scan_keys[i][1]==i//407+1,'SPM run ordering mismatch')
                entry={'source':str(path),'first_source_volumes':[names[i] for i in REPLAY_VOLUMES],'hemispheres':{},'replay':{}}
                report['conditions'][condition]=entry
                ids_by_condition[condition]={}; means[condition]={}; deep[condition]={}; coords[condition]={}
                for side in SIDES:
                    g=f[side]; ds=g['layeractivation']; ids=g['surface_vertex_indices'][:].astype(int)
                    ensure(ds.shape==(len(ids),20,1221) and len(ids)>0,f'{condition}/{side}: invalid dimensions')
                    ensure(len(np.unique(ids))==len(ids) and np.all((ids>=0)&(ids<len(white))),'Invalid/duplicate surface IDs')
                    ids_by_condition[condition][side]=ids
                    xyz=white[ids,None,:]+DEPTH[None,:,None]*(pial-white)[ids,None,:]
                    coords[condition][side]=xyz
                    ensure(np.all((xyz>=0)&(xyz<=np.array(ref.shape)-1)),'Sampling coordinates outside T1')
                    if 'sampling_coordinates_vox' in g:
                        ensure(np.allclose(g['sampling_coordinates_vox'][:],xyz,rtol=0,atol=1e-5),'Saved coordinates disagree with direct zero-based mapping')
                    total=np.zeros((20,1221)); ydeep=np.empty((len(ids),1221)); constant=0; identical_adjacent=0; nonzero=np.zeros(1221,bool)
                    for start in range(0,len(ids),BLOCK):
                        stop=min(start+BLOCK,len(ids))
                        values=np.asarray(ds[start:stop,:,:],dtype=float)
                        ensure(np.isfinite(values).all(),f'{condition}/{side}: NaN/Inf in columns {start}:{stop}')
                        nonzero|=np.any(values!=0,axis=(0,1))
                        constant+=int(np.count_nonzero(np.ptp(values,axis=2)==0))
                        identical_adjacent+=int(np.count_nonzero(np.all(values[:,1:,:]==values[:,:-1,:],axis=2)))
                        ydeep[start:stop]=values[:,:15,:].mean(axis=1)
                        total+=values.sum(axis=0)
                        if 'columnspecificts' in g:
                            ensure(np.allclose(g['columnspecificts'][start:stop,:],ydeep[start:stop],rtol=1e-5,atol=1e-4),'Stored deep75 average mismatch')
                    ensure(nonzero.all(),f'{condition}/{side}: entire timepoints are zero')
                    ensure(constant==0,f'{condition}/{side}: {constant} constant column/depth time courses; inspect sampling')
                    if identical_adjacent: report['warnings'].append(f'{condition}/{side}: {identical_adjacent} EXACTLY identical adjacent-depth time courses; inspect spatial sampling')
                    means[condition][side]=total/len(ids); deep[condition][side]=ydeep
                    if 'all_columns_average' in g: ensure(np.allclose(g['all_columns_average'][:],means[condition][side],rtol=1e-5,atol=1e-4),'Stored all-column mean mismatch')
                    entry['hemispheres'][side]={'n_columns':len(ids),'constant_timecourses':constant,'exact_identical_adjacent_depth_pairs':identical_adjacent,'nonzero_timepoints':int(nonzero.sum())}
                    for run,(a,b) in enumerate(zip((0,407,814),(407,814,1221)),1):
                        m=means[condition][side][:,a:b]
                        for d in range(20): summaries.append({'subject':SUBJECT,'condition':condition,'hemisphere':side,'run':run,'depth':d,'raw_mean':float(m[d].mean()),'sd_time_ROImean':float(m[d].std(ddof=1))})
                for vi in REPLAY_VOLUMES:
                    imagepath=Path(names[vi]); ensure(imagepath.is_file(),f'Missing source image for replay: {imagepath}')
                    image=nib.load(str(imagepath))
                    ensure(image.shape==ref.shape and np.allclose(image.affine,ref.affine,rtol=0,atol=1e-5),f'Source image/T1 grid mismatch: {imagepath}')
                    data=image.get_fdata(dtype=np.float32)
                    for side in SIDES:
                        expected=f[side]['layeractivation'][:,:,vi]
                        xyz=coords[condition][side]
                        actual=map_coordinates(data,xyz.reshape(-1,3).T,order=int(f.attrs.get('interpolation_order',3)),mode='constant',cval=np.nan,prefilter=True).reshape(expected.shape)
                        err=float(np.max(np.abs(actual-expected)))
                        entry['replay'][f'{side}_volume{vi}']={'max_abs_error':err,'pass':bool(np.allclose(actual,expected,rtol=1e-5,atol=1e-3))}
                        ensure(entry['replay'][f'{side}_volume{vi}']['pass'],f'{condition}/{side}: source replay failed at {vi}. Possible old H5/coordinate mismatch; DO NOT proceed to GLM.')
                    del data
        for side in SIDES:
            ensure(np.array_equal(ids_by_condition[CONDITIONS[0]][side],ids_by_condition[CONDITIONS[1]][side]),'Pre/post vertex identity/order differs; explicit ID remapping required')
            ensure(not np.array_equal(deep[CONDITIONS[0]][side],deep[CONDITIONS[1]][side]),'Pre/post deep75 arrays exactly identical: check duplicate/wrong inputs')
        report['qc_pass']=True
        log('QC PASS: reproduction and consistency checks passed; anatomy not independently certified')
    except Exception as error:
        report['error']=str(error)
        raise
    finally:
        (output/'sampling_QC_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        pd.DataFrame(summaries).to_csv(output/'sampling_raw_means_by_run_depth.csv',index=False)
    return ids_by_condition,means,deep,report,pd.DataFrame(summaries)


def selected_mean(path,side,indices):
    with h5py.File(path,'r') as f:
        ds=f[side]['layeractivation']; total=np.zeros((20,1221)); ordered=np.sort(indices)
        for start in range(0,len(ordered),BLOCK):
            total+=np.asarray(ds[ordered[start:start+BLOCK],:,:],dtype=float).sum(axis=0)
    return total/len(indices)


def scale_per_run(mean):
    y=mean.copy(); baseline=np.empty((20,3))
    for run,(a,b) in enumerate(zip((0,407,814),(407,814,1221))):
        mu=y[:,a:b].mean(axis=1)
        ensure(np.all(mu>0),'Non-positive mean baseline prevents percent-of-mean scaling')
        baseline[:,run]=mu
        y[:,a:b]=100*(y[:,a:b]/mu[:,None]-1)
    return y,baseline


def figures(frame,qc,output):
    import matplotlib.pyplot as plt
    depth=DEPTH
    for selection in ('all_ROI','NORDIC_top200_matched'):
        for unit in ('raw','percent_of_run_mean'):
            fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
            for r,side in enumerate(SIDES):
                for col,key in enumerate(('pain','cognition')):
                    ax=axes[r,col]
                    for condition,label,color in (('without_NORDIC','Without NORDIC','#0072B2'),('with_NORDIC','With NORDIC','#D55E00')):
                        sub=frame[(frame.selection==selection)&(frame.units==unit)&(frame.hemisphere==side)&(frame.condition==condition)&(frame.contrast==key)].sort_values('depth_index')
                        ax.plot(depth,sub.effect_mean,'-o',ms=3,color=color,label=label)
                    ax.axhline(0,color='0.5',lw=.6); ax.set(title=f'{side}: {key}',xlabel='Depth (0=white; 1=pial)',ylabel='Contrast estimate: raw units' if unit=='raw' else 'Contrast estimate: % of own run/depth mean')
                    ax.legend(fontsize=9)
            fig.suptitle(f'{selection} | {unit} | fixed SPM matrices | descriptive')
            fig.savefig(output/f'profiles_{selection}_{unit}.png',dpi=180); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    for ax,side in zip(axes,SIDES):
        for condition in CONDITIONS:
            s=qc[(qc.hemisphere==side)&(qc.condition==condition)].groupby('depth').raw_mean.mean()
            ax.plot(DEPTH,s.values,'-o',ms=3,label=condition)
        ax.set(title=side,xlabel='Depth (0=white; 1=pial)',ylabel='All-ROI raw mean signal'); ax.legend()
    fig.suptitle('Raw intensity scale check: descriptive, not GLM')
    fig.savefig(output/'QC_raw_signal_depth.png',dpi=180); plt.close(fig)


def main():
    for path in (WITHOUT_H5,WITH_H5,SPM_FILE,T1_FILE): ensure(path.is_file(),f'Missing input: {path}')
    ensure(WITHOUT_H5.resolve()!=WITH_H5.resolve(),'Both inputs point to the same H5')
    output=OUT_ROOT/time.strftime('%Y%m%d_%H%M%S')
    output.mkdir(parents=True,exist_ok=False)
    log(f'New output folder: {output}')
    log('Subject 7485 ONLY; fixed SPM W/pKX/Bcov/K/trRV. Not fresh rWLS or group inference.')
    model=Model()
    ids,allmeans,deep,report,qc=audit_sampling(model,output)
    selections={}; tables=[]
    for side in SIDES:
        beta,resms,stats=model.fit(deep['with_NORDIC'][side].T)
        tp=stats['pain']['t']; tc=stats['cognition']['t']; conjunction=np.minimum(tp,tc)
        valid=np.flatnonzero(np.isfinite(conjunction)&(conjunction>0))
        ensure(len(valid)>=TOP_N,f'NORDIC/{side}: only {len(valid)} positive columns. No automatic relaxation.')
        selected=valid[np.argsort(-conjunction[valid],kind='stable')[:TOP_N]]
        selections[side]=selected
        flag=np.zeros(len(tp),bool); flag[selected]=True
        table=pd.DataFrame({'subject':SUBJECT,'hemisphere':side,'local_index_zero_based':np.arange(len(tp)),'surface_vertex_id':ids['with_NORDIC'][side],'t_pain_NORDIC':tp,'t_cognition_NORDIC':tc,'conjunction_min_t_NORDIC':conjunction,'selected_NORDIC_top200':flag})
        table.to_csv(output/f'{side}_NORDIC_selection_all_columns.csv',index=False)
        ranked=table.iloc[selected].copy(); ranked.insert(0,'rank',np.arange(1,TOP_N+1)); ranked.to_csv(output/f'{side}_NORDIC_TOP200_indices.csv',index=False)
        tables.append(ranked)
        log(f'{side}: selected {TOP_N} from {len(valid)} positive NORDIC columns; SAME indices used in both conditions')
    rows=[]
    target=output/'NORDIC_selected_and_AllROI_profiles.h5'
    partial=target.with_suffix('.h5.partial')
    with h5py.File(partial,'x') as dst:
        dst.attrs['complete']=False; dst.attrs['subject']=SUBJECT
        dst.attrs['selection_source']='NORDIC raw deep75 GLM minimum positive T; full-data descriptive selection'
        dst.attrs['estimation']='Frozen SPM matrices; NOT fresh rWLS'
        dst.attrs['limitations']='Selection and evaluation share data; no independent claim of NORDIC superiority. Percent-of-run/depth-mean supplemental scaling, not guaranteed calibrated conventional PSC.'
        dst.create_dataset('regressor_names',data=np.array(model.names,dtype=h5py.string_dtype()))
        for key,c in model.C.items(): dst.create_dataset(f'contrast_{key}',data=c)
        for condition in CONDITIONS:
            for side in SIDES:
                for selection in ('all_ROI','NORDIC_top200_matched'):
                    indices=np.arange(len(ids[condition][side])) if selection=='all_ROI' else selections[side]
                    mean=allmeans[condition][side] if selection=='all_ROI' else selected_mean(FILES[condition],side,indices)
                    root=dst.require_group(f'{condition}/{side}/{selection}')
                    root.create_dataset('selected_local_indices',data=indices)
                    root.create_dataset('selected_vertex_ids',data=ids[condition][side][indices])
                    root.create_dataset('raw_mean_timeseries',data=mean,compression='gzip')
                    normalized,baselines=scale_per_run(mean)
                    root.create_dataset('run_depth_baselines',data=baselines)
                    for unit,y in (('raw',mean),('percent_of_run_mean',normalized)):
                        beta,resms,stats=model.fit(y.T)
                        g=root.create_group(unit)
                        g.create_dataset('beta',data=beta.T); g.create_dataset('resms',data=resms)
                        for key,s in stats.items():
                            g.create_dataset(f'effect_mean_{key}',data=s['effect_mean']); g.create_dataset(f't_{key}',data=s['t'])
                            for d in range(20): rows.append({'subject':SUBJECT,'condition':condition,'hemisphere':side,'selection':selection,'selection_source':'all_ROI' if selection=='all_ROI' else 'with_NORDIC','units':unit,'contrast':key,'depth_index':d,'normalized_depth':DEPTH[d],'n_columns':len(indices),'effect_mean':s['effect_mean'][d],'t_value':s['t'][d],'resms':resms[d]})
        dst.attrs['complete']=True
    partial.replace(target)
    frame=pd.DataFrame(rows)
    frame.to_csv(output/'profiles_long_for_future_group_analysis.csv',index=False)
    with pd.ExcelWriter(output/'QC_and_NORDIC_matched_profiles.xlsx',engine='openpyxl') as writer:
        qc.to_excel(writer,sheet_name='raw_means_QC',index=False)
        pd.concat(tables,ignore_index=True).to_excel(writer,sheet_name='NORDIC_top200_indices',index=False)
        frame.to_excel(writer,sheet_name='profiles',index=False)
        pd.DataFrame({'note':['Subject7485; no group analysis performed','Raw NORDIC deep75 determines Top200; same indices for both conditions','All-ROI sensitivity analysis included','Fixed SPM weights; not fresh rWLS','Percent-of-own-run/depth-mean scaled effects are supplemental, not guaranteed conventional PSC','Selection bias remains: full data used for selection and estimation','Sampling reproduction does not certify anatomical registration','No T threshold significance/FDR/FWE claim']}).to_excel(writer,sheet_name='limitations',index=False)
    figures(frame,qc,output)
    (output/'analysis_complete.json').write_text(json.dumps({'complete':True,'subject':SUBJECT,'inputs':{k:str(v) for k,v in FILES.items()},'SPM':str(SPM_FILE),'qc_pass':report['qc_pass'],'warnings':report['warnings'],'selection_source':'with_NORDIC','group_analysis_performed':False},indent=2),encoding='utf-8')
    log('DONE: QC, NORDIC-matched Top200, All-ROI, raw/scaled profiles, Excel, CSV, H5 and PNG saved')
    log(f'Results: {output}')
    log('Use sampling_QC_report.json + raw signal QC first; no automatic conclusion about NORDIC superiority.')


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        log(f'ERROR: {error}')
        log('Not complete. Source files unchanged. Inspect new report/.partial output; do not treat it as final.')
        sys.exit(1)
