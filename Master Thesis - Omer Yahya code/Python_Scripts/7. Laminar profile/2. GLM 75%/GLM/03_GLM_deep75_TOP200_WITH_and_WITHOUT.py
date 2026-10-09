"""Subject 7485: deep75 column GLM, positive conjunction Top200, layer GLM.
Uses the existing SPM design/filter/estimation matrices with FROZEN W.
NOT a fresh rWLS estimation and NOT certified numerically identical to the
supervisor's complete pipeline. Raw sampled intensity is used (no PSC scaling).
Requirements: numpy scipy h5py matplotlib
Outputs: H5, CSV, PNG, audit TXT. Sampling/SPM inputs remain unchanged.
"""
from pathlib import Path
import sys
import re
import csv
import json
import time
import numpy as np
import h5py
from scipy.io import loadmat
from scipy import sparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path(r'D:\NORDIC_Subject1_project\Laminarprofile')
SPM_FILE = Path(r'D:\Downloads\s7485\rwls_stats_compcor_UNsmoothed_hpf180\SPM.mat')
OUT = BASE / "2. GLM 75%" / "GLM" / "GLM_results_fixed_SPM_weights"
CONDITIONS = ('without_NORDIC','with_NORDIC')
SIDES = ('left','right')
TOP_N = 200
OVERWRITE = True
DEPTH = np.linspace(0,1,20)
RUN_LENGTHS = np.array([407,407,407])
FILES = {
    c: {
        'deep': BASE / '2. GLM 75%' / 'deep75_timeseries' / f'pIns_laminar_sampling_7485_{c}_deep75_column_timeseries.h5',
        'layers': BASE / '1. Sampling' / ('Without' if c == 'without_NORDIC' else 'With') / f'pIns_laminar_sampling_7485_{c}.h5'
    } for c in CONDITIONS
}


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}",flush=True)


def get(obj,name):
    return obj[name] if isinstance(obj,dict) else getattr(obj,name)


def items(obj):
    return list(obj.ravel()) if isinstance(obj,np.ndarray) else [obj]


def dense(obj):
    return np.asarray(obj.toarray() if sparse.issparse(obj) else obj,dtype=float)


def text(obj):
    if isinstance(obj,bytes):
        return obj.decode()
    if isinstance(obj,np.ndarray) and obj.dtype.kind in 'US':
        return ''.join(obj.ravel().tolist())
    return str(obj)


class Model:
    def __init__(self):
        s = loadmat(SPM_FILE,struct_as_record=False,squeeze_me=True)['SPM']
        x = get(s,'xX')
        self.X = dense(get(x,'X'))
        self.W = get(x,'W')
        self.P = dense(get(x,'pKX'))
        self.B = dense(get(x,'Bcov'))
        self.trRV = float(get(x,'trRV'))
        self.filters = []
        self.names = [text(v) for v in items(get(x,'name'))]
        n,p = self.X.shape
        if (n,p) != (1221,57) or self.W.shape != (n,n) or self.P.shape != (p,n) or self.B.shape != (p,p):
            raise ValueError('SPM dimensions differ from validated subject-7485 report')
        if self.trRV <= 0:
            raise ValueError('Invalid trRV')
        covered = np.zeros(n,dtype=int)
        for k in items(get(x,'K')):
            rows = np.asarray(get(k,'row'),dtype=int).ravel()-1
            x0 = dense(get(k,'X0'))
            if x0.ndim == 1:
                x0 = x0[:,None]
            if x0.shape[0] != rows.size or np.any(rows < 0) or np.any(rows >= n):
                raise ValueError('Invalid SPM filter rows/X0')
            covered[rows] += 1
            self.filters.append((rows,x0))
        if not np.all(covered == 1):
            raise ValueError('Filter rows do not cover each scan exactly once')
        if [len(rows) for rows,_ in self.filters] != RUN_LENGTHS.tolist():
            raise ValueError('SPM run lengths do not match H5')
        self.A = self.filter(self.W @ self.X)
        actual = dense(get(get(x,'xKXs'),'X'))
        if not np.allclose(self.A,actual,rtol=1e-6,atol=1e-8):
            raise ValueError('Reconstructed K*W*X differs from saved SPM.xX.xKXs.X')
        self.C = {}
        contrasts = items(get(s,'xCon'))
        for key,number,sign in (('pain',8,1),('cognition',4,-1)):
            con = contrasts[number-1]
            if text(get(con,'STAT')).strip() != 'T':
                raise ValueError(f'Contrast {number} is not T')
            raw = dense(get(con,'c')).reshape(-1)
            expected = np.zeros(p)
            for i,name in enumerate(self.names):
                if 'pain_' not in name:
                    continue
                if key == 'pain':
                    expected[i] = 1 if '_high_pain*' in name else -1 if '_low_pain*' in name else 0
                else:
                    expected[i] = 1 if 'pain_high_cogn_' in name else -1 if 'pain_low_cogn_' in name else 0
            if not np.array_equal(raw,expected) or raw.size != p:
                raise ValueError(f'Contrast {number} weights differ from inspected report')
            self.C[key] = sign*raw
        # Descriptive anticipation condition betas; NOT invented task contrasts.
        self.anticipation = {}
        for level in ('high','low'):
            indices = [i for i,name in enumerate(self.names) if f'anticipation_{level}_cognition*bf(1)' in name]
            if len(indices) != 3:
                raise ValueError(f'Expected three anticipation_{level} regressors')
            self.anticipation[level] = indices
        self.scan_keys = []
        xy = get(s,'xY')
        for v in items(get(xy,'VY')):
            self.scan_keys.append(self.file_key(text(get(v,'fname'))))
        log(f'SPM loaded: {n} scans x {p} regressors; cognition=-contrast4; pain=contrast8')

    @staticmethod
    def file_key(name):
        match = re.search(r'(\d+)_Warped-to-Anat',name)
        if not match:
            raise ValueError(f'Cannot identify scan index: {name}')
        run = re.search(r'mag_POCS_r(\d+)_',name)
        return int(match.group(1)),int(run.group(1)) if run else None

    def filter(self,y):
        out = np.asarray(y,dtype=float).copy()
        for rows,x0 in self.filters:
            out[rows] -= x0 @ (x0.T @ out[rows])
        return out

    def fit(self,y):
        y = np.asarray(y,dtype=float)
        if y.ndim != 2 or y.shape[0] != self.X.shape[0] or not np.isfinite(y).all():
            raise ValueError('Invalid/non-finite GLM input')
        ky = self.filter(self.W @ y)
        beta = self.P @ ky
        residual = ky-self.A @ beta
        resms = (residual*residual).sum(axis=0)/self.trRV
        if np.any(resms <= 0) or not np.isfinite(resms).all():
            raise ValueError('Invalid residual variance')
        stats = {}
        for name,c in self.C.items():
            effect = c @ beta
            variance_factor = float(c @ self.B @ c)
            if variance_factor <= 0:
                raise ValueError('Invalid contrast covariance')
            stats[name] = {'effect_spm_sum':effect,'effect_mean':effect/6.,'t':effect/np.sqrt(resms*variance_factor)}
        return beta,resms,stats


def check_h5(model):
    reference = {}
    for condition in CONDITIONS:
        for path in FILES[condition].values():
            if not path.is_file():
                raise FileNotFoundError(path)
        with h5py.File(FILES[condition]['deep'],'r') as deep, h5py.File(FILES[condition]['layers'],'r') as layers:
            for handle in (deep,layers):
                if not np.array_equal(handle['run_lengths'][:],RUN_LENGTHS):
                    raise ValueError('Run lengths differ')
            if 'volume_files' not in layers:
                raise ValueError('Missing sampling volume list for scan-order verification')
            names = [v.decode() if isinstance(v,bytes) else str(v) for v in layers['volume_files'][:]]
            if len(names) != len(model.scan_keys):
                raise ValueError('SPM/sampling scan counts differ')
            for i,name in enumerate(names):
                key = model.file_key(name)
                if key[0] != model.scan_keys[i][0] or (key[1] is not None and key[1] != i//407+1):
                    raise ValueError(f'Sampling/SPM volume-order mismatch at row {i}')
                if model.scan_keys[i][1] is not None and model.scan_keys[i][1] != i//407+1:
                    raise ValueError('Unexpected SPM run order')
            for side in SIDES:
                ids = deep[side]['surface_vertex_indices'][:]
                if not np.array_equal(ids,layers[side]['surface_vertex_indices'][:]):
                    raise ValueError(f'{side}: deep/layer vertex order mismatch')
                if condition == CONDITIONS[0]:
                    reference[side] = ids
                elif not np.array_equal(ids,reference[side]):
                    raise ValueError(f'{side}: pre/post vertex order mismatch')
                if deep[side]['deep75_column_timeseries'].shape != (ids.size,1221) or layers[side]['layeractivation'].shape != (ids.size,20,1221):
                    raise ValueError(f'{side}: inconsistent sampling dimensions')
                for row in np.unique([0,len(ids)//2,len(ids)-1]):
                    expected = np.asarray(layers[side]['layeractivation'][row,:15,:],dtype=float).mean(axis=0)
                    if not np.allclose(expected,deep[side]['deep75_column_timeseries'][row,:],rtol=1e-6,atol=1e-5):
                        raise ValueError(f'{side}: deep75 values do not match original layers')
    log('H5 identities, dimensions, sampled deep75 values and scan indices validated')


def save(g,name,data):
    g.create_dataset(name,data=np.asarray(data),compression='gzip',compression_opts=4)


def csv_write(path,header,rows):
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def layer_mean(ds,indices):
    total = np.zeros((20,1221))
    ordered = np.sort(indices)
    for start in range(0,len(ordered),32):
        block = np.asarray(ds[ordered[start:start+32],:,:],dtype=float)
        if not np.isfinite(block).all():
            raise ValueError('Non-finite selected layer data')
        total += block.sum(axis=0)
    return total/len(ordered)


def write_layer(g,mean,model):
    beta,resms,stats = model.fit(mean.T)
    save(g,'layer_mean_timeseries',mean)
    save(g,'layer_depth',DEPTH)
    save(g,'layer_beta',beta.T)
    save(g,'layer_resms',resms)
    for key,value in stats.items():
        save(g,f'layer_t_{key}',value['t'])
        save(g,f'layer_effect_{key}',value['effect_mean'])
        save(g,f'layer_effect_{key}_spm_sum',value['effect_spm_sum'])
    for level,idx in model.anticipation.items():
        save(g,f'layer_anticipation_{level}_beta_mean',beta[idx,:].mean(axis=0))
    rows = zip(DEPTH,stats['pain']['effect_mean'],stats['cognition']['effect_mean'],stats['pain']['t'],stats['cognition']['t'])
    return list(rows)


def main():
    log('WARNING: frozen SPM W is reused for BOTH conditions; fresh rWLS is NOT estimated.')
    model = Model()
    check_h5(model)
    OUT.mkdir(parents=True,exist_ok=True)
    outputs = {c:OUT/f'pIns_GLM_{c}_deep75_top200_layerwise.h5' for c in CONDITIONS}
    for path in outputs.values():
        if path.exists() and not OVERWRITE:
            raise FileExistsError(f'Output exists: {path}; set OVERWRITE=True only for intentional replacement')
    selected_by_condition = {}
    baseline = {}
    for condition in CONDITIONS:
        chosen = {}
        temporary = outputs[condition].with_suffix('.h5.partial')
        if temporary.exists():
            raise FileExistsError(f'Incomplete file exists: {temporary}')
        try:
            with h5py.File(FILES[condition]['deep'],'r') as deep, h5py.File(FILES[condition]['layers'],'r') as layers, h5py.File(temporary,'x') as dst:
                dst.attrs['condition'] = condition
                dst.attrs['spm_source'] = str(SPM_FILE)
                dst.attrs['estimation'] = 'Frozen SPM W, pKX, Bcov, K and trRV; NOT fresh rWLS'
                dst.attrs['effect_scaling'] = 'SPM contrast sum / 6: averaged over 2 factor levels x 3 runs; t unchanged'
                dst.attrs['cognition_direction'] = 'Low cognition - High cognition; negative of SPM contrast 4'
                dst.attrs['pain_direction'] = 'High pain - Low pain; SPM contrast 8'
                dst.attrs['anticipation'] = 'Descriptive condition betas averaged over runs; not used in column selection'
                dst.attrs['complete'] = False
                save(dst,'run_lengths',RUN_LENGTHS)
                save(dst,'pain_contrast_spm_sum',model.C['pain'])
                save(dst,'cognition_contrast_spm_sum',model.C['cognition'])
                dst.create_dataset('regressor_names',data=np.array(model.names,dtype=h5py.string_dtype('utf-8')))
                for side in SIDES:
                    log(f'{condition}/{side}: column-wise deep75 GLM')
                    ids = deep[side]['surface_vertex_indices'][:]
                    y = np.asarray(deep[side]['deep75_column_timeseries'][:],dtype=float).T
                    beta,resms,stats = model.fit(y)
                    conjunction = np.minimum(stats['pain']['t'],stats['cognition']['t'])
                    valid = np.flatnonzero(np.isfinite(conjunction)&(conjunction>0))
                    if valid.size < TOP_N:
                        raise ValueError(f'{condition}/{side}: only {valid.size} positive conjunction columns; cannot select exactly 200. No automatic threshold relaxation.')
                    selection = valid[np.argsort(-conjunction[valid],kind='stable')[:TOP_N]]
                    chosen[side] = selection
                    if condition == CONDITIONS[0]:
                        baseline[side] = selection.copy()
                    g = dst.create_group(side)
                    save(g,'surface_vertex_indices',ids)
                    save(g,'selected_column_indices_zero_based',selection)
                    save(g,'selected_surface_vertex_indices',ids[selection])
                    save(g,'column_beta',beta.T)
                    save(g,'column_resms',resms)
                    save(g,'column_t_conjunction',conjunction)
                    for key,value in stats.items():
                        save(g,f'column_t_{key}',value['t'])
                        save(g,f'column_effect_{key}',value['effect_mean'])
                    selected_flag = np.zeros(len(ids),dtype=int)
                    selected_flag[selection]=1
                    csv_write(OUT/f'{condition}_{side}_columns.csv',['local_index_zero_based','surface_vertex_id','t_pain','t_cognition_low_minus_high','conjunction_min_t','selected_top200'],zip(range(len(ids)),ids,stats['pain']['t'],stats['cognition']['t'],conjunction,selected_flag))
                    log(f'{condition}/{side}: selected 200 of {valid.size} positive columns; layer-wise GLM')
                    mean = layer_mean(layers[side]['layeractivation'],selection)
                    profile = write_layer(g,mean,model)
                    csv_write(OUT/f'{condition}_{side}_profile.csv',['depth_white0_pial1','pain_effect_mean','cognition_effect_mean','t_pain','t_cognition'],profile)
                    # Secondary NORDIC comparison: exactly the same baseline columns.
                    matched = g.create_group('matched_without_selection')
                    save(matched,'selected_column_indices_zero_based',baseline[side])
                    save(matched,'selected_surface_vertex_indices',ids[baseline[side]])
                    matched_mean = mean if np.array_equal(selection,baseline[side]) else layer_mean(layers[side]['layeractivation'],baseline[side])
                    profile = write_layer(matched,matched_mean,model)
                    csv_write(OUT/f'{condition}_{side}_matched_profile.csv',['depth_white0_pial1','pain_effect_mean','cognition_effect_mean','t_pain','t_cognition'],profile)
                dst.attrs['complete'] = True
            temporary.replace(outputs[condition])
        except Exception:
            if temporary.exists():
                temporary.unlink()
            raise
        selected_by_condition[condition]=chosen
        log(f'Saved: {outputs[condition]}')
    audit = {'method':'Frozen existing SPM weights; not fresh rWLS','inputs':{c:{k:str(v) for k,v in FILES[c].items()} for c in CONDITIONS},'spm':str(SPM_FILE),'selection_overlap':{}}
    for side in SIDES:
        overlap = len(np.intersect1d(selected_by_condition[CONDITIONS[0]][side],selected_by_condition[CONDITIONS[1]][side]))
        audit['selection_overlap'][side]={'overlap':overlap,'dice':overlap/TOP_N}
    (OUT/'method_and_overlap.txt').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    figures(outputs)
    log(f'DONE: H5, CSV and PNG files saved in {OUT}')


def figures(outputs):
    with h5py.File(outputs[CONDITIONS[0]],'r') as pre,h5py.File(outputs[CONDITIONS[1]],'r') as post:
        for matched in (False,True):
            fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
            for row,side in enumerate(SIDES):
                for col,key in enumerate(('pain','cognition')):
                    ax=axes[row,col]
                    for label,handle in (('Without NORDIC',pre),('With NORDIC',post)):
                        g=handle[side]['matched_without_selection'] if matched else handle[side]
                        ax.plot(DEPTH,g[f'layer_effect_{key}'][:],'-o',markersize=3,label=label)
                    ax.axhline(0,color='gray',linewidth=.6)
                    ax.set(title=f'{side}: {key}',xlabel='Depth (0=white, 1=pial)',ylabel='Mean contrast estimate (not t, not PSC)')
                    ax.legend()
            fig.suptitle('Frozen SPM weights | '+('SAME baseline 200 columns in both conditions' if matched else 'Top200 selected separately in each condition'))
            fig.savefig(OUT/('profiles_matched_columns.png' if matched else 'profiles_independent_top200.png'),dpi=180)
            plt.close(fig)
        fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
        for row,side in enumerate(SIDES):
            for col,level in enumerate(('high','low')):
                ax=axes[row,col]
                for label,handle in (('Without NORDIC',pre),('With NORDIC',post)):
                    g=handle[side]['matched_without_selection']
                    ax.plot(DEPTH,g[f'layer_anticipation_{level}_beta_mean'][:],'-o',markersize=3,label=label)
                ax.set(title=f'{side}: anticipation {level} cognition',xlabel='Depth (0=white, 1=pial)',ylabel='Condition beta averaged across runs')
                ax.axhline(0,color='gray',linewidth=.6)
                ax.legend()
        fig.suptitle('DESCRIPTIVE anticipation betas | matched baseline columns | frozen SPM weights')
        fig.savefig(OUT/'anticipation_descriptive_betas.png',dpi=180)
        plt.close(fig)


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        log(f'ERROR: {error}')
        sys.exit(1)
