"""Create deep75 H5 inputs for both conditions; NO GLM or Top200.
Requires numpy and h5py. Original sampling files are opened read-only.
"""
from pathlib import Path
import sys
import time
import h5py
import numpy as np

BASE = Path(r"D:\NORDIC_Subject1_project\Laminarprofile")
INPUTS = {
    "without_NORDIC": BASE / "1. Sampling" / "Without" / "pIns_laminar_sampling_7485_without_NORDIC.h5",
    "with_NORDIC": BASE / "1. Sampling" / "With" / "pIns_laminar_sampling_7485_with_NORDIC.h5",
}
OUT_DIR = BASE / "2. GLM 75%" / "deep75_timeseries"
SIDES = ("left", "right")
BLOCK_SIZE = 64
OVERWRITE = True


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def output_path(condition):
    return OUT_DIR / f"pIns_laminar_sampling_7485_{condition}_deep75_column_timeseries.h5"


def validate_pair():
    for path in INPUTS.values():
        if not path.is_file():
            raise FileNotFoundError(f"Missing sampling file:\n{path}")
    with h5py.File(INPUTS['without_NORDIC'],'r') as pre, h5py.File(INPUTS['with_NORDIC'],'r') as post:
        for handle in (pre,post):
            lengths = np.asarray(handle['run_lengths'][:],dtype=int)
            if lengths.ndim != 1 or np.any(lengths <= 0):
                raise ValueError('Invalid run lengths')
            for side in SIDES:
                data = handle[side]['layeractivation']
                ids = handle[side]['surface_vertex_indices'][:]
                if data.ndim != 3 or data.shape[1] != 20:
                    raise ValueError(f'{side}: expected columns x 20 depths x time; got {data.shape}')
                if not data.shape[0] or data.shape[0] != ids.size or data.shape[2] != lengths.sum():
                    raise ValueError(f'{side}: inconsistent columns/time dimensions')
                if np.unique(ids).size != ids.size:
                    raise ValueError(f'{side}: duplicate vertex IDs')
        if not np.array_equal(pre['run_lengths'][:],post['run_lengths'][:]):
            raise ValueError('Pre/post run lengths differ')
        for side in SIDES:
            if pre[side]['layeractivation'].shape != post[side]['layeractivation'].shape:
                raise ValueError(f'{side}: pre/post data dimensions differ')
            if not np.array_equal(pre[side]['surface_vertex_indices'][:],post[side]['surface_vertex_indices'][:]):
                raise ValueError(f'{side}: pre/post vertex identity/order differs')
    log('Validated matching pre/post columns, depths, ordering and run lengths')


def process_one(source,condition,target):
    temporary = target.with_suffix('.h5.partial')
    if temporary.exists():
        raise FileExistsError(f'Incomplete output exists; inspect/remove first:\n{temporary}')
    try:
        with h5py.File(source,'r') as src, h5py.File(temporary,'x') as dst:
            for key,value in src.attrs.items():
                dst.attrs[key] = value
            dst.attrs['condition'] = condition
            dst.attrs['source_sampling_file'] = str(source)
            dst.attrs['analysis_step'] = 'Mean time series across depth indices 0..14; NO GLM'
            dst.attrs['n_depths_total'] = 20
            dst.attrs['deep_depth_indices_zero_based'] = np.arange(15,dtype=np.int32)
            dst.attrs['deep_depth_indices_one_based'] = np.arange(1,16,dtype=np.int32)
            dst.attrs['deep_depth_fraction'] = .75
            dst.attrs['depth_direction'] = '0=white; 19=pial (input convention, not independently verified)'
            dst.attrs['complete'] = False
            for name in ('run_lengths','volume_files','depth_fractions'):
                if name in src:
                    src.copy(name,dst)
            for side in SIDES:
                gin = src[side]
                data = gin['layeractivation']
                ncol,_,ntime = data.shape
                gout = dst.create_group(side)
                for name in ('surface_vertex_indices','column_thickness_vox','column_thickness_mm'):
                    if name in gin:
                        gin.copy(name,gout)
                series = gout.create_dataset('deep75_column_timeseries',shape=(ncol,ntime),dtype='float32',chunks=(min(BLOCK_SIZE,ncol),min(407,ntime)),compression='gzip',compression_opts=4)
                log(f'{condition} | {side}: {ncol} columns x {ntime} time points')
                for start in range(0,ncol,BLOCK_SIZE):
                    stop = min(start+BLOCK_SIZE,ncol)
                    block = np.asarray(data[start:stop,:15,:],dtype=np.float64)
                    bad = np.argwhere(~np.isfinite(block))
                    if bad.size:
                        col,depth,t = bad[0]
                        raise ValueError(f'{condition}/{side}: non-finite input at column {start+col}, depth {depth}, time {t}. Inspect sampling; do not substitute zeros.')
                    series[start:stop,:] = block.mean(axis=1).astype(np.float32)
                gout.create_dataset('finite_fraction_per_column',data=np.ones(ncol,dtype=np.float32))
                for row in np.unique([0,ncol//2,ncol-1]):
                    expected = np.asarray(data[row,:15,:],dtype=np.float64).mean(axis=0).astype(np.float32)
                    if not np.array_equal(series[row,:],expected):
                        raise RuntimeError('Saved output verification failed')
                log(f'{condition} | {side}: saved and verified {series.shape}')
            dst.attrs['complete'] = True
            dst.flush()
        temporary.replace(target)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    log(f'Saved:\n{target}')


def main():
    validate_pair()
    OUT_DIR.mkdir(parents=True,exist_ok=True)
    for condition in INPUTS:
        target = output_path(condition)
        if target.exists() and not OVERWRITE:
            raise FileExistsError(f'Output exists:\n{target}\nSet OVERWRITE=True only to replace it intentionally.')
    for condition,source in INPUTS.items():
        process_one(source,condition,output_path(condition))
    log('DONE: both deep75 files ready. No GLM, t-values or Top200 calculated.')


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        log(f'ERROR: {error}')
        sys.exit(1)
