"""Subject 7485 WITH NORDIC sampling: corrected loop and zero-based coordinates.
Requirements: numpy scipy nibabel h5py tqdm
Original images remain read-only. Existing output is backed up ONLY after
new sampling has completed and passed completeness checks.
The direct surface/world mapping is specific to the inspected subject 7485;
this script is not a claim of complete MATLAB/SPM numerical equivalence.
"""
from pathlib import Path
import sys
import time
import h5py
import nibabel as nib
import numpy as np
from nibabel.freesurfer.io import read_geometry
from scipy.ndimage import map_coordinates
from tqdm import tqdm

SUBJECT = '7485'
BASE = Path(r'D:\NORDIC_Subject1_project')
FS_SUBJECT = BASE / '7485_extracted' / SUBJECT
T1_FILE = BASE / '7485_extracted' / 'UNI_MoCo_MPRAGEised_biascorrected.nii'
ROI_DIR = BASE / 'Laminarprofile' / 'masks'
RUN_DIRS = [BASE / 'Motion and Regstration with Nordic' / f'final_warps_run{i}' for i in (1,2,3)]
OUTPUT_FILE = BASE / 'Laminarprofile' / '1. Sampling' / 'With' / 'pIns_laminar_sampling_7485_with_NORDIC.h5'
ROI_STEMS = {'left':'roi_pInsleft_17','right':'roi_pInsright_118'}
SIDES = ('left','right')
N_LAYERS = 20
N_PER_RUN = 407
ORDER = 3


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}",flush=True)


def ordered_run_files(folder):
    import re
    pattern = re.compile(r'^NORDIC_(\d+)_Warped-to-Anat\.nii\.gz$')
    pairs = []
    for path in folder.glob('NORDIC_*_Warped-to-Anat.nii.gz'):
        match = pattern.fullmatch(path.name)
        if match:
            pairs.append((int(match.group(1)),path))
    pairs.sort(key=lambda pair:pair[0])
    indices = [index for index,_ in pairs]
    expected = list(range(1000,1407))
    if indices != expected:
        raise ValueError(f'{folder}: expected 407 indices 1000..1406; found {len(pairs)}. Missing: {sorted(set(expected)-set(indices))}')
    return [path for _,path in pairs]


def ras_to_vox(xyz,affine):
    hom = np.column_stack((xyz,np.ones(len(xyz))))
    return np.linalg.solve(affine,hom.T).T[:,:3]


def main():
    surf = FS_SUBJECT / 'surf'
    required = [T1_FILE]+[surf/f'{hemi}.{boundary}' for hemi in ('lh','rh') for boundary in ('white','pial')]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    runs = [ordered_run_files(folder) for folder in RUN_DIRS]
    files = [path for run in runs for path in run]
    ntime = len(files)
    if ntime != 1221:
        raise ValueError('Expected exactly 1221 volumes')
    OUTPUT_FILE.parent.mkdir(parents=True,exist_ok=True)
    temporary = OUTPUT_FILE.with_suffix('.h5.partial')
    if temporary.exists():
        raise FileExistsError(f'Incomplete previous attempt exists:\n{temporary}\nRename it before restarting; original final H5 is untouched.')
    t1 = nib.load(str(T1_FILE))
    if len(t1.shape)!=3:
        raise ValueError('Expected 3D T1')
    white = np.vstack([read_geometry(str(surf/f'{hemi}.white'))[0] for hemi in ('lh','rh')])
    pial = np.vstack([read_geometry(str(surf/f'{hemi}.pial'))[0] for hemi in ('lh','rh')])
    if white.shape != pial.shape:
        raise ValueError('White/pial dimensions differ')
    white_vox = ras_to_vox(white,t1.affine)
    pial_vox = ras_to_vox(pial,t1.affine)
    masks,coordinates,thickness = {},{},{}
    for side,stem in ROI_STEMS.items():
        values=[]
        for prefix in ('lh','rh'):
            path=ROI_DIR/f'{stem}_{prefix}.shape.gii'
            if not path.is_file():
                raise FileNotFoundError(path)
            values.append(np.asarray(nib.load(str(path)).darrays[0].data).ravel())
        mask=np.abs(np.concatenate(values))>=.01
        if mask.size!=len(white) or not mask.any():
            raise ValueError(f'{side}: invalid ROI mask')
        masks[side]=mask
        w,p=white_vox[mask],pial_vox[mask]
        points=w[:,None,:]+np.linspace(0,1,20)[None,:,None]*(p-w)[:,None,:]
        coordinates[side]=points.reshape(-1,3).T
        thickness[side]=np.linalg.norm(p-w,axis=1).astype(np.float32)
        inside=np.all((points>=0)&(points<=np.array(t1.shape)-1),axis=2)
        if not inside.all():
            raise ValueError(f'{side}: some sampling points fall outside the T1 grid. Inspect before continuing.')
        log(f'{side}: {mask.sum()} columns x 20 depths x {ntime} time points')
    log('Sampling every volume INSIDE the volume loop; unwritten values initialized to NaN')
    with h5py.File(temporary,'x') as h:
        h.attrs['subject']=SUBJECT
        h.attrs['condition']='with_NORDIC'
        h.attrs['t1_reference']=str(T1_FILE)
        h.attrs['n_layers']=20
        h.attrs['n_volumes']=ntime
        h.attrs['interpolation_order']=ORDER
        h.attrs['coordinate_offset_vox']=0.0
        h.attrs['coordinate_method']='inverse nibabel T1 affine; no -1; subject7485 inspected mapping'
        h.attrs['sampling_direction']='0=white; 19=pial'
        h.attrs['complete']=False
        h.attrs['n_written_volumes']=0
        h.create_dataset('run_lengths',data=np.array([407,407,407],dtype=np.int32))
        h.create_dataset('volume_files',data=np.array([str(path) for path in files],dtype=h5py.string_dtype('utf-8')))
        h.create_dataset('depth_fractions',data=np.linspace(0,1,20))
        written=h.create_dataset('written_volumes',shape=(ntime,),dtype='bool',fillvalue=False)
        for side in SIDES:
            ncol=int(masks[side].sum())
            g=h.create_group(side)
            g.create_dataset('surface_vertex_indices',data=np.flatnonzero(masks[side]))
            g.create_dataset('column_thickness_vox',data=thickness[side])
            g.create_dataset('sampling_coordinates_vox',data=coordinates[side].T.reshape(ncol,20,3))
            g.create_dataset('layeractivation',shape=(ncol,20,ntime),dtype='float32',chunks=(min(256,ncol),20,1),compression='gzip',compression_opts=4,fillvalue=np.nan)
            g.create_dataset('columnspecificts',shape=(ncol,ntime),dtype='float32',chunks=(min(256,ncol),1),compression='gzip',compression_opts=4,fillvalue=np.nan)
            g.create_dataset('all_columns_average',shape=(20,ntime),dtype='float32',fillvalue=np.nan)
        for vol_idx,file_path in enumerate(tqdm(files,desc='Sampling With NORDIC',unit='volume')):
            image=nib.load(str(file_path))
            if image.shape!=t1.shape or not np.allclose(image.affine,t1.affine,rtol=0,atol=1e-5):
                raise ValueError(f'Functional/T1 grid mismatch:\n{file_path}')
            data=image.get_fdata(dtype=np.float32)
            for side in SIDES:
                ncol=int(masks[side].sum())
                values=map_coordinates(data,coordinates[side],order=ORDER,mode='constant',cval=np.nan,prefilter=True).reshape(ncol,20).astype(np.float32)
                if not np.isfinite(values).all():
                    raise ValueError(f'{side}: non-finite samples at volume {vol_idx}')
                if not np.any(values!=0):
                    raise ValueError(f'{side}: ALL samples zero at volume {vol_idx}; inspect input image')
                h[side]['layeractivation'][:,:,vol_idx]=values
                h[side]['columnspecificts'][:,vol_idx]=values[:,:15].mean(axis=1,dtype=np.float64).astype(np.float32)
                h[side]['all_columns_average'][:,vol_idx]=values.mean(axis=0,dtype=np.float64).astype(np.float32)
            written[vol_idx]=True
            h.attrs['n_written_volumes']=vol_idx+1
            if (vol_idx+1)%25==0 or vol_idx+1==ntime:
                h.flush()
                log(f'Checkpoint: {vol_idx+1}/{ntime} actually sampled and written')
        if not written[:].all() or int(h.attrs['n_written_volumes'])!=ntime:
            raise RuntimeError('Completeness verification failed')
        for side in SIDES:
            ts=h[side]['columnspecificts']
            for start in range(0,ntime,100):
                block=ts[:,start:start+100]
                if not np.isfinite(block).all() or np.any(np.all(block==0,axis=0)):
                    raise RuntimeError(f'{side}: invalid or unwritten time points after sampling')
            for start in range(0,ntime,25):
                if not np.isfinite(h[side]['layeractivation'][:,:,start:start+25]).all():
                    raise RuntimeError(f'{side}: layer data contain unwritten/non-finite samples')
        h.attrs['complete']=True
        h.flush()
    if OUTPUT_FILE.exists():
        backup=OUTPUT_FILE.with_name(OUTPUT_FILE.stem+'_OLD_'+time.strftime('%Y%m%d_%H%M%S')+OUTPUT_FILE.suffix)
        if backup.exists():
            raise FileExistsError(backup)
        OUTPUT_FILE.rename(backup)
        log(f'Previous output preserved:\n{backup}')
    temporary.replace(OUTPUT_FILE)
    log(f'Completed and verified: {ntime}/{ntime} volumes written for BOTH hemispheres')
    log(f'Output:\n{OUTPUT_FILE}')
    log('Next: regenerate deep75, then rerun GLM. Do not use the old deep75 file.')


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        log(f'ERROR: {error}')
        log('No successful completion declared. Existing final output is unchanged; inspect any .partial file before restarting.')
        sys.exit(1)
