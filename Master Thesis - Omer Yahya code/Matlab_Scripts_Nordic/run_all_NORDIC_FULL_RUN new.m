%% ====================================================================
%  run_all_NORDIC_FULL_RUN.m
%
%  Automatically processes Run 1, Run 2, and Run 3 in sequence using
%  the existing NIFTI_NORDIC_lowmem.m function -- no manual re-running
%  of the script between runs, and no manual editing after a crash.
%
%  >>> RUN THIS FILE MANUALLY IN MATLAB (once). <<<
%  It calls NIFTI_NORDIC_lowmem.m automatically for each run -- do NOT
%  run that file yourself; it must be on the MATLAB path (e.g. in the
%  same folder as this script).
%
%  Insertion point in the pipeline: immediately after POCS
%  reconstruction, before motion correction / preprocessing / a_compcor.
%
%  ----------------------------------------------------------------
%  STRUCTURE: ONE FULLY INDEPENDENT OUTPUT FOLDER PER RUN
%  ----------------------------------------------------------------
%  Each run now has its own field, runs(k).out_dir, e.g.:
%
%     NORDIC_Output/
%     |-- output_run1/     <- runs(1).out_dir
%     |     |-- NORDIC_mag_POCS_r1.nii.gz
%     |     |-- gfactor_NORDIC_mag_POCS_r1.nii.gz
%     |     |-- NORDIC_mag_POCS_r1.mat
%     |     `-- NORDIC_QC/
%     |            |-- QC_before_slice.png ... QC_combined.png
%     |            `-- RUN_COMPLETE.marker
%     |-- output_run2/     <- runs(2).out_dir  (same contents, run2)
%     `-- output_run3/     <- runs(3).out_dir  (same contents, run3)
%
%  Every file a run produces -- the denoised NIfTI, the g-factor map,
%  the .mat diagnostics, the whole NORDIC_QC folder, and that run's
%  RUN_COMPLETE.marker -- lives ONLY inside that run's own out_dir.
%  Nothing is shared across runs, so there is no possibility of run1's
%  and run2's files ever mixing, colliding, or being confused for one
%  another.
%
%  ----------------------------------------------------------------
%  CRASH / RESUME BEHAVIOR (unchanged in spirit, now per-folder)
%  ----------------------------------------------------------------
%  Resume/crash-safety is based on a per-run COMPLETION MARKER, not
%  just "do the output files exist" -- file existence alone is not
%  proof of validity; a crash mid-write (e.g. partway through
%  gzip-compressing the .nii.gz) can leave a truncated file that still
%  "exists" but is corrupt. So instead:
%    a) At the end of a successful run, EVERY output in that run's
%       out_dir (denoised NIfTI, g-factor NIfTI, .mat diagnostics, all
%       QC PNGs) is actually opened and validated (niftiinfo / load /
%       imfinfo), not just checked for existence -- validate_run_outputs().
%    b) Only if ALL of those validate does the script write
%       <out_dir>/NORDIC_QC/RUN_COMPLETE.marker, using an atomic
%       write-then-rename so a crash during the marker write itself
%       cannot leave a fake "complete" marker behind -- write_completion_marker().
%    c) On (re)launch, a run is skipped ONLY if its OWN out_dir's
%       marker exists -- get_marker_path(). No marker = not complete,
%       full stop, regardless of what partial files are sitting in
%       that folder. Because each run's marker lives in its own
%       independent folder, the resume check for one run can never be
%       confused by another run's files.
%    d) Before a run is (re)started, any leftover files inside its OWN
%       out_dir from a previous incomplete attempt are explicitly
%       deleted first (clean_run_outputs()) -- not just silently
%       overwritten -- so a half-written file from a crash can never
%       collide with the new attempt. This only ever touches that run's
%       own folder, never another run's.
%  Net effect: you can crash MATLAB or restart the whole computer at
%  ANY point during ANY run, then simply press Run again. Completed
%  runs (marker present in their own folder) are skipped instantly;
%  the interrupted run is fully cleaned and redone from scratch inside
%  its own folder only; later runs proceed normally.
%  ====================================================================

clearvars
clc

%% ======================================================================
%  EDIT THIS SECTION ONLY -- one entry per run: input paths, its own
%  independent output folder, and an optional expected volume count.
%  >>> Set fn_magn_in / fn_phase_in / out_dir for each run to match
%      your own dataset. <<<
%  ======================================================================

% NOTE: each run's out_dir below is a full, independent absolute path,
% so it is assigned directly rather than via fullfile(parent, ...) --
% fullfile() in MATLAB does NOT treat a leading '/' as "ignore the
% first argument" (unlike os.path.join in some other languages); it
% would simply concatenate both parts into a wrong, nested path.

runs = struct('label', {}, 'fn_magn_in', {}, 'fn_phase_in', {}, 'out_dir', {}, 'expected_volumes', {});

runs(1).label            = 'run1';
runs(1).fn_magn_in       = '/home/SHK/Documents/layerdata/7468/layers/run1/func/mag_POCS_r1.nii.gz';
runs(1).fn_phase_in      = '/home/SHK/Documents/layerdata/7468/layers/run1/func/pha_POCS.nii.gz';
runs(1).out_dir          = '/home/SHK/Documents/OY_SUBJECT2/RUN1';   % independent output folder for run1
runs(1).expected_volumes = 407;             % set to [] to skip the volume-count sanity check

runs(2).label            = 'run2';
runs(2).fn_magn_in       = '/home/SHK/Documents/layerdata/7468/layers/run2/func/mag_POCS_r2.nii.gz';
runs(2).fn_phase_in      = '/home/SHK/Documents/layerdata/7468/layers/run2/func/pha_POCS.nii.gz';
runs(2).out_dir          = '/home/SHK/Documents/OY_SUBJECT2/RUN2';   % independent output folder for run2
runs(2).expected_volumes = [];              % fill in if you know it, otherwise leave empty

runs(3).label            = 'run3';
runs(3).fn_magn_in       = '/home/SHK/Documents/layerdata/7468/layers/run3/func/mag_POCS_r3.nii.gz';
runs(3).fn_phase_in      = '/home/SHK/Documents/layerdata/7468/layers/run3/func/pha_POCS.nii.gz';
runs(3).out_dir          = '/home/SHK/Documents/OY_SUBJECT2/RUN3';   % independent output folder for run3
runs(3).expected_volumes = [];              % fill in if you know it, otherwise leave empty

% NOTE: each runs(k).out_dir above is a fully independent absolute
% path -- they don't need any common parent folder. Nothing below this
% section assumes they share one.

%  ======================================================================
%  DO NOT EDIT BELOW THIS LINE.
%  ======================================================================

%% ---- ARG configuration: best scientifically defensible full-run settings ----
% IDENTICAL to the original single-run script. Shared across all three
% runs since the acquisition/reconstruction settings don't change
% between runs of the same session. ARG.DIROUT is intentionally NOT set
% here -- it is set per-run, inside process_single_run(), to that run's
% own out_dir, so this shared ARG template never points at any
% particular run's folder.
clear ARG

ARG.magnitude_only = 0;          % REQUIRED: complex processing using both magnitude and phase.
                                  % This is the entire point of having the phase data at all --
                                  % magnitude-only NORDIC discards information the complex method uses.

ARG.temporal_phase = 1;          % fMRI default per NIFTI_NORDIC's own documented fMRI usage
                                  % example. (=3 is the dMRI-specific setting -- not applicable.)

ARG.phase_filter_width = 10;     % fMRI value from that same usage example. The function's
                                  % internal fallback default (3) is commented in the source as
                                  % a later change motivated by dMRI use -- explicitly overriding
                                  % so this fMRI run doesn't silently inherit a dMRI-tuned value.

ARG.NORDIC = 1;  ARG.MP = 0;     % Hard NORDIC threshold -- the validated method from Moeller/
                                  % Vizioli et al. 2021 (ref [1] in your proposal), not the
                                  % alternative Marchenko-Pastur soft-thresholding mode.

ARG.noise_volume_last = 0;       % No separate trailing noise-calibration volume exists in this
                                  % acquisition -> the noise threshold is derived entirely from
                                  % the self-estimated g-factor map. This IS the "auto-estimated
                                  % g-factor mode" -- not a simplification, the intended mode.

ARG.kernel_size_gfactor = [];    % Leave EMPTY (default [14 14 1]). Standard in-plane g-factor
                                  % estimation kernel -- no reason from your data to override it.

ARG.kernel_size_PCA = [];        % Leave EMPTY -- this is the important one for the full run.
                                  % Empty triggers the auto 11:1 spatial:temporal patch ratio,
                                  % computed from however many volumes are given:
                                  %   kernel = round((T*11)^(1/3))
                                  % This IS the validated, intended behavior of the auto-mode --
                                  % manually overriding it would be an unvalidated deviation, not
                                  % an improvement.

ARG.factor_error = 1.0;          % No manual adjustment to the estimated noise floor.

ARG.full_dynamic_range = 0;      % Preserve the input's native intensity scale. This is the
                                  % scientifically correct choice for your before/after
                                  % comparison: it keeps NORDIC output and the raw POCS
                                  % magnitude in the same units, so mean-signal and tSNR
                                  % comparisons aren't confounded by an extra rescaling factor.

ARG.save_gfactor_map = 1;        % Save the relative g-factor map. Computed internally anyway
                                  % (near-zero extra cost); directly supports your proposal's
                                  % "quantify the amount of thermal noise removed" objective and
                                  % is good QC evidence for the thesis.

ARG.write_gzipped_niftis = true; % Compressed output, matching the existing .nii.gz convention
                                  % of mag_POCS_r*.nii.gz / pha_POCS.nii.gz, and saving real disk
                                  % space at full resolution.

ARG.save_add_info = 1;           % Saves an additional .mat file alongside the output containing
                                  % the ARG settings actually used (and internal diagnostics).
                                  % Cheap, and valuable for thesis Methods/reproducibility.

% ARG.make_complex_nii intentionally NOT set -> a single real-valued
% magnitude NIfTI output per run, matching what downstream motion
% correction / a_compcor expects.

fprintf('\nARG configuration (shared across all runs, DIROUT set per-run):\n'); disp(ARG)

%% ---- Main loop: process Run 1 -> Run 2 -> Run 3 automatically --------
% AUTOMATIC RESUME: each run's completion marker -- inside that run's
% OWN independent out_dir -- is checked before it is (re)started. This
% makes the script safe to simply re-launch after a crash/restart at
% ANY point -- already-finished runs are detected and skipped, and
% processing resumes from the first unfinished run, with any leftover
% partial outputs from an interrupted attempt cleaned up first, always
% confined to that one run's own folder.
n_runs = numel(runs);
results = struct('label', {}, 'status', {}, 'message', {}, 'elapsed_sec', {}, 'peak_mem_GB', {});

for k = 1:n_runs
    fprintf('\n============================================================\n');
    fprintf(' Checking %s (%d of %d)\n', runs(k).label, k, n_runs);
    fprintf(' Output folder: %s\n', runs(k).out_dir);
    fprintf('============================================================\n');

    % Defensive path normalization, done per-run since each run now has
    % its own independent out_dir. This codebase builds output paths by
    % direct string concatenation ([ARG.DIROUT fn_out ...]), NOT
    % fullfile(), so out_dir must end in a separator or the output
    % filename gets mashed onto the folder name.
    this_out_dir = runs(k).out_dir;
    if ~(this_out_dir(end) == filesep)
        this_out_dir = [this_out_dir filesep];
    end
    if ~exist(this_out_dir, 'dir'); mkdir(this_out_dir); end
    runs(k).out_dir = this_out_dir;   % store the normalized form back

    marker_path = get_marker_path(runs(k));

    if exist(marker_path, 'file') == 2
        % Marker present in THIS run's own out_dir -> this run was
        % previously validated as fully complete (every output was
        % opened and checked, not just "exists"). Never reprocess it.
        fprintf('%s already completed - skipping.\n', runs(k).label);
        results(k).label       = runs(k).label;
        results(k).status      = 'SKIPPED (already complete)';
        results(k).message     = '';
        results(k).elapsed_sec = NaN;
        results(k).peak_mem_GB = NaN;
        continue
    end

    % No marker -> this run is not proven complete. Either this is the
    % first attempt, or a previous attempt was interrupted (MATLAB or
    % computer crash) partway through. Either way: report and remove
    % any leftover partial outputs from THIS run's own out_dir first,
    % so the upcoming (re)run starts from a guaranteed-clean slate.
    % This never touches any other run's out_dir.
    leftovers = list_existing_outputs(runs(k));
    if ~isempty(leftovers)
        fprintf('%s has no completion marker -- found %d leftover item(s) in its\n', ...
            runs(k).label, numel(leftovers));
        fprintf('own output folder from an incomplete previous attempt. Removing:\n');
        for i = 1:numel(leftovers)
            fprintf('    %s\n', leftovers{i});
        end
    end
    clean_run_outputs(runs(k));

    fprintf('Starting %s...\n', runs(k).label);

    try
        [elapsed_sec, peak_mem_GB] = process_single_run(runs(k), ARG);
        results(k).label       = runs(k).label;
        results(k).status      = 'OK';
        results(k).message     = '';
        results(k).elapsed_sec = elapsed_sec;
        results(k).peak_mem_GB = peak_mem_GB;
        fprintf('\n%s finished successfully and was validated + marked complete.\n', runs(k).label);
        fprintf('Continuing automatically...\n');
    catch ME
        % Do not stop the whole batch if one run fails -- log it and
        % move on to the next run. No marker was written, so a future
        % re-launch will retry this run (after cleaning its own folder
        % again). Other runs' folders are completely unaffected.
        results(k).label       = runs(k).label;
        results(k).status      = 'FAILED';
        results(k).message     = ME.message;
        results(k).elapsed_sec = NaN;
        results(k).peak_mem_GB = NaN;
        fprintf(2, '\n*** %s FAILED: %s\n', runs(k).label, ME.message);
        fprintf(2, 'No completion marker was written, so re-running this script later\n');
        fprintf(2, 'will retry %s automatically. Continuing to the next run now...\n', runs(k).label);
    end
end

%% ---- Completion message / summary -------------------------------------
fprintf('\n============================================================\n');
fprintf(' ALL RUNS COMPLETE (%d of %d attempted)\n', n_runs, n_runs);
fprintf('============================================================\n');
for k = 1:n_runs
    switch results(k).status
        case 'OK'
            fprintf('  %-6s: OK      | runtime = %6.1f min | peak mem = %5.2f GB | %s\n', ...
                results(k).label, results(k).elapsed_sec/60, results(k).peak_mem_GB, runs(k).out_dir);
        case 'SKIPPED (already complete)'
            fprintf('  %-6s: SKIPPED | validated complete in a previous run | %s\n', ...
                results(k).label, runs(k).out_dir);
        otherwise
            fprintf('  %-6s: FAILED  | reason: %s | %s\n', ...
                results(k).label, results(k).message, runs(k).out_dir);
    end
end
fprintf('============================================================\n');


%% ====================================================================
%  Local functions -- must stay at the end of the script file
%  ====================================================================

function [fn_out, qc_dir] = compute_output_names(run_cfg)
% Derives the bare NORDIC output stem and the QC folder for a given
% run, both relative to that run's OWN independent out_dir. Pulled out
% into its own function so that every other function (get_marker_path,
% list_existing_outputs, clean_run_outputs, validate_run_outputs,
% process_single_run) always agrees on exactly which filenames/folders
% a run produces -- they can never drift out of sync with each other,
% and every path returned here is confined to run_cfg.out_dir.

fn_magn_in = run_cfg.fn_magn_in;

[~, magn_name, magn_ext] = fileparts(fn_magn_in);   % fileparts strips only the LAST extension (.gz)
if strcmpi(magn_ext, '.gz')
    [~, magn_name, ~] = fileparts(magn_name);       % strip the remaining .nii too
end
fn_out = ['NORDIC_' magn_name];                      % e.g. 'NORDIC_mag_POCS_r1' / '..._r2' / '..._r3'

% No per-run label subfolder needed anymore -- run_cfg.out_dir is
% already an independent folder for this run alone.
qc_dir = fullfile(run_cfg.out_dir, 'NORDIC_QC');

end


function marker_path = get_marker_path(run_cfg)
% Path to this run's completion marker, inside that run's OWN out_dir.
% Its presence is the ONLY thing that makes the main loop skip a run --
% see the header comment for the rationale (existence of output files
% alone is not proof of validity; a crash mid-write can leave a
% corrupt-but-present file). Because it lives inside run_cfg.out_dir,
% it can never be confused with another run's marker.

[~, qc_dir] = compute_output_names(run_cfg);
marker_path = fullfile(qc_dir, 'RUN_COMPLETE.marker');

end


function existing = list_existing_outputs(run_cfg)
% Lists any output files/folders for this run already present in its
% OWN out_dir, purely for the informational message printed before a
% run is (re)started. Does not affect the skip/run decision (only the
% marker does) -- this is diagnostic only, so you can see what's being
% cleaned up and why. Never looks outside run_cfg.out_dir.

[fn_out, qc_dir] = compute_output_names(run_cfg);
out_dir = run_cfg.out_dir;
existing = {};

candidates = { fullfile(out_dir, [fn_out '.nii']), ...
               fullfile(out_dir, [fn_out '.nii.gz']), ...
               fullfile(out_dir, ['gfactor_' fn_out '.nii']), ...
               fullfile(out_dir, ['gfactor_' fn_out '.nii.gz']), ...
               fullfile(out_dir, [fn_out '.mat']) };
for i = 1:numel(candidates)
    if exist(candidates{i}, 'file') == 2
        existing{end+1} = candidates{i}; %#ok<AGROW>
    end
end
if exist(qc_dir, 'dir') == 7
    existing{end+1} = [qc_dir ' (QC folder)']; %#ok<AGROW>
end

end


function clean_run_outputs(run_cfg)
% Deletes ALL of this run's known output artifacts from its OWN
% out_dir before a (re)processing attempt, so a crashed/partial
% previous attempt can never leave a stray file that collides with the
% new attempt. Called every time a run is about to be (re)started
% (i.e. whenever its own marker is absent) -- for a brand-new run this
% is simply a no-op since nothing exists yet. Only ever touches
% run_cfg.out_dir -- never any other run's folder.

[fn_out, qc_dir] = compute_output_names(run_cfg);
out_dir = run_cfg.out_dir;

candidates = { fullfile(out_dir, [fn_out '.nii']), ...
               fullfile(out_dir, [fn_out '.nii.gz']), ...
               fullfile(out_dir, ['gfactor_' fn_out '.nii']), ...
               fullfile(out_dir, ['gfactor_' fn_out '.nii.gz']), ...
               fullfile(out_dir, [fn_out '.mat']) };
for i = 1:numel(candidates)
    if exist(candidates{i}, 'file') == 2
        delete(candidates{i});
    end
end

% Remove the whole QC subfolder (including any stale marker inside it)
% rather than deleting files one by one, so nothing can be missed.
if exist(qc_dir, 'dir') == 7
    rmdir(qc_dir, 's');
end

end


function validate_run_outputs(run_cfg, ARG)
% Opens/reads every required output for this run -- all inside its OWN
% out_dir -- to confirm it is actually valid, not merely present. This
% is what a completion marker is allowed to certify -- if any check
% here fails, an error is thrown, no marker is written, and the run
% will be cleaned and retried (within its own folder only) on the next
% launch.

[fn_out, qc_dir] = compute_output_names(run_cfg);
out_dir = run_cfg.out_dir;

% 1) NORDIC denoised NIfTI -- must exist AND be readable as a NIfTI
fn_denoised = find_existing_output(out_dir, fn_out);
try
    niftiinfo(fn_denoised);
catch ME
    error('Validation failed: denoised NIfTI is unreadable/corrupt (%s): %s', fn_denoised, ME.message);
end

% 2) g-factor map -- same check, only expected because save_gfactor_map is set
if isfield(ARG,'save_gfactor_map') && ~isempty(ARG.save_gfactor_map)
    fn_gfactor = find_existing_output(out_dir, ['gfactor_' fn_out]);
    try
        niftiinfo(fn_gfactor);
    catch ME
        error('Validation failed: g-factor NIfTI is unreadable/corrupt (%s): %s', fn_gfactor, ME.message);
    end
end

% 3) .mat diagnostics file -- only expected because save_add_info is set
if isfield(ARG,'save_add_info') && ARG.save_add_info
    fn_addinfo = fullfile(out_dir, [fn_out '.mat']);
    assert(exist(fn_addinfo,'file')==2, 'Validation failed: .mat diagnostics file missing (%s)', fn_addinfo);
    try
        load(fn_addinfo); %#ok<LOAD> -- just confirming the file loads without error
    catch ME
        error('Validation failed: .mat diagnostics file is unreadable/corrupt (%s): %s', fn_addinfo, ME.message);
    end
end

% 4) All 7 QC PNGs -- must exist AND be readable as images
qc_files = {'QC_before_slice.png', 'QC_after_slice.png', 'QC_diff_slice.png', ...
            'QC_before_meanslice.png', 'QC_after_meanslice.png', 'QC_diff_meanslice.png', ...
            'QC_combined.png'};
for i = 1:numel(qc_files)
    p = fullfile(qc_dir, qc_files{i});
    assert(exist(p,'file')==2, 'Validation failed: QC image missing (%s)', p);
    try
        imfinfo(p);
    catch ME
        error('Validation failed: QC image is unreadable/corrupt (%s): %s', p, ME.message);
    end
end

end


function write_completion_marker(marker_path)
% Atomically creates the completion marker: write to a temp file first,
% then rename it into place. A same-folder rename is effectively
% instantaneous and atomic on the filesystem, so even a crash occurring
% at the exact moment of marker creation cannot leave a half-written
% (and therefore misleading) marker file behind -- either the rename
% completed and the marker fully exists, or it didn't and the marker
% simply doesn't exist yet.

tmp_path = [marker_path '.tmp'];
fid = fopen(tmp_path, 'w');
assert(fid ~= -1, 'Could not create temporary marker file: %s', tmp_path);
fprintf(fid, 'NORDIC run validated complete at %s\n', datestr(now));
fclose(fid);
movefile(tmp_path, marker_path);   % atomic rename within the same folder

end


function [elapsed_sec, peak_mem_GB] = process_single_run(run_cfg, ARG)
% Runs the full NORDIC + QC pipeline for a single run, writing
% EVERYTHING -- denoised NIfTI, g-factor map, .mat diagnostics, QC
% folder, and completion marker -- exclusively into run_cfg.out_dir.
% This is the same logic as the original single-run script -- just
% parameterized by run_cfg so the same code serves Run 1, Run 2, and
% Run 3 without duplication, and without ever writing outside its own
% folder.

fn_magn_in  = run_cfg.fn_magn_in;
fn_phase_in = run_cfg.fn_phase_in;
out_dir     = run_cfg.out_dir;
label       = run_cfg.label;

%% ---- Sanity check on inputs before doing anything expensive ----------
assert(exist(fn_magn_in,'file')==2, 'Magnitude input not found: %s', fn_magn_in);
assert(exist(fn_phase_in,'file')==2, 'Phase input not found: %s', fn_phase_in);

info_check = niftiinfo(fn_magn_in);
fprintf('[%s] Input confirmed: %s\n  ImageSize = %s\n', label, fn_magn_in, mat2str(info_check.ImageSize));
if ~isempty(run_cfg.expected_volumes) && info_check.ImageSize(4) ~= run_cfg.expected_volumes
    warning('[%s] Expected %d volumes, found %d. Double-check this is the right file.', ...
        label, run_cfg.expected_volumes, info_check.ImageSize(4));
end
clear info_check

%% ---- Output filename: bare stem only, no path -------------------------
% Reuses the same helper the resume-check uses, so processing and
% completion-checking can never disagree about expected filenames.
[fn_out, ~] = compute_output_names(run_cfg);

%% ---- ARG.DIROUT: set to THIS run's own independent output folder -----
% ARG is passed in as a shared template (all settings identical across
% runs); DIROUT is the one field that must differ per run, so it is set
% here, on a local copy, immediately before calling NIFTI_NORDIC_lowmem.
% Because MATLAB structs are copied on assignment, this cannot leak
% into the caller's shared ARG or affect any other run.
ARG.DIROUT = out_dir;

%% ---- Run, with runtime + peak memory recorded for your records -------
memTimer = timer('ExecutionMode','fixedRate','Period',2);
memTimer.UserData = 0;
memTimer.TimerFcn = @(t,~) set(t,'UserData', max(t.UserData, sampleMatlabMemGB()));
start(memTimer);

fprintf('\n[%s] Starting full-run NORDIC (single pass)...\n', label);
fprintf('[%s] Output folder: %s\n', label, out_dir);
tic
NIFTI_NORDIC_lowmem(fn_magn_in, fn_phase_in, fn_out, ARG);
elapsed_sec = toc;

stop(memTimer);
peak_mem_GB = memTimer.UserData;
delete(memTimer);

fprintf('\n--- %s full run ---\n', label);
fprintf('Runtime:                 %.1f sec (%.1f min)\n', elapsed_sec, elapsed_sec/60);
fprintf('Peak MATLAB memory seen: %.2f GB\n', peak_mem_GB);

%% ---- Robustly locate the denoised output and the g-factor map --------
fn_denoised = find_existing_output(out_dir, fn_out);
fprintf('[%s] Denoised output found:  %s\n', label, fn_denoised);

fn_gfactor = find_existing_output(out_dir, ['gfactor_' fn_out]);
fprintf('[%s] G-factor map found:     %s\n', label, fn_gfactor);

fn_addinfo = fullfile(out_dir, [fn_out '.mat']);
if exist(fn_addinfo,'file')==2
    fprintf('[%s] ARG/diagnostics saved:   %s\n', label, fn_addinfo);
end

%% ---- QC: load both full series once each (workstation RAM assumed) ---
fprintf('\n[%s] Loading data for QC...\n', label);
before_vol = niftiread(fn_magn_in);     % raw POCS magnitude, all volumes
after_vol  = niftiread(fn_denoised);    % NORDIC-denoised magnitude, all volumes

% QC images go into out_dir/NORDIC_QC -- entirely inside this run's own
% independent folder. qc_dir comes from compute_output_names() so it
% always matches what get_marker_path()/validate_run_outputs() look for.
[~, qc_dir] = compute_output_names(run_cfg);
if ~exist(qc_dir,'dir'); mkdir(qc_dir); end

mid_slice = round(size(before_vol,3)/2);   % middle of the 3rd (slice) array dimension;
                                            % no anatomical orientation assumed/corrected

% --- Single-volume panels (volume 1) ---
before_single = double(before_vol(:,:,mid_slice,1));
after_single  = double(after_vol(:,:,mid_slice,1));
diff_single   = after_single - before_single;

save_slice_png(before_single, fullfile(qc_dir, 'QC_before_slice.png'));
save_slice_png(after_single,  fullfile(qc_dir, 'QC_after_slice.png'));
save_diff_png(diff_single,    fullfile(qc_dir, 'QC_diff_slice.png'));
fprintf('[%s] Saved: single-volume before/after/diff PNGs.\n', label);

% --- Temporal-mean panels ---
before_mean_slice = double(mean(before_vol(:,:,mid_slice,:), 4));
after_mean_slice  = double(mean(after_vol(:,:,mid_slice,:), 4));
diff_mean_slice   = after_mean_slice - before_mean_slice;

save_slice_png(before_mean_slice, fullfile(qc_dir, 'QC_before_meanslice.png'));
save_slice_png(after_mean_slice,  fullfile(qc_dir, 'QC_after_meanslice.png'));
save_diff_png(diff_mean_slice,    fullfile(qc_dir, 'QC_diff_meanslice.png'));
fprintf('[%s] Saved: temporal-mean before/after/diff PNGs.\n', label);

n_vols = size(before_vol,4);
clear before_vol after_vol   % no need to keep the full 4D arrays alive any further

%% ---- Combined QC figure: 2 rows (single-volume / temporal-mean) x -----
%       3 cols (before / after / difference)
tmp = sort(before_single(:));
clim_single = [0, tmp(round(0.99*numel(tmp)))];
tmp = sort(before_mean_slice(:));
clim_mean = [0, tmp(round(0.99*numel(tmp)))];
diff_lim_single = max(abs(diff_single(:))) + eps;
diff_lim_mean   = max(abs(diff_mean_slice(:))) + eps;
divmap = diverging_colormap(256);

fig = figure('Color','w','Position',[80 80 1300 800]);

ax1 = subplot(2,3,1); imagesc(before_single, clim_single); axis image off; colormap(ax1,gray);
title(sprintf('Before NORDIC -- vol 1 (%s)', label));
ax2 = subplot(2,3,2); imagesc(after_single, clim_single); axis image off; colormap(ax2,gray);
title(sprintf('After NORDIC -- vol 1 (%s)', label));
ax3 = subplot(2,3,3); imagesc(diff_single, [-diff_lim_single diff_lim_single]); axis image off;
colormap(ax3,divmap); colorbar; title('Difference (after - before), vol 1');

ax4 = subplot(2,3,4); imagesc(before_mean_slice, clim_mean); axis image off; colormap(ax4,gray);
title(sprintf('Before NORDIC -- mean of %d vols', n_vols));
ax5 = subplot(2,3,5); imagesc(after_mean_slice, clim_mean); axis image off; colormap(ax5,gray);
title(sprintf('After NORDIC -- mean of %d vols', n_vols));
ax6 = subplot(2,3,6); imagesc(diff_mean_slice, [-diff_lim_mean diff_lim_mean]); axis image off;
colormap(ax6,divmap); colorbar; title('Difference (after - before), mean');

sgtitle(sprintf('%s -- full-run NORDIC QC (slice %d)', label, mid_slice));

qc_png = fullfile(qc_dir, 'QC_combined.png');
saveas(fig, qc_png);
close(fig);   % close each run's figure so figures don't pile up across 3 automatic runs
fprintf('[%s] Saved: combined QC figure -> %s\n', label, qc_png);

%% ---- Validate every output, then mark the run complete ---------------
% Only after every required output has actually been opened and
% confirmed valid (not just "exists") is the completion marker written,
% inside this run's own out_dir. If validation throws, no marker is
% written and the calling loop's catch block will report this run as
% FAILED -- a future re-launch will then clean and retry only this
% run's own folder automatically.
fprintf('[%s] Validating all outputs before marking complete...\n', label);
validate_run_outputs(run_cfg, ARG);

marker_path = get_marker_path(run_cfg);
write_completion_marker(marker_path);
fprintf('[%s] All outputs validated. Completion marker written: %s\n', label, marker_path);

fprintf('\n[%s] Run complete. Outputs in: %s\n', label, out_dir);
fprintf('[%s] QC images in: %s\n', label, qc_dir);

end


function g = sampleMatlabMemGB()
    m = memory;
    g = m.MemUsedMATLAB/1e9;
end

function fn_found = find_existing_output(out_dir, bare_name)
    % Robustly locate a NORDIC output file regardless of whether it was
    % written compressed (.nii.gz) or uncompressed (.nii).
    candidate_nii   = fullfile(out_dir, [bare_name '.nii']);
    candidate_niigz = fullfile(out_dir, [bare_name '.nii.gz']);
    if exist(candidate_nii, 'file') == 2
        fn_found = candidate_nii;
    elseif exist(candidate_niigz, 'file') == 2
        fn_found = candidate_niigz;
    else
        error(['Could not find expected output. Looked for:\n  %s\n  %s\n' ...
               'Check that NIFTI_NORDIC_lowmem completed without error.'], ...
               candidate_nii, candidate_niigz);
    end
end

function save_slice_png(slice_img, out_path)
    % Memory-light: operates on an already-extracted 2D slice.
    tmp = sort(slice_img(:));
    hi = tmp(max(1, round(0.99*numel(tmp))));
    if hi <= 0; hi = max(slice_img(:)) + eps; end
    scaled = uint8(255 * min(max(slice_img,0), hi) / hi);
    imwrite(scaled, out_path);
end

function save_diff_png(diff_img, out_path)
    % Saves a difference image using a symmetric blue-white-red scale
    % centered at zero, so increases and decreases are distinguishable.
    lim = max(abs(diff_img(:))) + eps;
    norm01 = (diff_img + lim) / (2*lim);   % maps [-lim,lim] -> [0,1]
    idx = max(1, min(256, round(norm01*255)+1));
    cmap = diverging_colormap(256);
    rgb = ind2rgb(idx, cmap);
    imwrite(rgb, out_path);
end

function cmap = diverging_colormap(n)
    % Simple blue-white-red diverging colormap, no toolbox dependency.
    if nargin < 1; n = 256; end
    half = floor(n/2);
    blue_to_white = [linspace(0,1,half)', linspace(0,1,half)', ones(half,1)];
    white_to_red  = [ones(n-half,1), linspace(1,0,n-half)', linspace(1,0,n-half)'];
    cmap = [blue_to_white; white_to_red];
end
