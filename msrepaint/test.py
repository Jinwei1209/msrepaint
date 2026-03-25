import os
import subprocess
import argparse
from pathlib import Path

# Project root is one level up from this file
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)


def run_ddpm(gpu_id, t1_path, t2_path, flair_path, lesion_mask_fpath,
             synth_t1_fpath, synth_t2_fpath, synth_flair_fpath,
             out_t1_fpath, out_t2_fpath, out_flair_fpath,
             task_id, inference_plane, pretrained_model):
    cmd = f'python -m msrepaint.test_volume_mc'
    cmd += f' --gpu_id {gpu_id}'
    cmd += f' --input_fpath {t1_path}'
    cmd += f' --input_fpath {t2_path}'
    cmd += f' --input_fpath {flair_path}'
    cmd += f' --input_synthetic_fpath {synth_t1_fpath}'
    cmd += f' --input_synthetic_fpath {synth_t2_fpath}'
    cmd += f' --input_synthetic_fpath {synth_flair_fpath}'
    cmd += f' --lesion_mask_fpath {lesion_mask_fpath}'
    cmd += f' --output_fpath {out_t1_fpath}'
    cmd += f' --output_fpath {out_t2_fpath}'
    cmd += f' --output_fpath {out_flair_fpath}'
    cmd += f' --task {task_id}'
    cmd += f' --inference_plane {inference_plane}'
    cmd += f' --permute_id 0'
    cmd += f' --flip_id 0'
    cmd += f' --pretrained_model {pretrained_model}'
    subprocess.run(cmd, shell=True, cwd=PROJECT_ROOT)


def run_fusion(gpu_id, image_paths, out_path, pretrained_fusion):
    cmd = f'python -m msrepaint.fusion'
    cmd += f' --gpu_id {gpu_id}'
    cmd += f' --image_paths {",".join(image_paths)}'
    cmd += f' --out_path {out_path}'
    if pretrained_fusion:
        cmd += f' --pretrained_fusion {pretrained_fusion}'
    subprocess.run(cmd, shell=True, cwd=PROJECT_ROOT)


def main():
    parser = argparse.ArgumentParser(description="Single-case BiDDPM lesion filling / synthesis")
    parser.add_argument('--t1_path',           type=str, default='no_exist_fpath')
    parser.add_argument('--t2_path',           type=str, default='no_exist_fpath')
    parser.add_argument('--flair_path',        type=str, default='no_exist_fpath')
    parser.add_argument('--lesion_mask_fpath', type=str, required=True)
    parser.add_argument('--pretrained_model',  type=str, required=True)
    parser.add_argument('--pretrained_fusion', type=str, default='')
    parser.add_argument('--output_dir',        type=str, default='')   # default: same dir as t1
    parser.add_argument('--task',              type=int, default=0)    # 0: filling; 1: synthesis
    parser.add_argument('--gpu_id',            type=str, default='0')
    opt = vars(parser.parse_args())

    t1_path           = opt['t1_path']
    t2_path           = opt['t2_path']
    flair_path        = opt['flair_path']
    lesion_mask_fpath = opt['lesion_mask_fpath']
    pretrained_model  = opt['pretrained_model']
    pretrained_fusion = opt['pretrained_fusion']
    task_id           = opt['task']
    gpu_id            = opt['gpu_id']
    _first_existing = next(p for p in [t1_path, t2_path, flair_path] if p != 'no_exist_fpath')
    _first_existing_path = Path(_first_existing).resolve()
    proc_dir   = _first_existing_path.parent.parent / 'tmp' / 'msrepaint'   # intermediate per-plane outputs
    final_dir  = opt['output_dir'] if opt['output_dir'] else str(_first_existing_path.parent)  # final fusion outputs
    os.makedirs(proc_dir, exist_ok=True)

    task_tag = 'lesionfilling' if task_id == 0 else 'lesionsynthesis'

    def out_path(src_path, plane):
        if src_path == 'no_exist_fpath':
            return 'no_exist_fpath'
        basename = os.path.basename(src_path).replace('.nii.gz', f'_{task_tag}_ddpm_{plane}_permute0_flip0.nii.gz')
        return os.path.join(proc_dir, basename)

    def fusion_path(src_path):
        if src_path == 'no_exist_fpath':
            return 'no_exist_fpath'
        basename = os.path.basename(src_path).replace('.nii.gz', '_lesionfilled.nii.gz' if task_id == 0 else '_lesionsynth.nii.gz')
        return os.path.join(final_dir, basename)

    planes = ['axial', 'coronal', 'sagittal']
    t1_plane_paths    = [out_path(t1_path,    p) for p in planes]
    t2_plane_paths    = [out_path(t2_path,    p) for p in planes]
    flair_plane_paths = [out_path(flair_path, p) for p in planes]

    # Run DDPM inference for each plane
    for i, plane in enumerate(planes):
        # warm-start: axial output feeds coronal/sagittal; axial uses None (will be ignored)
        synth_t1    = t1_plane_paths[0]    if plane != 'axial' else t1_path
        synth_t2    = t2_plane_paths[0]    if plane != 'axial' else t2_path
        synth_flair = flair_plane_paths[0] if plane != 'axial' else flair_path

        run_ddpm(
            gpu_id=gpu_id,
            t1_path=t1_path, t2_path=t2_path, flair_path=flair_path,
            lesion_mask_fpath=lesion_mask_fpath,
            synth_t1_fpath=synth_t1, synth_t2_fpath=synth_t2, synth_flair_fpath=synth_flair,
            out_t1_fpath=t1_plane_paths[i], out_t2_fpath=t2_plane_paths[i], out_flair_fpath=flair_plane_paths[i],
            task_id=task_id, inference_plane=plane,
            pretrained_model=pretrained_model,
        )

    # Fuse 3 planes for each contrast
    print('START FUSION')
    for plane_paths, src_path in [
        (t1_plane_paths,    t1_path),
        (t2_plane_paths,    t2_path),
        (flair_plane_paths, flair_path),
    ]:
        if src_path == 'no_exist_fpath':
            continue
        valid_paths = [p for p in plane_paths if p != 'no_exist_fpath']
        run_fusion(gpu_id=gpu_id, image_paths=valid_paths,
                   out_path=fusion_path(src_path),
                   pretrained_fusion=pretrained_fusion)


if __name__ == '__main__':
    main()
