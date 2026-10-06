"""Build an immutable paired OOD task manifest using the active LIBERO-Plus."""
import argparse
import contextlib
import hashlib
import io
import json
import random
from collections import Counter
from pathlib import Path

from libero.libero import benchmark

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
if (args.output / 'tasks.txt').exists():
    raise FileExistsError('Refusing to overwrite an existing experiment task list')
classification = Path('/data/pxchen/LIBERO-plus/libero/libero/benchmark/task_classification.json')
data = json.loads(classification.read_text())
suites = ['libero_spatial', 'libero_object', 'libero_goal']
categories = ['Objects Layout', 'Camera Viewpoints', 'Robot Initial States',
              'Language Instructions', 'Light Conditions', 'Background Textures', 'Sensor Noise']
rows = []
for si, suite in enumerate(suites):
    with contextlib.redirect_stdout(io.StringIO()):
        bench = benchmark.get_benchmark_dict()[suite](task_order_index=0)
    name_to_id = {bench.get_task(i).name: i for i in range(bench.n_tasks)}
    for ci, category in enumerate(categories):
        candidates = sorted((r for r in data[suite] if r['category'] == category), key=lambda r: r['id'])
        selected = random.Random(239 + si * 100 + ci).sample(candidates, 15)
        for source in selected:
            task_id = name_to_id[source['name']]
            assert task_id == source['id'] - 1, (suite, source, task_id)
            task = bench.get_task(task_id)
            bddl = Path(bench.get_task_bddl_file_path(task_id))
            # Camera/robot variants are virtual BDDL names resolved by
            # LIBERO-Plus OffScreenRenderEnv to the base scene file.
            resolved_bddl = Path(str(bddl).split('_view_')[0] + '.bddl') if '_view_' in str(bddl) else bddl
            if not resolved_bddl.is_file():
                raise FileNotFoundError(resolved_bddl)
            states = bench.get_task_init_states(task_id)
            if len(states) == 0:
                raise ValueError(f'Empty initial states: {suite}/{task_id}')
            rows.append({'suite': suite, 'task_id': task_id, 'classification_id': source['id'],
                         'category': category, 'official_difficulty_level': source['difficulty_level'],
                         'name': task.name, 'instruction': task.language, 'bddl_path': str(bddl),
                         'resolved_bddl_path': str(resolved_bddl),
                         'init_states_filename': task.init_states_file, 'initial_state_index': 0,
                         'available_initial_states': len(states), 'max_control_steps': 400})
random.Random(239).shuffle(rows)
assert len(rows) == 315 and len({(r['suite'], r['task_id']) for r in rows}) == 315
counts = Counter((r['suite'], r['category']) for r in rows)
assert len(counts) == 21 and set(counts.values()) == {15}
settings = {
    'sampling_seed': 239, 'rollout_seed': 7, 'trials_per_task': 1, 'tasks_per_group': 315,
    'total_episodes': 630, 'sampling': '15 unique tasks sampled uniformly within each suite/category cell; identical shuffled order in both groups',
    'difficulty_note': 'Official difficulty metadata is descriptive only; sampling is not stratified by difficulty.',
    'fastwam_checkpoint': '/data/pxchen/PACT-WAM/checkpoints/fastwam/libero_optional_idm_2cam224.pt',
    'dataset_stats': '/data/pxchen/PACT-WAM/checkpoints/fastwam/libero_optional_idm_2cam224_dataset_stats.json',
    'fastwam_dtype': 'bfloat16', 'action_infer_mode': 'idm', 'num_inference_steps': 10,
    'replan_steps': 8, 'video_sigma_shift': 1.0, 'action_sigma_shift': 1.0,
    'fastwam_image_rotation_degrees': 180, 'compile_action_infer': False,
    'visualize_future_video': True, 'offload_text_encoder': True,
    'camera_pose_overrides': None, 'preserve_native_ood_camera': True,
    'trigger_enabled': False,
    'control': {'gpu': 1, 'jepa_enabled': False, 'guidance_enabled': False},
    'experiment': {'gpu': 3, 'jepa_enabled': True, 'guidance_enabled': True,
                   'checkpoint': '/data/pxchen/PACT-WAM/checkpoints/vjepa2-ac-libero90-rot180-step2000-inference.pt',
                   'training_orientation': '180-degree rotated LIBERO demonstrations',
                   'encoder_extra_rotation_degrees': 0, 'jepa_input_orientation': 'inverted, matching FastWAM',
                   'jepa_dtype': 'float32', 'encoder_camera': 'agentview only',
                   'after_flow_steps': [7, 8, 9], 'step_size': 4.0, 'normalize_gradient': False,
                   'max_delta_rms': 0.02, 'ac_steps': 2, 'verify_descent': True,
                   'overlap_encoder_with_action': True, 'num_action_candidates': 8,
                   'candidate_generation_mode': 'separate'},
    'classification_source': str(classification),
    'classification_sha256': hashlib.sha256(classification.read_bytes()).hexdigest(),
}
task_text = ''.join(f"{r['suite']},{r['task_id']}\n" for r in rows)
settings['task_list_sha256'] = hashlib.sha256(task_text.encode()).hexdigest()
args.output.mkdir(parents=True, exist_ok=True)
(args.output / 'tasks.txt').write_text(task_text)
(args.output / 'task_manifest.json').write_text(json.dumps({'settings': settings, 'tasks': rows}, indent=2, ensure_ascii=False) + '\n')
md = ['# LIBERO-Plus short-horizon paired OOD evaluation', '',
      '315 fixed tasks per group, 630 episodes total. Spatial/Object/Goal × seven OOD categories × 15 tasks.', '',
      'Sampling seed: 239; rollout seed: 7; one trial per task, initial-state index 0. Both groups use the same task order.', '',
      'Control: GPU 1, FastWAM IDM, no JEPA/no trigger. Experiment: GPU 3, rot180-trained step-2000 JEPA predictor, inverted online JEPA inputs, full 4× raw-gradient guidance at flow steps 7/8/9.', '',
      'FastWAM remains BF16 with its original 180° image convention. JEPA is FP32 and adds no extra rotation after extracting the primary view; current and generated future frames remain inverted. Native OOD camera poses are preserved. Max episode horizon is 400 control steps (+30 settling steps).', '',
      'Complete settings and metadata are in task_manifest.json. Official difficulty labels are retained, not used as labels of our model performance.', '',
      f"Task-list SHA256: `{settings['task_list_sha256']}`", '',
      '| Order | Suite | Task ID (0-based) | Classification ID (1-based) | OOD category | Official difficulty | Task name |',
      '|---:|---|---:|---:|---|---:|---|']
for i, r in enumerate(rows, 1):
    md.append(f"| {i} | {r['suite']} | {r['task_id']} | {r['classification_id']} | {r['category']} | {r['official_difficulty_level']} | {r['name']} |")
(args.output / 'TASKS_AND_SETTINGS.md').write_text('\n'.join(md) + '\n')
print(json.dumps({'validated_tasks': len(rows), 'cell_counts': {f'{s}/{c}': n for (s,c),n in counts.items()}, 'output': str(args.output)}, indent=2))
