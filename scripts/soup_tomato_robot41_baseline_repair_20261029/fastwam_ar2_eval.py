"""Run-local JEPA replacement; native FastWAM policy/execution stay untouched."""
import sys, json, os, types
from pathlib import Path
from types import SimpleNamespace
# Resolve ImageMagick before vision packages preload conflicting codec libraries.
from wand.api import library as _wand_library
import numpy as np
import torch
from PIL import Image

ROOT = Path('/data/pxchen/PACT-WAM')
sys.path[:0] = [str(ROOT), str(ROOT/'artifacts/predictor2.1ac1009'), '/data/pxchen/vjepa2']
from smoke_joint_predictor import JointPredictor
from vectorized_ac_attention import enable_vectorized

class ARPredictor(JointPredictor):
    def __init__(self):
        super().__init__()
        self.action_steps = 4
        self.action_position = torch.nn.Parameter(self.action_position[:, :4].detach().clone())
        self.phase = torch.nn.Embedding(2, 1024)
    def forward(self, z, a, s, phase):
        return super().forward(z + self.phase.weight[phase][None, None], a, s)

class RawAdapter:
    def __init__(self, low_level_steps_per_ac_step=4, **kwargs):
        assert low_level_steps_per_ac_step == 4
        self.low_level_steps_per_ac_step = 4
    def state_from_observation(self, obs):
        from experiments.libero.libero_utils import quat2axisangle
        state = np.concatenate((obs['robot0_eef_pos'], quat2axisangle(obs['robot0_eef_quat']), obs['robot0_gripper_qpos'])).astype(np.float32)
        assert state.shape == (8,)
        return state
    def convert_torch(self, actions, state):
        assert actions.shape[1:] == (8, 7) and state.shape == (8,)
        return SimpleNamespace(actions=actions, states=state[None].expand(len(actions), -1))

class ARRanker:
    def __init__(self, checkpoint_path, device, dtype, compile_predictor=False,
                 encoder_image_rotation_degrees=0, **kwargs):
        assert dtype == torch.float32 and not compile_predictor
        assert encoder_image_rotation_degrees == 0
        self.device = torch.device(device)
        self.dtype = dtype
        self.encoder_image_rotation_degrees = 0
        self.last_timing = {}
        # Load the official wrapper without executing unrelated Prismatic registries.
        for package in ['prismatic','prismatic.models','prismatic.models.backbones','prismatic.models.backbones.vision']:
            if package not in sys.modules:
                module = types.ModuleType(package)
                module.__path__ = ['/data/pxchen/JEPA_WAM/'+package.replace('.', '/')]
                sys.modules[package] = module
        from prismatic.models.backbones.vision.vjepa_vit import VJEPA21ViTBackbone
        enc = ROOT/'checkpoints/jepa_wam_assets_20261009/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt'
        self.encoder = VJEPA21ViTBackbone('vjepa2_1-vit-l-384px', 'resize-naive', checkpoint_path=str(enc))
        self.encoder.to(device=self.device, dtype=dtype).eval().requires_grad_(False)
        self.predictor = ARPredictor()
        payload = torch.load(checkpoint_path, map_location='cpu', weights_only=False, mmap=True)
        assert payload['step'] == 19690, payload['step']
        self.predictor.load_state_dict(payload['model'], strict=True)
        del payload
        self.predictor.to(device=self.device, dtype=dtype).eval().requires_grad_(False)
        enable_vectorized(self.predictor)
        stats = json.loads((ROOT/'artifacts/fastwam_predictor21_ar2_stride20_20261014/train_only_stats.json').read_text())
        self.stats = {k: (torch.tensor(stats[k]['mean'],device=self.device,dtype=dtype),
                          torch.tensor(stats[k]['std'],device=self.device,dtype=dtype).clamp_min(1e-6)) for k in ('action','state')}
        self.calls = 0
        print('AR2_LOADED '+json.dumps(dict(step=19690,epoch=10,checkpoint=checkpoint_path,encoder=str(enc),precision='FP32',compile=False)), flush=True)

    def pixels(self, frame):
        image = frame if isinstance(frame, Image.Image) else Image.fromarray(np.asarray(frame, dtype=np.uint8))
        if image.width == 2*image.height:
            image = image.crop((0,0,image.height,image.height))
        # Native FastWAM camera path already applies the same rot180 as training.
        # No additional rotation; only the primary view enters JEPA.
        return self.encoder.image_transform(image.convert('RGB'))

    @torch.no_grad()
    def encode_guidance_context(self, current_clip, target_future_clips):
        assert len(target_future_clips) == 2
        with torch.autocast('cuda', enabled=False):
            current = self.pixels(current_clip[-1])[None].to(self.device, self.dtype)
            pairs = torch.stack([torch.stack([self.pixels(f) for f in pair]) for pair in target_future_clips]).to(self.device,self.dtype)
            z = self.encoder(current)
            targets = self.encoder.encode_pair(pairs).reshape(2,576,1024)
        assert z.shape == (1,576,1024) and torch.isfinite(z).all() and torch.isfinite(targets).all()
        return z.detach(), targets.detach()

    def differentiable_energy(self, current_rep, target_reps, ac_actions, ac_states):
        assert ac_actions.shape[1:] == (8,7) and ac_states.shape[1:] == (8,)
        am, ast = self.stats['action']; sm, sst = self.stats['state']
        actions = (ac_actions-am)/ast
        state = (ac_states-sm)/sst
        with torch.autocast('cuda', enabled=False):
            p1 = self.predictor(current_rep.expand(len(actions),-1,-1),actions[:,:4],state,0)
            p2 = self.predictor(p1,actions[:,4:8],state,1)
            losses = torch.stack([(p1-target_reps[0]).abs().mean((1,2)),(p2-target_reps[1]).abs().mean((1,2))],dim=1).mean(1)
        assert torch.isfinite(losses).all()
        if self.calls == 0:
            print('AR2_INTERFACE '+json.dumps(dict(current=list(current_rep.shape),targets=list(target_reps.shape),actions=list(ac_actions.shape),state=list(ac_states.shape),output=list(p2.shape),loss=float(losses.detach().mean()),autoregressive=True,extra_layer_norm=False)),flush=True)
        self.calls += 1
        return losses

def main():
    # Preserve Hydra's original file-based config lookup rather than importing
    # the evaluator as a package (its ../../configs is not a Python package).
    source = ROOT/'experiments/libero/eval_libero_single.py'
    code = source.read_text()
    needle = 'if __name__ == "__main__":\n    eval_single_process()'
    assert code.count(needle) == 1
    code = code.replace(needle, 'if __name__ == "__main__":\n    VJEPA2ACRanker = ARRanker\n    LiberoACAdapter = RawAdapter\n    eval_single_process()')
    namespace = {'__name__':'__main__','__file__':str(source),'ARRanker':ARRanker,'RawAdapter':RawAdapter}
    exec(compile(code, str(source), 'exec'), namespace)

if __name__ == '__main__':
    main()
