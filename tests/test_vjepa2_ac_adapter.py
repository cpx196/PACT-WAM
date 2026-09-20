import unittest

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from experiments.libero.vjepa2_ac_ranker import LiberoACAdapter, VJEPA2ACRanker
from fastwam.models.wan22.schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler


class LiberoACAdapterTest(unittest.TestCase):
    def test_state_from_open_gripper_observation(self):
        adapter = LiberoACAdapter()
        obs = {
            "robot0_eef_pos": np.array([0.1, -0.2, 0.3]),
            "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
            "robot0_gripper_qpos": np.array([0.04, -0.04]),
        }
        state = adapter.state_from_observation(obs)
        np.testing.assert_allclose(state, [0.1, -0.2, 0.3, 0, 0, 0, 0], atol=1e-6)

    def test_four_low_level_actions_compose_to_one_ac_transition(self):
        adapter = LiberoACAdapter(low_level_steps_per_ac_step=4)
        chunk = np.zeros((1, 4, 7), dtype=np.float32)
        chunk[0, :, 0] = 0.01
        chunk[0, :, 5] = np.pi / 8
        chunk[0, :, 6] = 1.0
        result = adapter.convert(chunk, np.zeros(7, dtype=np.float32))

        self.assertEqual(result.actions.shape, (1, 1, 7))
        self.assertEqual(result.states.shape, (1, 1, 7))
        np.testing.assert_allclose(result.actions[0, 0, :3], [0.002, 0, 0], atol=1e-6)
        expected_rotation = Rotation.from_rotvec([0, 0, np.pi / 4]).as_euler("xyz")
        np.testing.assert_allclose(result.actions[0, 0, 3:6], expected_rotation, atol=1e-6)
        self.assertAlmostEqual(float(result.actions[0, 0, 6]), 1.0, places=6)
        np.testing.assert_allclose(result.terminal_states[0], result.actions[0, 0], atol=1e-6)

    def test_state_is_integrated_across_groups(self):
        adapter = LiberoACAdapter(low_level_steps_per_ac_step=2)
        chunk = np.zeros((2, 4, 7), dtype=np.float32)
        chunk[:, :, 1] = 0.01
        chunk[:, :, 6] = -1.0
        initial = np.array([1, 2, 3, 0, 0, 0, 0.5], dtype=np.float32)
        result = adapter.convert(chunk, initial)
        self.assertEqual(result.actions.shape, (2, 2, 7))
        np.testing.assert_allclose(result.states[:, 0], np.repeat(initial[None], 2, axis=0), atol=1e-6)
        np.testing.assert_allclose(result.states[:, 1, 1], [2.001, 2.001], atol=1e-6)
        np.testing.assert_allclose(result.terminal_states[:, 1], [2.002, 2.002], atol=1e-6)
        np.testing.assert_allclose(result.terminal_states[:, 6], [0.0, 0.0], atol=1e-6)

    def test_osc_commands_are_clipped_before_metric_scaling(self):
        adapter = LiberoACAdapter(low_level_steps_per_ac_step=1)
        chunk = np.array([[[2.0, -2.0, 0.5, 2.0, 0.0, 0.0, -1.0]]], dtype=np.float32)
        result = adapter.convert(chunk, np.zeros(7, dtype=np.float32))

        np.testing.assert_allclose(result.actions[0, 0, :3], [0.05, -0.05, 0.025], atol=1e-6)
        expected_rotation = Rotation.from_rotvec([0.5, 0.0, 0.0]).as_euler("xyz")
        np.testing.assert_allclose(result.actions[0, 0, 3:6], expected_rotation, atol=1e-6)

    def test_torch_conversion_matches_existing_numpy_path(self):
        adapter = LiberoACAdapter(low_level_steps_per_ac_step=2)
        rng = np.random.default_rng(7)
        chunks = rng.uniform(-0.4, 0.4, size=(2, 4, 7)).astype(np.float32)
        initial = np.array([0.1, -0.2, 0.3, 0.1, -0.2, 0.05, 0.4], dtype=np.float32)

        expected = adapter.convert(chunks, initial)
        actual = adapter.convert_torch(torch.from_numpy(chunks), torch.from_numpy(initial))

        np.testing.assert_allclose(actual.actions.detach().numpy(), expected.actions, atol=2e-6)
        np.testing.assert_allclose(actual.states.detach().numpy(), expected.states, atol=2e-6)
        np.testing.assert_allclose(
            actual.terminal_states.detach().numpy(), expected.terminal_states, atol=2e-6
        )

    def test_torch_conversion_retains_action_gradient(self):
        adapter = LiberoACAdapter(low_level_steps_per_ac_step=2)
        chunks = torch.full((1, 4, 7), 0.1, dtype=torch.float64, requires_grad=True)
        result = adapter.convert_torch(chunks, torch.zeros(7, dtype=torch.float64))

        loss = result.actions.square().sum() + result.states.square().sum()
        gradient = torch.autograd.grad(loss, chunks)[0]

        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.abs().sum()), 0.0)


class GuidedSchedulerStepTest(unittest.TestCase):
    def test_guidance_is_explicit_gradient_descent_when_delta_is_negative(self):
        sample = torch.tensor([[[1.0]]])
        velocity = torch.tensor([[[2.0]]])
        gradient = torch.tensor([[[3.0]]])
        delta = torch.tensor(-0.1)

        base = WanContinuousFlowMatchScheduler.step(velocity, delta, sample)
        guided = WanContinuousFlowMatchScheduler.step_with_loss_guidance(
            velocity, delta, sample, gradient, guidance_step_size=0.05
        )

        torch.testing.assert_close(base, torch.tensor([[[0.8]]]))
        torch.testing.assert_close(guided, base - 0.05 * gradient)


class VJEPA2ACRankerValidationTest(unittest.TestCase):
    def test_encode_clips_preserves_two_distinct_temporal_frames(self):
        class RecordingEncoder(torch.nn.Module):
            def forward(self, clips):
                self.last_input = clips.detach().clone()
                return torch.zeros((clips.shape[0], 256, 4), dtype=clips.dtype)

        ranker = object.__new__(VJEPA2ACRanker)
        ranker.device = torch.device("cpu")
        ranker.dtype = torch.float32
        ranker.encoder = RecordingEncoder()
        black = np.zeros((16, 16, 3), dtype=np.uint8)
        white = np.full((16, 16, 3), 255, dtype=np.uint8)

        reps = ranker.encode_clips([[black, white]])

        self.assertEqual(tuple(reps.shape), (1, 256, 4))
        self.assertEqual(tuple(ranker.encoder.last_input.shape), (1, 3, 2, 256, 256))
        self.assertFalse(
            torch.equal(
                ranker.encoder.last_input[:, :, 0],
                ranker.encoder.last_input[:, :, 1],
            )
        )

    def test_rejects_rollout_longer_than_initialized_attention_mask(self):
        ranker = object.__new__(VJEPA2ACRanker)
        ranker.device = torch.device("cpu")
        ranker.dtype = torch.float32
        ranker.max_ac_steps = 2
        actions = np.zeros((2, 3, 7), dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "must be in"):
            ranker.rank(
                current_clip=[np.zeros((2, 2, 3), dtype=np.uint8)] * 2,
                target_future_clips=[[np.zeros((2, 2, 3), dtype=np.uint8)] * 2] * 3,
                ac_actions=actions,
                ac_states=actions,
            )

    def test_differentiable_energy_retains_action_and_state_gradients(self):
        class ActionSensitivePredictor(torch.nn.Module):
            def forward(self, history, actions, states):
                del history
                value = actions.sum(dim=(1, 2)) + states.sum(dim=(1, 2))
                features = torch.stack(
                    (value, torch.ones_like(value), 2 * torch.ones_like(value), 3 * torch.ones_like(value)),
                    dim=-1,
                )
                return features[:, None].expand(-1, 2, -1)

        ranker = object.__new__(VJEPA2ACRanker)
        ranker.device = torch.device("cpu")
        ranker.dtype = torch.float32
        ranker.max_ac_steps = 2
        ranker.tokens_per_frame = 2
        ranker.predictor = ActionSensitivePredictor()
        actions = torch.full((1, 2, 7), 0.1, requires_grad=True)
        states = torch.full((1, 2, 7), 0.2, requires_grad=True)

        energy = ranker.differentiable_energy(
            current_rep=torch.zeros(2, 4),
            target_reps=torch.zeros(2, 2, 4),
            ac_actions=actions,
            ac_states=states,
        ).sum()
        action_gradient, state_gradient = torch.autograd.grad(energy, (actions, states))

        self.assertTrue(torch.isfinite(action_gradient).all())
        self.assertTrue(torch.isfinite(state_gradient).all())
        self.assertGreater(float(action_gradient.abs().sum()), 0.0)
        self.assertGreater(float(state_gradient.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
