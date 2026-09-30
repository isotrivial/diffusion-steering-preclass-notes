"""CPU regressions for guidance, solver semantics, NFE and frozen scoring."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
import benchmark
import common
import fidm
import sample


class LabelNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.images_seen = 0

    def forward(self, x, t, y):
        self.images_seen += len(x)
        return torch.ones_like(x) * (y[:, None, None, None] / 20)


class SamplerTests(unittest.TestCase):
    def test_guidance_endpoints(self):
        for w, expected in ((0, .5), (1, .1), (2, -.3)):
            x, nfe = sample.sample(LabelNet(), 10, torch.full((10,), 2), "cpu", steps=1, w=w)
            torch.testing.assert_close(x, torch.full_like(x, expected))
            self.assertEqual(nfe, 1 if w in (0, 1) else 2)

    def test_nfe_counts_actual_images_across_chunks(self):
        for solver, evaluations in (("ddim", 4), ("euler", 4), ("heun", 7)):
            for w, branches in ((0, 1), (1, 1), (2, 2)):
                net = LabelNet()
                _, nfe = sample.sample(net, 20, torch.arange(20) % 10, "cpu",
                                       steps=4, w=w, order=solver, batch=7)
                self.assertEqual(nfe, branches * evaluations)
                self.assertEqual(net.images_seen, 20 * nfe)

    def test_heun_integrates_velocity(self):
        class LinearVelocity(torch.nn.Module):
            def forward(self, x, t, y):
                self.last_input = x.clone()
                a, s = common.alpha_sigma(t)
                ad, sd = common.alpha_sigma_dot(t)
                shape = (-1, 1, 1, 1)
                # Construct a denoiser whose implied velocity is v(t)=2t.
                return (s.reshape(shape) * (2*t).reshape(shape) - sd.reshape(shape)*x) / (
                    ad*s - sd*a).reshape(shape)
        states = {}
        for order in ("euler", "heun"):
            net = LinearVelocity()
            sample._sample_chunk(net, torch.zeros(1, 1, 28, 28), torch.zeros(1).long(),
                                 "cpu", 4, 1., 0., order, 0)
            states[order] = net.last_input
        torch.testing.assert_close(states["heun"], torch.full_like(states["heun"], .75**2))
        self.assertGreater((states["euler"] - .75**2).abs().max(), .1)

    def test_repeatability_and_initial_noise_pairing(self):
        class NoiseNet(torch.nn.Module):
            def forward(self, x, t, y):
                return .05*x + t[:, None, None, None].square()/10
        net = NoiseNet()
        labels = torch.arange(10)
        a, na = sample.sample(net, 10, labels, "cpu", steps=3, order="heun", batch=10)
        b, nb = sample.sample(net, 10, labels, "cpu", steps=3, order="heun", batch=4)
        torch.testing.assert_close(a, b, atol=0, rtol=0)
        self.assertEqual(na, nb)
        c, _ = sample.sample(net, 10, labels, "cpu", steps=3, eta=.5, batch=4)
        d, _ = sample.sample(net, 10, labels, "cpu", steps=3, eta=.5, batch=4)
        torch.testing.assert_close(c, d, atol=0, rtol=0)
        different, _ = sample.sample(net, 10, labels, "cpu", steps=3, seed=1, batch=4)
        self.assertFalse(torch.equal(c, different))

    def test_invalid_settings_fail(self):
        for config in ({"order": "typo"}, {"steps": 0}, {"eta": 1.1},
                       {"eta": .1, "order": "heun"}, {"batch": 0}):
            with self.assertRaises(ValueError):
                sample.sample(LabelNet(), 10, torch.arange(10), "cpu", **config)


class MetricTests(unittest.TestCase):
    def test_gaussian_distance(self):
        s1 = (np.zeros(2), np.diag([1., 4.]))
        s2 = (np.array([1., 2.]), np.diag([4., 9.]))
        self.assertAlmostEqual(fidm.frechet(s1, s2), 7.)
        self.assertAlmostEqual(fidm.frechet(s1, s1), 0.)
        self.assertAlmostEqual(fidm.frechet(s1, s2), fidm.frechet(s2, s1))

    def test_duplicates_and_blockwise_distances(self):
        f = np.random.default_rng(0).normal(size=(32, 5))
        full = torch.cdist(torch.tensor(f), torch.tensor(f))
        full.fill_diagonal_(float("inf"))
        torch.testing.assert_close(fidm.nearest_distances(f, batch=7), full.min(1).values)
        self.assertEqual(fidm.coverage(np.repeat(f[:1], 100, axis=0), .01), 0.)

    def test_changed_release_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / "weights.pt"
            weights.write_bytes(b"frozen")
            (root / "release.json").write_text(json.dumps({"sha256": {"weights.pt": common.sha256(weights)}}))
            benchmark.verify_release(root)
            weights.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                benchmark.verify_release(root)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
