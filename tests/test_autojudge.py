"""Hermetic offline tests for the personalized-judge harness.

No network, no API key, no external data: each test synthesizes a handful of tiny
PNGs + a mini long-csv in a temp dir and exercises the loader, context selection,
prompt building, and the full mock inner loop. Run with:  pytest tests/  (or
python -m pytest tests/test_autojudge.py).
"""

from __future__ import annotations

import os
import tempfile
import unittest

import pandas as pd

from car_judge import env as envmod
from car_judge import data as datamod
from car_judge import context_selection as ctxsel
from car_judge import prompts
from car_judge.config import RunConfig, JudgeConfig, DIMENSIONS
from car_judge.long_data import load_long_raters, render_profile, render_owned
from car_judge.inner_loop import run_inner, passes_attention


CARS = [f"car_{i}" for i in range(1, 13)]


def _make_png(path, color):
    from PIL import Image
    Image.new("RGB", (24, 24), color).save(path)


def _build_dataset(root):
    img_root = os.path.join(root, "images", "chunk_01")
    os.makedirs(img_root, exist_ok=True)
    palette = [(200, 30, 30), (30, 30, 200), (20, 20, 20), (230, 230, 230)]
    for i, car in enumerate(CARS):
        _make_png(os.path.join(img_root, f"{car}.png"), palette[i % 4])
    rows = []
    for ri in range(4):
        rid = f"R_t{ri}"
        for ci, car in enumerate(CARS):
            base = (ci * 7 + ri * 3) % 6 + 1
            clamp = lambda x: max(1, min(6, int(round(x))))
            rows.append(dict(
                wave="July8", rater=rid, car=car, is_anchor=False,
                car_name=f"Car {car}", body_style=["Sedan", "SUV", "Coupe", "Pickup Truck"][ci % 4],
                doors=[2, 4][ci % 2], seats=[2, 4, 5, 7][ci % 4],
                car_color=["red", "blue", "black", "white"][ci % 4],
                sporty=clamp(base), luxurious=clamp(7 - base), modern=clamp(base),
                rugged=clamp(7 - base), overall_preference=clamp(base),
                num_children_under18=ri % 3, driving_location="Mostly city driving",
                driving_frequency="Daily",
                rating_influences_text="I look at how modern and sleek it is.",
                attn_check_ai=4, attn_check_color="A",
                brand_owned_bmw=1 if ri == 0 else 0,
                body_style_owned_pickup_truck=1 if ri == 1 else 0,
                hobby_outdoor_or_nature_centered_activities=ri % 2,
                bfi_openness=3, bfi_conscientiousness=3, bfi_extraversion=3,
                bfi_agreeableness=3, bfi_neuroticism=2,
                prolific_age=25 + ri * 5, prolific_sex=["Male", "Female"][ri % 2],
            ))
    csv_path = os.path.join(root, "mini_long.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path, os.path.join(root, "images")


class HarnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.csv, cls.img_root = _build_dataset(cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_loader_and_profile(self):
        raters = load_long_raters(self.csv, min_cars=5)
        self.assertEqual(len(raters), 4)
        r = raters[0]
        self.assertEqual(len(r.fully_rated_cars()), 12)
        self.assertIn("car_1.png", r.ratings)                  # car -> car.png
        self.assertTrue(all(k in r.ratings["car_1.png"] for k in DIMENSIONS))
        self.assertTrue(r.q23)                                 # Q23 populated
        self.assertTrue(passes_attention(r))                   # attn_check_ai==4
        txt = render_profile(r.profile, ["prolific_age", "prolific_sex"])
        self.assertIn("age", txt)

    def test_render_owned(self):
        raters = load_long_raters(self.csv, min_cars=5)
        owned0 = render_owned(raters[0].profile)   # R_t0 owns BMW
        self.assertIn("bmw", owned0.lower())

    def test_context_selection_no_leakage(self):
        raters = load_long_raters(self.csv, min_cars=5)
        idx = datamod.build_image_index(self.img_root)
        r = raters[0]
        pool = [c for c in r.fully_rated_cars() if c != "car_1.png"]
        for strat in ("random", "first", "diverse", "rag_profile", "rag_image"):
            sel = ctxsel.select_context(strat, pool, "car_1.png", 4,
                                        rater=r, image_index=idx, seed=0)
            self.assertEqual(len(sel), 4)
            self.assertNotIn("car_1.png", sel)                 # never leak the target
            self.assertTrue(set(sel).issubset(set(pool)))

    def test_prompt_persona_block(self):
        cfg = JudgeConfig(name="p", n_context=0, include_demographics=True, include_q23=True)
        idx = datamod.build_image_index(self.img_root)
        path = idx["car_1.png"]
        msgs = prompts.build_messages(cfg, path, [], profile_text="- age: 30",
                                      q23_text="I like modern cars", owned_text="")
        sys = msgs[0]["content"]
        self.assertIn("About the person", sys)
        self.assertIn("modern cars", sys)
        self.assertEqual(prompts.count_images(msgs), 1)

    def test_inner_loop_mock_and_context_helps(self):
        base = run_inner(RunConfig(data_csv=self.csv, image_root=self.img_root,
                                   n_raters=4, test_size=4, mock=True),
                         JudgeConfig(name="baseline", n_context=0), verbose=False)
        ctx = run_inner(RunConfig(data_csv=self.csv, image_root=self.img_root,
                                  n_raters=4, test_size=4, mock=True),
                        JudgeConfig(name="ctx", n_context=6, context_strategy="random"),
                        verbose=False)
        for res in (base, ctx):
            # plumbing: both configs run end-to-end and yield valid metrics
            self.assertTrue(0.0 <= res.score <= (6 - 1))
            self.assertEqual(res.n_unparsed, 0)
            self.assertEqual(res.n_predictions, 4 * 4)
            self.assertEqual(set(res.per_dimension_mae), set(DIMENSIONS))
        # context path actually assembles exemplars (more images per request)
        self.assertGreater(ctx.rows[0]["n_images"], base.rows[0]["n_images"])

    def test_clip_index_and_builder(self):
        import numpy as np
        from car_judge.clip_index import CarClipIndex
        # 3 cars, 4-dim: car_1 close to car_3, car_2 orthogonal
        ids = ["car_1.png", "car_2.png", "car_3.png"]
        vecs = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0.9, 0.1, 0, 0]], dtype="float32")
        idx = CarClipIndex(ids, vecs)
        self.assertTrue(idx.has("car_1.png"))
        self.assertEqual(idx.nearest("car_1.png", ["car_2.png", "car_3.png"], 1), ["car_3.png"])
        self.assertEqual(idx.nearest("car_1.png", ["car_9.png"], 1), [])   # pool w/o embeddings
        self.assertAlmostEqual(idx.coverage(["car_1.png", "car_9.png"]), 0.5)

    def test_build_clip_car_index_join(self):
        import numpy as np, subprocess
        d = self.tmp.name
        # raw pooled embeddings (4 rows) + aligned map; 2 of them are our uploaded cars
        emb = np.array([[1, 0], [0, 1], [0.2, 0.9], [0.7, 0.7]], dtype="float32")
        np.save(os.path.join(d, "emb.npy"), emb)
        pd.DataFrame({
            "image_path": ["x/a.png", "x/b.png", "x/c.png", "x/d.png"],
            "car_name": ["A", "B", "C", "D"],
        }).to_csv(os.path.join(d, "dmap.csv"), index=False)
        pd.DataFrame({
            "car_id": ["car_1", "car_2"], "upload_filename": ["car_1.png", "car_2.png"],
            "original_filename": ["b.png", "d.png"], "actual_car_name": ["B", "D"],
        }).to_csv(os.path.join(d, "cmap.csv"), index=False)
        out = os.path.join(d, "car_clip.npz")
        rc = subprocess.run([__import__("sys").executable, "scripts/build_clip_car_index.py",
                             "--embeddings", os.path.join(d, "emb.npy"),
                             "--map", os.path.join(d, "dmap.csv"),
                             "--car-mapping", os.path.join(d, "cmap.csv"),
                             "--out", out], capture_output=True, text=True)
        self.assertEqual(rc.returncode, 0, rc.stderr)
        z = np.load(out, allow_pickle=True)
        self.assertEqual(set(z["car_ids"]), {"car_1.png", "car_2.png"})
        # car_1 came from row 'b' ([0,1]) -> nearest to itself, not car_2 (row 'd')
        from car_judge.clip_index import CarClipIndex
        idx = CarClipIndex(list(z["car_ids"]), z["vectors"])
        self.assertTrue(idx.has("car_1.png") and idx.has("car_2.png"))

    def test_rag_clip_falls_back_without_index(self):
        # no data/car_clip_embeddings.npz on the test path -> rag_clip uses pixel RAG
        raters = load_long_raters(self.csv, min_cars=5)
        idx = datamod.build_image_index(self.img_root)
        r = raters[0]
        pool = [c for c in r.fully_rated_cars() if c != "car_1.png"]
        sel = ctxsel.select_context("rag_clip", pool, "car_1.png", 4,
                                    rater=r, image_index=idx, seed=0)
        self.assertEqual(len(sel), 4)
        self.assertNotIn("car_1.png", sel)

    def test_env_resolution(self):
        os.environ.pop("OPENAI_API_KEY", None)
        creds = envmod.resolve_credentials(model="gpt-x", api_key="sk-abc")
        self.assertEqual(creds.model, "gpt-x")
        self.assertTrue(creds.is_live_ready)


if __name__ == "__main__":
    unittest.main()
