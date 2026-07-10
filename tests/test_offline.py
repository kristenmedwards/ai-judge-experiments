"""Offline unit tests — no VLM server required. Run with: python -m pytest -q
(or plain `python tests/test_offline.py`)."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from car_judge.parsing import parse_ratings
from car_judge.config import DIMENSIONS
from car_judge import prompts, metrics


def test_parse_clean_json():
    txt = '{"sporty": 4, "luxurious": 2, "modern": 5, "rugged": 1, "preference": 3}'
    r = parse_ratings(txt)
    assert r == {"sporty": 4, "luxurious": 2, "modern": 5, "rugged": 1, "preference": 3}


def test_parse_fenced_and_pretext():
    txt = "Sure! Here you go:\n```json\n{\"sporty\":6,\"luxurious\":6," \
          "\"modern\":6,\"rugged\":6,\"preference\":6}\n```"
    r = parse_ratings(txt)
    assert all(r[d] == 6 for d in DIMENSIONS)


def test_parse_clamps_out_of_range():
    txt = '{"sporty": 9, "luxurious": 0, "modern": 3, "rugged": 3, "preference": 3}'
    r = parse_ratings(txt)
    assert r["sporty"] == 6 and r["luxurious"] == 1


def test_parse_regex_fallback():
    txt = "sporty: 3, luxurious = 4\nmodern 2 (ignore), rugged: 5, preference: 1"
    r = parse_ratings(txt)
    assert r["sporty"] == 3 and r["rugged"] == 5 and r["preference"] == 1


def test_message_builders_image_order():
    # image part must come before its ratings text in each exemplar.
    import base64, tempfile
    png = base64.b64decode(
        b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42m"
        b"NkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==")
    with tempfile.TemporaryDirectory() as d:
        a = os.path.join(d, "a.png"); b = os.path.join(d, "b.png"); t = os.path.join(d, "t.png")
        for p in (a, b, t):
            open(p, "wb").write(png)
        msgs = prompts.build_in_context_messages(
            [(a, {"sporty": 1, "luxurious": 1, "modern": 1, "rugged": 1, "preference": 1}),
             (b, {"sporty": 6, "luxurious": 6, "modern": 6, "rugged": 6, "preference": 6})],
            t)
        content = msgs[1]["content"]
        # first two content parts: image then text
        assert content[0]["type"] == "image_url"
        assert content[1]["type"] == "text"
        # 3 images total (2 exemplars + target)
        assert prompts.count_images(msgs) == 3

        msgs0 = prompts.build_no_context_messages(t)
        assert prompts.count_images(msgs0) == 1


def test_metrics():
    recs = [(3, 3), (4, 5), (2, 4)]  # errors 0,1,2
    assert abs(metrics.mae(recs) - 1.0) < 1e-9
    assert abs(metrics.exact_accuracy(recs) - 1 / 3) < 1e-9
    assert abs(metrics.within_one(recs) - 2 / 3) < 1e-9


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok:", fn.__name__)
    print(f"\nall {len(fns)} tests passed")
