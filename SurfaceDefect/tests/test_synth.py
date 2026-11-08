import cv2
import numpy as np
import pytest

from partinspect import synth


def fake_part(h=803, w=1163):
    """White background with a blue textured frame, roughly like the real photos."""
    rng = np.random.default_rng(0)
    img = np.full((h, w, 3), 250, np.uint8)
    part = np.clip(rng.normal((170, 80, 60), 20, (h, w, 3)), 0, 255).astype(np.uint8)
    frame = np.zeros((h, w), bool)
    frame[20:h - 20, 20:180] = frame[20:h - 20, w - 180:w - 20] = True
    frame[20:150, 20:w - 20] = frame[h - 150:h - 20, 20:w - 20] = True
    img[frame] = part[frame]
    return img


@pytest.mark.parametrize("kind", synth.TYPES)
def test_changes_only_inside_mask(kind):
    img = fake_part()
    bad, mask = synth.apply(img, kind, np.random.default_rng(1))
    assert mask.dtype == bool and 20 < mask.sum() < 0.05 * mask.size
    assert np.array_equal(bad[~mask], img[~mask])
    assert np.abs(bad[mask].astype(int) - img[mask].astype(int)).mean() > 3


def test_defects_stay_on_the_part():
    img = fake_part()
    for kind in ("scratch", "spot", "burn", "edge_chip"):
        _, mask = synth.apply(img, kind, np.random.default_rng(2))
        assert synth.foreground(img)[mask].all()


def test_normalize_matches_reference_part_colour():
    from partinspect.normalize import REF_BGR, normalize

    img = np.full((60, 80, 3), 250, np.uint8)
    img[10:50, 10:70] = (120, 70, 60)
    out = normalize(img)
    part = out[10:50, 10:70].reshape(-1, 3).mean(0)
    assert np.allclose(part, REF_BGR, atol=1)
    assert (out[0, 0] == 255).all()  # white background stays white (clipped)
