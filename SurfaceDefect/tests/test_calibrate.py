from partinspect.calibrate import blocks


def test_blocks_cover_pool_contiguously():
    bs = blocks(231, 5)
    assert [len(b) for b in bs] == [46, 46, 47, 46, 46]
    assert [i for b in bs for i in b] == list(range(231))
