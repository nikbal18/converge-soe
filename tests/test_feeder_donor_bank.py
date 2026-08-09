"""The donor pool must be feeder-wide even when --only names one substation.

``allow_feeder_donors`` promises a pool drawn from every metered NMI on the
feeder. It builds that pool from ``bundles``, and ``run_feeder --only`` with a
single substation puts exactly ONE substation in ``bundles`` — so the promise
quietly becomes "this substation's own meters", which for a thinly-metered
substation is below ``min_donors`` and falls through to the disaggregation
fallback.

``feeder_donor_bank`` closes that gap by feeding in the sibling ``.npz``
bundles. These tests pin the three properties that matter: the extras widen
the pool, synthetic columns never become donors, and a mismatched time grid is
skipped rather than crashing ``np.stack``.
"""

import numpy as np

from converge_soe import synthetic as syn


def _bundle(ids, T=48, complete=True, synthetic=None, seed=0):
    rng = np.random.default_rng(seed)
    n = len(ids)
    P = rng.normal(1000.0, 100.0, (T, n)).astype(np.float32)
    Q = (P * 0.4).astype(np.float32)
    mask = np.ones((T, n), dtype=bool)
    if not complete:
        mask[0, :] = False
    b = {"load_ids": np.array(ids, dtype="<U32"), "P": P, "Q": Q,
         "mask": mask, "dt_minutes": np.float64(30.0)}
    if synthetic is not None:
        b["synthetic"] = np.array(synthetic, dtype=bool)
    return b


def test_extras_widen_the_pool():
    own = {"S_A": _bundle(["nmi_1", "nmi_2"], seed=1)}
    extra = {"S_B": _bundle([f"nmi_{i}" for i in range(10, 20)], seed=2)}

    narrow = syn._donor_bank(own, log=lambda *a: None)
    wide = syn._donor_bank(own, extra, log=lambda *a: None)

    assert len(narrow[0]) == 2
    assert len(wide[0]) == 12
    # the substation's own profiles come first and are unchanged
    assert wide[0][:2] == narrow[0]
    assert np.array_equal(wide[1][:, :2], narrow[1])


def test_synthetic_columns_are_never_donors():
    """Otherwise the second pass would resample the first pass's inventions."""
    own = {"S_A": _bundle(["nmi_1"], seed=1)}
    extra = {"S_B": _bundle(["nmi_10", "nmi_11", "nmi_12"], seed=2,
                            synthetic=[False, True, True])}

    ids, P, Q = syn._donor_bank(own, extra, log=lambda *a: None)
    assert ids == ["nmi_1", "nmi_10"]


def test_incomplete_columns_are_never_donors():
    own = {"S_A": _bundle(["nmi_1"], seed=1)}
    extra = {"S_B": _bundle(["nmi_10", "nmi_11"], seed=2, complete=False)}

    ids, _, _ = syn._donor_bank(own, extra, log=lambda *a: None)
    assert ids == ["nmi_1"]


def test_mismatched_time_grid_is_skipped_not_fatal():
    """pre_index sizes each substation's grid from its own NMIs' coverage."""
    own = {"S_A": _bundle(["nmi_1"], T=48, seed=1)}
    extra = {"S_B": _bundle(["nmi_10"], T=96, seed=2),
             "S_C": _bundle(["nmi_20"], T=48, seed=3)}

    ids, P, _ = syn._donor_bank(own, extra, log=lambda *a: None)
    assert ids == ["nmi_1", "nmi_20"]
    assert P.shape == (48, 2)


def test_duplicate_ids_are_not_double_counted():
    own = {"S_A": _bundle(["nmi_1", "nmi_2"], seed=1)}
    extra = {"S_B": _bundle(["nmi_2", "nmi_3"], seed=2)}

    ids, _, _ = syn._donor_bank(own, extra, log=lambda *a: None)
    assert ids == ["nmi_1", "nmi_2", "nmi_3"]


def test_empty_extras_behaves_like_before():
    own = {"S_A": _bundle(["nmi_1", "nmi_2"], seed=1)}
    assert syn._donor_bank(own, {}, log=lambda *a: None)[0] == \
        syn._donor_bank(own, log=lambda *a: None)[0]


def test_no_donors_at_all_returns_none():
    own = {"S_A": _bundle(["nmi_1"], complete=False, seed=1)}
    assert syn._donor_bank(own, log=lambda *a: None) is None
