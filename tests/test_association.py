import numpy as np

import locate_event as le
from helpers import C, FS, square_xyz


class TestMaxLag:
    def test_matrix_properties(self):
        XYZ = square_xyz(20.0)
        L = le.max_pairwise_lag(XYZ, C, slack_s=0.005)
        assert L.shape == (4, 4) and np.allclose(L, L.T)
        assert np.allclose(np.diag(L), 0.005)
        assert abs(L[0, 1] - (np.linalg.norm(XYZ[0] - XYZ[1]) / C + 0.005)) < 1e-12

    def test_clock_prior_widens_gate(self):
        XYZ = square_xyz(20.0)
        a = le.max_pairwise_lag(XYZ, C, 0.0, clock_sigma_s=0.0)
        b = le.max_pairwise_lag(XYZ, C, 0.0, clock_sigma_s=0.002)
        assert np.allclose(b - a, 0.006)


class TestAssociate:
    def setup_method(self):
        self.XYZ = square_xyz(20.0)
        self.L = le.max_pairwise_lag(self.XYZ, C, 0.005)
        self.true = (np.array([1.0, 1.02, 1.03, 1.015]) * FS).astype(int)

    def test_all_consistent(self):
        cands = [[(k, 10.0)] for k in self.true]
        chosen, missing = le.associate_onsets(cands, FS, self.L)
        assert missing == [] and [chosen[i] for i in range(4)] == list(self.true)

    def test_strong_decoy_rejected_for_weaker_consistent_onset(self):
        cands = [[(k, 10.0)] for k in self.true]
        cands[1] = [(int(4.0 * FS), 50.0), (self.true[1], 6.0)]  # loud horn 3 s later, weak true onset
        chosen, missing = le.associate_onsets(cands, FS, self.L)
        assert missing == [] and chosen[1] == self.true[1]

    def test_decoy_is_top_candidate_of_strongest_track(self):
        cands = [[(k, 10.0)] for k in self.true]
        cands[0] = [(int(4.0 * FS), 80.0), (self.true[0], 12.0)]
        chosen, missing = le.associate_onsets(cands, FS, self.L)
        assert missing == [] and chosen[0] == self.true[0]

    def test_track_without_consistent_onset_is_dropped(self):
        cands = [[(k, 10.0)] for k in self.true]
        cands[2] = [(int(7.0 * FS), 30.0)]
        chosen, missing = le.associate_onsets(cands, FS, self.L)
        assert missing == [2] and set(chosen) == {0, 1, 3}

    def test_empty_candidates(self):
        chosen, missing = le.associate_onsets([[], [], [], []], FS, self.L)
        assert chosen == {} and missing == [0, 1, 2, 3]
        chosen, missing = le.associate_onsets([[(self.true[0], 5.0)], [], [], []], FS, self.L)
        assert chosen == {0: int(self.true[0])} and missing == [1, 2, 3]

    def test_pairwise_consistency_enforced(self):
        # track 1 and 2 are each within the gate of the anchor but not of each other
        L = np.full((3, 3), 0.010)
        np.fill_diagonal(L, 0.0)
        cands = [[(1000, 10.0)], [(1000 + int(0.009 * FS), 5.0)], [(1000 - int(0.009 * FS), 8.0)]]
        chosen, missing = le.associate_onsets(cands, FS, L)
        assert 0 in chosen and len(chosen) == 2 and missing == [1]
