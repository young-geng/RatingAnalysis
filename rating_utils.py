from __future__ import division

import numpy as np
import pandas as pd
import scipy.stats


# Basic utilities.

def get_teams(data):
    return data['Visitor'].append(data['Home']).unique()


class RatingAlgorithm(object):
    """Base class for rating algorithm."""

    def __init__(self):
        raise NotImplementedError()

    def fit(self, data, teams):
        raise NotImplementedError()

    def predict(self, team1, team2):
        raise NotImplementedError()

    def test_error_rate(self, data):
        n_errors = 0
        for i in xrange(len(data)):
            if (data['HomePTS'][i] >= data['VisitorPTS'][i]
                and self.predict(data['Home'][i], data['Visitor'][i]) < 0.5):
                n_errors += 1
        return n_errors / len(data)

    @property
    def ratings(self):
        raise NotImplementedError()


class MasseyMethod(RatingAlgorithm):
    """Messy's method for rating."""

    def __init__(self):
        pass

    def _count_match(self):
        self._match_result_count = np.zeros((self._n_teams, self._n_teams))

        for i in xrange(len(self._data)):

            if self._data['HomePTS'][i] >= self._data['VisitorPTS'][i]:
                self._match_result_count[
                    self._team_idx[self._data['Home'][i]],
                    self._team_idx[self._data['Visitor'][i]]
                ] += 1

            if self._data['HomePTS'][i] <= self._data['VisitorPTS'][i]:
                self._match_result_count[
                    self._team_idx[self._data['Visitor'][i]],
                    self._team_idx[self._data['Home'][i]]
                ] += 1

        return self._match_result_count

    def _count_score(self, team):
        total = 0
        for i in xrange(len(self._data)):
            if self._data['Home'][i] == team:
                total += self._data['HomePTS'][i] - self._data['VisitorPTS'][i]

            elif self._data['Visitor'][i] == team:
                total += self._data['VisitorPTS'][i] - self._data['HomePTS'][i]

        return total

    def _correct_probability(self, p):
        epsilon = 0.01
        if p == 0:
            return p + epsilon
        elif p == 1:
            return p - epsilon
        return p

    def fit(self, data, teams=None):

        if teams is None:
            teams = get_teams(data)

        self._team_names = teams
        self._data = data
        self._n_teams = len(teams)

        self._team_idx = {}
        for i in xrange(len(teams)):
            self._team_idx[teams[i]] = i

        # Compute Massey's matrix
        M = np.zeros((self._n_teams, self._n_teams))

        for i in xrange(len(self._data)):
            M[self._team_idx[self._data['Home'][i]], self._team_idx[self._data['Visitor'][i]]] -= 1
            M[self._team_idx[self._data['Visitor'][i]], self._team_idx[self._data['Home'][i]]] -= 1

        for i in xrange(self._n_teams - 1):
            M[i, i] = -np.sum(M[i, :])

        M[self._n_teams - 1, :] = 1

        # Compute score difference vector
        p = np.zeros((self._n_teams, 1))

        for i in xrange(self._n_teams - 1):
            p[i, 0] = self._count_score(self._team_names[i])

        self._ratings = np.dot(np.linalg.inv(M), p).flatten()
        self._estimate_variance()

    @property
    def ratings(self):
        return self._ratings


    def _estimate_variance(self):
        s = []
        for i in xrange(len(self._data)):
            t1 = self._team_idx[self._data['Home'][i]]
            t2 = self._team_idx[self._data['Visitor'][i]]
            score_diff = self._data['HomePTS'][i] - self._data['VisitorPTS'][i]
            s.append(score_diff - (self._ratings[t1] - self._ratings[t2]))

        s = np.array(s)

        # Unbiased estimator for variance
        self._gaussian_variance = np.sum(s ** 2) / (s.shape[0] - 1)

    def predict(self, team1, team2):
        rating_diff = (self._ratings[self._team_idx[team1]]
            - self._ratings[self._team_idx[team2]])

        # Compute probabilty with gaussian CDF
        return scipy.stats.norm.cdf(rating_diff, 0, self._gaussian_variance ** 0.5)
