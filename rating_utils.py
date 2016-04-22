from __future__ import division

import numpy as np
import pandas as pd


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
        self._optimize_logistic_parameter()

    @property
    def ratings(self):
        return self._ratings

    def _optimize_logistic_parameter(self):
        x = []
        p = []

        matches = self._count_match()
        for i in xrange(self._n_teams):
            for j in xrange(i):
                w1 = matches[i, j]
                w2 = matches[j, i]
                if w1 + w2 == 0:
                    # Haven't had a match in the training set
                    continue
                x.append(self._ratings[i] - self._ratings[j])
                p.append(self._correct_probability(w1 / (w1 + w2)))

        x = np.array(x)
        p = np.array(p)

        y = -np.log(1 / p - 1)


        self._logistic_scale = np.sum(x * y) / np.sum(x * x)

    @property
    def logistic_scale(self):
        return self._logistic_scale

    def _logistic_probability(self, x):
        return 1 / (1 + np.exp(-x * self._logistic_scale))

    def predict(self, team1, team2):
        return self._logistic_probability(
            self._ratings[self._team_idx[team1]]
            - self._ratings[self._team_idx[team2]]
        )
