import numpy as np
from sklearn.model_selection import train_test_split


class MAB_2_ARMS:

    def __init__(self, weights=None, alpha=0.05):
        if weights is None:
            self.weights = np.empty(1)
        else:
            self.weights = weights
        self.alpha = alpha
        self.trained = False

    def fit(self, data, n_loops=1):
        cols_with_1 = [col for col in data.columns if ' 1' in col and not 'Win' in col]
        cols_with_2 = [col for col in data.columns if ' 2' in col]
        other_cols = [col for col in data.columns if col not in cols_with_1 + cols_with_2]

        sorted_cols = cols_with_1 + cols_with_2 + other_cols
        data = data[sorted_cols]

        train, test = train_test_split(data, test_size=0.2, random_state=42, shuffle=True)

        # get the win rate out of the data
        # the oracle value represent the "reward" an optimal user would get for a given context
        train_oracle_value = data.iloc[:, -1].to_numpy()
        train_oracle_value[train_oracle_value < 0.5] = 1 - train_oracle_value[train_oracle_value < 0.5]

        batch_size = 10

        other_cols_wo_winrate = [col for col in other_cols if not 'Win' in col]
        data_2_arm_a1 = cols_with_1 + other_cols_wo_winrate
        data_2_arm_a1 = train[data_2_arm_a1].to_numpy()

        data_2_arm_a2 = cols_with_2 + other_cols_wo_winrate
        data_2_arm_a2 = train[data_2_arm_a2].to_numpy()

        win_rate_agent1 = train['Win rate of Agent 1'].to_numpy()
        win_rate_agent2 = 1 - win_rate_agent1

        agent1_general_context = np.concatenate(
            (np.zeros_like(data_2_arm_a2), data_2_arm_a2, train[cols_with_1].to_numpy()),
            axis=1).T
        agent2_general_context = np.concatenate(
            (data_2_arm_a1, np.zeros_like(data_2_arm_a1), train[cols_with_2].to_numpy()),
            axis=1).T

        shuffled_arrays = shuffle_arrays_in_unison(
            [agent1_general_context.T, agent2_general_context.T, win_rate_agent1, win_rate_agent2])
        agent1_general_context = shuffled_arrays[0]
        agent2_general_context = shuffled_arrays[1]
        win_rate_agent1 = shuffled_arrays[2].reshape(-1, 1)
        win_rate_agent2 = shuffled_arrays[3].reshape(-1, 1)

        agent1_batches = split_array_into_batches(agent1_general_context, batch_size)
        agent2_batches = split_array_into_batches(agent2_general_context, batch_size)
        win_rate_agent1_batches = split_array_into_batches(win_rate_agent1, batch_size)
        win_rate_agent2_batches = split_array_into_batches(win_rate_agent2, batch_size)

        self.weights = np.random.rand(agent1_general_context.shape[1]).reshape(-1, 1)

        for e in range(n_loops):
            print('Loop: ' + str(e + 1) + '/' + str(n_loops), end='\r')
            for agent1_batch, agent2_batch, win_rate_agent1_batch, win_rate_agent2_batch in zip(agent1_batches,
                                                                                                agent2_batches,
                                                                                                win_rate_agent1_batches,
                                                                                                win_rate_agent2_batches):
                z1 = np.dot(self.weights.T, agent1_batch.T)
                z2 = np.dot(self.weights.T, agent2_batch.T)
                pi1 = bandit_softmax(z1, [z1, z2]).T
                pi2 = bandit_softmax(z2, [z1, z2]).T

                avg = (pi1 * agent1_batch + pi2 * agent2_batch) / 2

                reward = np.empty_like(agent1_batch[:, 0])

                cont = np.empty_like(agent1_batch)

                for i in range(len(reward)):
                    if pi1[i] > pi2[i]:
                        reward[i] = win_rate_agent1_batch[i]
                        cont[i] = agent1_batch[i, :]
                    else:
                        reward[i] = win_rate_agent2_batch[i]
                        cont[i] = agent2_batch[i, :]

                w_update = cont - avg
                w_update = self.alpha * reward.reshape(-1, 1) * w_update
                w_update = np.mean(w_update, axis=0)

                self.weights = self.weights + w_update.reshape(-1, 1)

        self.trained = True

    def evaluate(self, data):
        cols_with_1 = [col for col in data.columns if ' 1' in col and not 'Win' in col]
        cols_with_2 = [col for col in data.columns if ' 2' in col]
        other_cols = [col for col in data.columns if col not in cols_with_1 + cols_with_2]

        sorted_cols = cols_with_1 + cols_with_2 + other_cols
        data = data[sorted_cols]

        test = data

        # get the win rate out of the data
        # the oracle value represent the "reward" an optimal user would get for a given context
        test_oracle_value = test.iloc[:, -1].to_numpy()
        test_oracle_value[test_oracle_value < 0.5] = 1 - test_oracle_value[test_oracle_value < 0.5]

        other_cols_wo_winrate = [col for col in other_cols if not 'Win' in col]
        test_2_arm_a1 = cols_with_1 + other_cols_wo_winrate
        test_2_arm_a1 = test[test_2_arm_a1].to_numpy()

        test_2_arm_a2 = cols_with_2 + other_cols_wo_winrate
        test_2_arm_a2 = test[test_2_arm_a2].to_numpy()

        win_rate_agent1_test = test['Win rate of Agent 1'].to_numpy()
        win_rate_agent2_test = 1 - win_rate_agent1_test

        agent1_test_context = np.concatenate(
            (np.zeros_like(test_2_arm_a2), test_2_arm_a2, test[cols_with_1].to_numpy()),
            axis=1).T
        agent2_test_context = np.concatenate(
            (test_2_arm_a1, np.zeros_like(test_2_arm_a1), test[cols_with_2].to_numpy()),
            axis=1).T

        test_z1 = np.dot(self.weights.T, agent1_test_context)
        test_z2 = np.dot(self.weights.T, agent2_test_context)
        test_p1 = bandit_softmax(test_z1, [test_z1, test_z2])
        test_p2 = bandit_softmax(test_z2, [test_z1, test_z2])

        regret = test_oracle_value - (test_p1 * win_rate_agent1_test + test_p2 * win_rate_agent2_test)
        return regret.mean()

    def predict(self, data):
        cols_with_1 = [col for col in data.columns if ' 1' in col and not 'Win' in col]
        cols_with_2 = [col for col in data.columns if ' 2' in col]
        other_cols = [col for col in data.columns if col not in cols_with_1 + cols_with_2]

        sorted_cols = cols_with_1 + cols_with_2 + other_cols
        data = data[sorted_cols]

        other_cols_wo_winrate = [col for col in other_cols if not 'Win' in col]
        data_2_arm_a1 = cols_with_1 + other_cols_wo_winrate
        data_2_arm_a1 = data[data_2_arm_a1].to_numpy()

        data_2_arm_a2 = cols_with_2 + other_cols_wo_winrate
        data_2_arm_a2 = data[data_2_arm_a2].to_numpy()

        agent1_general_context = np.concatenate(
            (np.zeros_like(data_2_arm_a2), data_2_arm_a2, data[cols_with_1].to_numpy()),
            axis=1).T
        agent2_general_context = np.concatenate(
            (data_2_arm_a1, np.zeros_like(data_2_arm_a1), data[cols_with_2].to_numpy()),
            axis=1).T

        z1 = np.dot(self.weights.T, agent1_general_context.T)
        z2 = np.dot(self.weights.T, agent2_general_context.T)
        pi1 = bandit_softmax(z1, [z1, z2]).T
        pi2 = bandit_softmax(z2, [z1, z2]).T
        return np.where(pi1 > pi2, 1, 2)


def bandit_softmax(wanted: float | np.ndarray, pi: [float | np.ndarray]) -> float:
    """
    This function returns the softmax probability of the wanted action given the current policy pi
    :param wanted: the value of the action for which we want to find the probability
    :param pi: the current policy with the z-values (context dot weights) for each action
    :return: the probability of choosing the wanted action
    """
    pi = np.array(pi).squeeze()
    max_val = np.max(pi, axis=0)
    wanted_rel = (wanted - max_val).astype(float)
    pi_rel = (pi - max_val).astype(float)

    numerator = np.exp(wanted_rel)
    denominator = np.zeros(wanted.shape)
    for p in pi_rel:
        denominator += np.exp(p)
    return numerator / denominator


def shuffle_arrays_in_unison(arr):
    """
    This function shuffles an array
    :param arr: the array to shuffle
    """
    p = np.random.permutation(len(arr[0]))
    ret_arr = []
    for i in range(len(arr)):
        ret_arr.append(arr[i][p])
    return ret_arr


def split_array_into_batches(arr, k):
    n = arr.shape[0]
    num_batches = n // k
    batches = [arr[i * k:(i + 1) * k] for i in range(num_batches)]
    if n % k != 0:
        batches.append(arr[num_batches * k:])
    return batches
