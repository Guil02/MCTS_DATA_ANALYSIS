import numpy as np
from sklearn.model_selection import train_test_split


class MAB_61_ARMS:
    def __init__(self, weights=None, alpha=0.05):
        if weights is None:
            self.weights = np.empty(1)
        else:
            self.weights = weights
        self.alpha = alpha
        self.trained = False

        expansion = ['ProgressiveHistory', 'Random', 'UCB1', 'UCB1GRAVE', 'UCB1Tuned']
        exploration = [0.1, 0.6, 1.41421356237]
        play_out = ['NST', 'MAST', 'Random', 'Random0', 'Random200', 'Random4']
        agent_combinations = 61
        agent_context_length = len(expansion) + 1 + len(play_out)
        self.agent_context = np.zeros((agent_combinations, agent_context_length))

        index = 0
        for e in expansion:
            for exp in exploration:
                for p in play_out:
                    if e == 'Random' and (exp != 1.41421356237 or p != 'Random') or p == 'Random' and (
                            e != 'Random' or exp != 1.41421356237):
                        continue

                    expansion_context_one_hot = np.zeros(len(expansion))
                    expansion_context_one_hot[expansion.index(e)] = 1
                    exploration_context = exp
                    play_out_context_one_hot = np.zeros(len(play_out))
                    play_out_context_one_hot[play_out.index(p)] = 1
                    self.agent_context[index] = np.concatenate(
                        (expansion_context_one_hot, [exploration_context], play_out_context_one_hot))
                    index += 1

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
        context_length_61_arms = 184

        other_cols_wo_winrate = [col for col in other_cols if not 'Win' in col]
        win_rate_agent1 = train['Win rate of Agent 1'].to_numpy()

        train_61_arms_environment_context = train[cols_with_2 + other_cols_wo_winrate]
        context_per_agent = []

        for agent in self.agent_context:
            agent = np.tile(agent, (len(train_61_arms_environment_context), 1))

            context_per_agent.append(np.concatenate((train_61_arms_environment_context, agent), axis=1))

        agent1_61_arms = train[cols_with_2 + other_cols_wo_winrate + cols_with_1].to_numpy()
        shuffled_arrays_61_arms = shuffle_arrays_in_unison([agent1_61_arms, win_rate_agent1] + context_per_agent)

        agent1_61_arms = shuffled_arrays_61_arms[0]
        win_rate_agent1_61_agents = shuffled_arrays_61_arms[1].reshape(-1, 1)
        context_per_agent = shuffled_arrays_61_arms[2:]

        agent1_61_arms_batches = split_array_into_batches(agent1_61_arms, batch_size)
        win_rate_agent1_61_agents_batches = split_array_into_batches(win_rate_agent1_61_agents, batch_size)
        context_per_agent_batches = []
        for context in context_per_agent:
            context_per_agent_batches.append(split_array_into_batches(context, batch_size))

        self.weights = np.random.rand(context_length_61_arms).reshape(-1, 1)

        alpha = 0.05
        n_loops = 1

        for e in range(n_loops):
            print('Loop: ' + str(e + 1) + '/' + str(n_loops), end='\r')
            for agent1_batch, win_rate_agent1_61_agents_batch, i in zip(agent1_61_arms_batches,
                                                                        win_rate_agent1_61_agents_batches,
                                                                        range(len(agent1_61_arms_batches))):
                z1 = np.dot(self.weights.T, agent1_batch.T)
                z_agent = []
                for context_batch in context_per_agent_batches:
                    z_agent.append(np.dot(self.weights.T, context_batch[i].T))

                pi1 = bandit_softmax(z1, z_agent).T
                pi_agent = []
                for z in z_agent:
                    pi_agent.append(bandit_softmax(z, z_agent).T)

                avg = np.zeros_like(agent1_batch)
                for j in range(len(pi_agent)):
                    avg += pi_agent[j] * context_per_agent_batches[j][i]

                avg = avg / len(pi_agent)

                cont = agent1_batch
                reward = win_rate_agent1_61_agents_batch
                w_update = cont - avg
                w_update = alpha * reward * w_update
                w_update = np.mean(w_update, axis=0)

                self.weights = self.weights + w_update.reshape(-1, 1)

    def predict(self, data):
        cols_with_1 = [col for col in data.columns if ' 1' in col and not 'Win' in col]
        cols_with_2 = [col for col in data.columns if ' 2' in col]
        other_cols = [col for col in data.columns if col not in cols_with_1 + cols_with_2]

        sorted_cols = cols_with_1 + cols_with_2 + other_cols
        data = data[sorted_cols]

        other_cols_wo_winrate = [col for col in other_cols if not 'Win' in col]
        data_61_arms_environment_context = data[cols_with_2 + other_cols_wo_winrate]
        context_per_agent = []

        for agent in self.agent_context:
            agent = np.tile(agent, (len(data_61_arms_environment_context), 1))
            context_per_agent.append(np.concatenate((data_61_arms_environment_context, agent), axis=1))

        z_agent = []
        for context in context_per_agent:
            z_agent.append(np.dot(self.weights.T, context.T))

        pi_agent = []
        for z in z_agent:
            pi_agent.append(bandit_softmax(z, z_agent).T)

        return np.max(np.array(pi_agent).squeeze(), axis=0)

    def evaluate(self, data):
        cols_with_1 = [col for col in data.columns if ' 1' in col and not 'Win' in col]
        cols_with_2 = [col for col in data.columns if ' 2' in col]
        other_cols = [col for col in data.columns if col not in cols_with_1 + cols_with_2]
        other_cols_wo_winrate = [col for col in other_cols if not 'Win' in col]

        sorted_cols = cols_with_1 + cols_with_2 + other_cols
        data = data[sorted_cols]

        test = data

        test_oracle_value = test.iloc[:, -1].to_numpy()
        test_oracle_value[test_oracle_value < 0.5] = 1 - test_oracle_value[test_oracle_value < 0.5]

        test_61_arms_environment_context = test[cols_with_2 + other_cols_wo_winrate]
        context_per_agent = []

        for agent in self.agent_context:
            agent = np.tile(agent, (len(test_61_arms_environment_context), 1))

            context_per_agent.append(np.concatenate((test_61_arms_environment_context, agent), axis=1))

        agent1_61_arms_test = test[cols_with_2 + other_cols_wo_winrate + cols_with_1].to_numpy()
        agent2_61_arms_test = test[cols_with_2 + other_cols_wo_winrate + cols_with_2].to_numpy()

        win_rate_agent1_test = test['Win rate of Agent 1'].to_numpy()
        win_rate_agent2_test = 1 - win_rate_agent1_test
        test_z1 = np.dot(self.weights.T, agent1_61_arms_test.T)
        test_z2 = np.dot(self.weights.T, agent2_61_arms_test.T)
        test_z_agent = []
        for context in context_per_agent:
            test_z_agent.append(np.dot(self.weights.T, context.T))

        test_p1 = bandit_softmax(test_z1, test_z_agent)
        test_p2 = bandit_softmax(test_z2, test_z_agent)

        regret = test_oracle_value - (test_p1 * win_rate_agent1_test + test_p2 * win_rate_agent2_test)
        return regret.mean()


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
