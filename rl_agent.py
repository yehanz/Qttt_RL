from tqdm import tqdm
from GameTree import GameTree
from env import Env, after_action_state
import random

gamma = 0.9


class TD_agent:
    def __init__(self, epsilon, alpha, decay_rate=0.1):
        self.epsilon = epsilon
        self.alpha = alpha
        self.decay_rate = decay_rate

    def act(self, free_qblock_id_lists, collapsed_qttts, mark):
        """
        Choose what action to take based on given collapsed Qttt states

        :param list(list(int))  free_qblock_id_lists:
            each element list contains ids of free QBlock under
            a given collapsed_qttt
        :param list(Qttt) collapsed_qttt:
            possible qttt states after collapse, if a state already reaches
            terminal state, the corresponding free_QBlock_id would be None
        :param int              mark: mark used by the agent

        :return:
            tuple(int, int) agent_action:
                pair of qblock id to place the spooky mark
                can be None if the state after collapse is already the terminal state
            Qttt      collapsed_qttt:
                qttt object with collapsed state on which agent's action based on
        """
        return self.epsilon_greedy_policy(free_qblock_id_lists, collapsed_qttts, mark)

    def epsilon_greedy_policy(self, free_qblock_id_lists, collapsed_qttts, mark):
        """
        Choose between random move(exploration) and best move we can come up with(exploitation)

        :param list(list(int))  free_qblock_id_lists:
            each element list contains ids of free QBlock under
            a given collapsed_qttt
        :param list(Qttt)       collapsed_qttt:
            possible qttt states after collapse, if a state already reaches
            terminal state, the corresponding free_QBlock_id would be None
        :param int              mark: mark used by the agent

        :return:
            tuple(int, int) agent_action:
                pair of qblock id to place the spooky mark
                can be None if the state after collapse is already the terminal state
            Qttt      collapsed_qttt:
                qttt object with collapsed state on which agent's action based on
        """
        e = random.random()
        if e < self.epsilon * self.decay_rate:
            collapsed_qttt, agent_move = self.random_action(free_qblock_id_lists, collapsed_qttts)
        else:
            collapsed_qttt, agent_move = self.greedy_action(free_qblock_id_lists, collapsed_qttts, mark)
        return collapsed_qttt, agent_move


    def random_action(self, free_qblock_id_lists, collapsed_qttts):
        n = len(collapsed_qttts)
        index = random.randint(0, n-1)
        move = random.sample(list(free_qblock_id_lists[index]), 2)
        return collapsed_qttts[index], tuple(move)

    def greedy_action(self, free_qblock_id_lists, collapsed_qttts, mark):
        assert len(collapsed_qttts) > 0
        states = {}
        for i in range(len(collapsed_qttts)):
            n = len(free_qblock_id_lists[i])
            for j in range(n-1):
                for k in range(j+1, n):
                    loc1 = free_qblock_id_lists[i][j]
                    loc2 = free_qblock_id_lists[i][k]
                    nstate = after_action_state(collapsed_qttts[i], (loc1, loc2), mark)
                    states[(i, j, k)] = GameTree.get_stat_val(nstate)
        if mark % 2 == 1:
            indices = GameTree.best_states(states, min)
        else:
            indices = GameTree.best_states(states, max)
        
        i, j, k = random.choice(indices)
        action = (collapsed_qttts[i], (j, k))
        return action

    
    def bellman_backup(self, qttt, next_qttt, reward, mark):
        """
        Bellman backup for TD learning

        :param Qttt state: current state of qttt
        :param Qttt next_state: next state after action is take
        :param int  reward: immediate reward for this round
        :return: None
        """
        state_value = GameTree.get_stat_val(qttt.get_state(mark))
        next_state_value = GameTree.get_stat_val(next_qttt.get_state(mark))
        updated_state_value = state_value + self.alpha*(reward + gamma*next_state_value - state_value)
        GameTree.set_state_value(qttt.get_state(mark), updated_state_value)


class ProgramDriver:
    def __init__(self, epsilon, alpha, decay_rate=0.1):
        self.epsilon = epsilon
        self.alpha = alpha
        self.decay_rate = decay_rate

    @staticmethod
    def get_agent_by_mark(agents, mark):
        return agents[mark % 2]

    def learn(self, max_episode):
        self._learn(max_episode)

    def _learn(self, max_episode, save_as_file='TD_policy.dat'):
        env = Env()
        agents = [TD_agent(self.epsilon, self.alpha, self.decay_rate),
                  TD_agent(self.epsilon, self.alpha, self.decay_rate)]

        for _ in tqdm(range(max_episode)):
            GameTree.reset_game_tree()

            # clear all state, env keep a counter for current round
            # odd round->x, even round->o, because for each piece, it has a submark on it!
            env.reset()

            while True:
                curr_qttt, mark = env.get_state()

                agent = ProgramDriver.get_agent_by_mark(agents, mark)

                free_qblock_id_lists, collapsed_qttts = env.get_valid_moves()

                collapsed_qttt, agent_move = agent.act(free_qblock_id_lists, collapsed_qttts, mark)

                next_qttt, next_round, reward, done = env.step(collapsed_qttt, agent_move, mark)

                agent.bellman_backup(curr_qttt, next_qttt, reward, mark)

                if done:
                    GameTree.set_state_value(next_qttt.get_state(mark), reward)
                    break

            ProgramDriver.exchange_agent_sequence(agents)

        ProgramDriver.save_model(save_as_file, max_episode, self.epsilon, self.alpha, self.decay_rate)

