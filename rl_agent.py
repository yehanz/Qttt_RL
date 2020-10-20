from tqdm import tqdm
from my_ttt.GameTree import GameTree
from my_ttt.env import Env

gamma = 0.9


class TD_agent:
    def __init__(self, epsilon, alpha, decay_rate=0.1):
        self.epsilon = epsilon
        self.alpha = alpha
        self.decay_rate = decay_rate

    def act(self, free_qblock_id_lists, collapsed_qttt_states, mark):
        """
        Choose what action to take based on given collapsed Qttt states

        :param list(list(int))  free_qblock_id_lists:
            each element list contains ids of free QBlock under
            a given collapsed_states
        :param list(Qttt.board)       collapsed_qttt_states:
            possible qttt states after collapse
        :param int              mark: mark used by the agent

        :return:
            tuple(int, int) agent_action:
                pair of qblock id to place the spooky mark
                can be None if the state after collapse is already the terminal state
            Qttt.board      collapsed_states:
                collapsed state on which agent's action based on
        """
        return self.epsilon_greedy_policy(free_qblock_id_lists, collapsed_qttt_states, mark)

    def epsilon_greedy_policy(self, free_qblock_id_lists, collapsed_qttt_states, mark):
        pass

    def bellman_backup(self, state, next_state, reward):
        """
        Bellman backup for TD learning

        :param Qttt.board   state: current state of qttt
        :param Qttt.board   next_state: next state after action is take
        :param int          reward: immediate reward for this round
        :return: None
        """
        state_value = GameTree.get_stat_val(state)
        next_state_value = GameTree.get_stat_val(next_state)
        updated_state_value = state_value + self.alpha*(reward + gamma*next_state_value - state_value)
        GameTree.set_state_value(state, updated_state_value)


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
                state, mark = env.get_state()

                agent = ProgramDriver.get_agent_by_mark(agents, mark)

                free_qblock_id_lists, collapsed_qttt_states = env.get_valid_moves()

                agent_move, collapsed_qttt_state = agent.get_move(free_qblock_id_lists, collapsed_qttt_states, mark)

                next_state, next_mark, reward, done = env.step(agent_move, collapsed_qttt_state, mark)

                agent.bellman_backup(state, next_state, reward)

                if done:
                    GameTree.set_state_value(next_state, reward)
                    break

            ProgramDriver.exchange_agent_sequence(agents)

        ProgramDriver.save_model(save_as_file, max_episode, self.epsilon, self.alpha, self.decay_rate)

