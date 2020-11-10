import sys

from env import Env, Qttt


class HumanAgent(object):
    def __init__(self, mark):
        self.mark = mark

    def get_location(self, ava_actions, posText):
        while True:
            uloc = input("Enter " + posText + " location[1-9]: ")
            if uloc.lower() == 'q':
                return None
            try:
                loc = int(uloc) - 1
                if loc not in ava_actions:
                    raise ValueError()
            except ValueError:
                print("Illegal location: '{}'".format(uloc))
            else:
                return loc

    def show_turn(self):
        if self.mark % 2 == 1:
            print("X's turn:")
        else:
            print("O's turn:")

    def act(self, free_block_ids, collapsed_qttts, mark=None):
        self.show_turn()

        if len(collapsed_qttts) > 1:
            index = self.observe(collapsed_qttts)
        else:
            index = 0

        if index is None:
            return None, None

        collapsed_qttt = collapsed_qttts[index]

        if free_block_ids[index] is None:
            return collapsed_qttt, None

        agent_move = self.put(free_block_ids[index])
        if agent_move is None:
            return None, None
        return collapsed_qttts[index], agent_move


    def put(self, free_block_ids):
        loc1 = self.get_location(free_block_ids, "first")
        if loc1 is None:
            return None
        loc2 = self.get_location(free_block_ids, "second")
        if loc2 is None:
            return None
        action = (loc1, loc2)
        self.mark += 2
        return action

    def observe(self, collapsed_qttts): 
        while True:
            n = len(collapsed_qttts)
            print("Choose the collapsed qttt board[1-" + str(n) + "]:")
            for i in range(n):
                print(str(i+1) + ":")
                collapsed_qttts[i].visualize_board()
            uindex = input("Choose the index: ")
            if uindex.lower() == 'q':
                return None
            try:
                index = int(uindex)-1
                if index < 0 or index >= n:
                    raise ValueError()
            except ValueError:
                print("Illegal index: '{}'".format(uindex))
            else:
                break
        return index


def play():
    env = Env()
    
    episode = 0
    while True:
        print("Game start. Enter q for quit.")
        _, mark = env.reset()
        done = False
        env.render()
        agents = [HumanAgent(1), HumanAgent(2)]
        while not done:
            agent = agent_by_mark(agents, mark)
            free_block_ids, collapsed_qttts = env.get_valid_moves()

            collapsed_qttt, agent_move = agent.act(free_block_ids, collapsed_qttts)
            
            if collapsed_qttt is None:
                sys.exit()

            state, mark, reward, done = env.step(collapsed_qttt, agent_move, mark)

            print('')
            env.render()
            if done:
                state.show_result()
                break

        episode += 1


def agent_by_mark(agents, mark):
    for agent in agents:
        if agent.mark == mark:
            return agent



if __name__ == '__main__':
    play()