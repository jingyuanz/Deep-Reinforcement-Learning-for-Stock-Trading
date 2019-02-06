from doubleQ_agent import *
from dueling_agent import *
from supervised_agent import *
from config import Config


# agent = SupervisedAgent()
# agent.train()

qagent = QAgent()
# qagent.train()
qagent.evaluate(random=True, baseline=True, agent=True)

duel_agent = DuelingAgent()
# duel_agent.train()
duel_agent.evaluate()
