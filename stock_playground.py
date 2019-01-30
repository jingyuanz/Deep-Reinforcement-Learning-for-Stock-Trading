from agent import *
from config import Config


# agent = SupervisedAgent()
# agent.train()

qagent = QAgent()
# qagent.train()
qagent.evaluate(random=True, baseline=True, agent=True)
