from .smpe_learner import SMPELearner

REGISTRY = {}

REGISTRY["smpe_learner"] = SMPELearner
REGISTRY["actor_critic_learner"] = SMPELearner
