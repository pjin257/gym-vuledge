import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='vuledge-v1',
    entry_point='gym_vuledge.envs:VulEdgeEnv',
)

register(
    id='eval-v1',
    entry_point='gym_vuledge.envs:EvalEnv',
)