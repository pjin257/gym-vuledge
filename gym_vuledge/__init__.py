import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='vuledge-v0',
    entry_point='gym_vuledge.envs:VulEdgeEnv',
)