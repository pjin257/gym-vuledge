import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='vuledge-v1',
    entry_point='gym_vuledge.envs:VulEdgeEnv',
)

register(
    id='vuledge2d-v1',
    entry_point='gym_vuledge.envs:VulEdge2dEnv',
)

register(
    id='vuledge3d-v1',
    entry_point='gym_vuledge.envs:VulEdge3dEnv',
)

register(
    id='vuledge4d-v1',
    entry_point='gym_vuledge.envs:VulEdge4dEnv',
)

register(
    id='reverse-v1',
    entry_point='gym_vuledge.envs:ReverseEnv',
)

register(
    id='eval-v1',
    entry_point='gym_vuledge.envs:EvalEnv',
)

register(
    id='reval-v1',
    entry_point='gym_vuledge.envs:REvalEnv',
)