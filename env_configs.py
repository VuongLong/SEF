from types import SimpleNamespace
from common import NetType


class EnvConfig(SimpleNamespace):
    V: NetType
    Z: NetType
    Policy: NetType
    gamma = 0.99
    env_kwargs = {}


env_configs = {
    "GridWorld": EnvConfig(
        Value=[324, 256, 356, 1],
        Z=[324, 256, 256, 2],
        Policy=[324, 256, 256, 8],
        map_index=2
    )
}
