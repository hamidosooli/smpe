REGISTRY = {}

from .basic_controller_smpe import BasicMACSMPE
from .non_shared_controller import NonSharedMAC
from .maddpg_controller import MADDPGMAC
from .centralized_controller import CentralizedMAC
from .non_shared_centralized_controller import NonSharedCentralizedMAC

REGISTRY["basic_mac_smpe"] = BasicMACSMPE
REGISTRY["basic_mac"] = BasicMACSMPE
REGISTRY["centralized_mac"] = CentralizedMAC
REGISTRY["non_shared_mac"] = NonSharedMAC
REGISTRY["maddpg_mac"] = MADDPGMAC
REGISTRY["ns_centralized_mac"] = NonSharedCentralizedMAC
