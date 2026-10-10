"""Integer codes of the omega container and the seed keys, frozen with schema_version 1 (design §4.1; Q86, Q87).

Every categorical key component is a 0-based index; regions, nodes, edges, lanes and commodities are positions in the
instance file's lists. Any change here bumps ``schema_version``. ``UNIT_KINDS`` and ``MESSAGE_KINDS`` are two §9.2
vocabularies of schema 1 (Q86) that milestone M3 defines here, their one home (every module imports them from here):
the design froze them at version 1, so defining them bumps nothing. ``TYPE_CHANNELS`` and ``DECOY_CHANNELS`` are the
channel sets Q97 fixes (symmetric decoys), frozen here for the same reason.
"""

from shockbench_flow.instance.schema import INSTANCE_KINDS as INSTANCE_KINDS  # one definition, re-exported
from shockbench_flow.instance.schema import SCHEMA_VERSION as SCHEMA_VERSION  # one definition, re-exported


BLOCKS = ("P", "M")  # policy block 0, militarised block 1; unexcited Poisson components use -1
LAYERS = ("conflict", "tension", "dyad")
EVENT_TYPES = (
    "tariff",  # 0, block P
    "sanction",  # 1, sanction or export control, block P
    "material_outage",  # 2, block P
    "militarised_closure",  # 3, block M
    "regional_conflict",  # 4, block M
    "piracy",  # 5, block M
    "energy_shock",  # 6, deliverable-energy shock, block M until the owner decides (§11 row 21)
    "weather_closure",  # 7, weather or accident closure, unexcited Poisson
    "port_strike",  # 8, unexcited Poisson
)
EVENT_BLOCK = (0, 0, 0, 1, 1, 1, 1, -1, -1)
TARGET_KINDS = ("chokepoint", "edge", "node", "region")  # ev_target: node index, edge index, node index, region index
CHANNELS = ("tariff_formal", "tariff_informal", "tariff_final", "sanction_legal", "ties_threat", "mid_threat")
BLACKOUT_SPELLS = ("3w", "11w", "to_end")
# signal-unit kinds, in the row order of omega's X and W (regions, then dyads, then chokepoints; §4.1 codes, §9.2)
UNIT_KINDS = ("region", "dyad", "chokepoint")
# the kind of a §9.2 message (Q86); a withdrawal carries its thread's msg_id and decoy-bearing channel (design §12, M3)
MESSAGE_KINDS = ("proposal", "final_notice", "threat", "publication", "withdrawal")
# the channel set of each announced event type (EVENT_TYPES names), real and shadow alike (Q97: symmetric decoys);
# 'proposal' is tariff_formal or tariff_informal by the informal share; "sanction" covers export controls; every other
# type is unannounced. ``disruption.params.InformationParams.channels`` must equal it (design §12 "Messages")
TYPE_CHANNELS = (
    ("tariff", ("proposal", "tariff_final")),
    ("sanction", ("ties_threat", "sanction_legal")),
    ("militarised_closure", ("mid_threat",)),
)
# the channels with a shadow process of (49): a final notice or a legal publication belongs to its thread's channel
DECOY_CHANNELS = ("tariff_formal", "tariff_informal", "ties_threat", "mid_threat")
RESTORATION_REGIMES = ("A", "B", "C", "D")  # ev_restoration 0-3, -1 if not a fab event
CONFLICT_STATES = ("none", "minor", "war")  # z_c codes
TENSION_STATES = ("normal", "tension")  # z_p codes

# stream ids of design §4.1 (27)
STREAM_REGIME = 0
STREAM_HAWKES = 1
STREAM_DURATION = 2
STREAM_SEVERITY = 3
STREAM_LEAD = 4
STREAM_DEMAND = 5
STREAM_SHADOW = 7
STREAM_INSTANCE = 8
STREAM_TYPE_TARGET = 9
STREAM_RESTORATION = 10
STREAM_BLACKOUT = 11
STREAM_ADVERSARIAL = 12
STREAM_LATENT = 13
STREAM_WEATHER = 14
STREAM_STRIKE = 15
STREAM_TARIFF = 16
STREAM_SANCTION_SET = 17
STREAM_OPPONENT = 19
STREAM_HC_WEIGHTS = 20
STREAM_COUNTERPART = 21
STREAM_NAIVE_FQ = 22  # naive's reset-time replications, outside omega
STREAM_SCENARIOS = 23  # M4 reading (design §12): G_pub draws of mpc_scen and hindsight_consensus, outside omega
