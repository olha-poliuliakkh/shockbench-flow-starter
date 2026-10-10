"""The information dial and the wire (design §5, §9.1-9.3; milestone M3).

The regime theta of (47) (``theta``), the observation wrapper outside the environment core (``view`` builds the
reset-time feeds once, ``observe.wrap`` fills and blanks them per theta; §5.4 "Rules"), the warning scores (45)
(``warning``), the demand forecast (48) (``forecast``), the coverage rung (``coverage``), the V27 labels of §5.2
(``labels``), the announcements and decoys of (49) (``messages``, ``decoys``), the §9.2 line-JSON schema and the §9.3
line-level validity (``wire``), the local runner (``runner``) and the §9.1 flat view (``flat``).

Nothing here imports pydantic, gymnasium or hydra (Q18, Q95): the wire shapes are stdlib ``TypedDict``s, which the
tests validate through a strict pydantic subclass outside the package, and the gymnasium adapter lives in the sibling
package ``shockbench_flow_gym``.
"""
