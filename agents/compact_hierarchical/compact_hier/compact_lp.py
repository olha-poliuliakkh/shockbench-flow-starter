"""The compact window LP: ``mpc_nobuf``'s formulation, base-load-first relaxed to continuous z, solved by HiGHS.

Columns per window week: flows on the action slots (x), stock (I), disposal (O), supply lifts, lot starts (p),
packaging (q), unserved demand (U), the rationing buffers, base-load-first's z in [0, 1], the simulator's grid
(fuel burnt per segment gk, fuel kept back short, the load factor lam, shed sh), tanker queues (Q) and releases (y),
and the safety slacks.

What this module adds to ``mpc_nobuf``:

- The maritime controller's bounds (``maritime.RouteBounds``): a dispatch or release barred in window week h gets an
  upper bound of 0 there, and tanker cargo stranded at a strait closed past the window loses its terminal credit.
- The week's safety floors come from ``safety.SafetyFloors`` (crisis-scaled) instead of a fixed list.
- Base-load-first is always continuous: shed_g <= y_bar (1 - z) and sum_f e_f p_f <= E_max z with z in [0, 1]
  (``mpc_nobuf`` with ``milp_weeks`` 0). Only the simulator's grid model is kept (``mpc_nobuf``'s ``grid_model`` 1).
- The conditional terminal credit: stock in the window's last week, goods still travelling past it and tanker queues
  are worth ``terminal_frac_small`` or ``terminal_frac_full`` x v_k (never below salvage) only when the window ends
  before the episode; a window holding the last week credits salvage only.
- Floors that fade at the episode's end (``floor_taper_weeks`` N > 0): in absolute week w the floor is scaled by
  min(1, (T - w) / N), so it reaches 0 in week T and the plan drains the stores instead of refilling them.
- Production aligned with the simulator (``align_chip_production``; off builds the LP exactly as before):
  - OSATs, week 1: packaging fixed at the simulator's rule, min(raw chips on hand after the week's known arrivals,
    throughput), pro rata (no route reaches an OSAT's raw chips in the week it ships, so this is exact);
  - OSATs, every week: raw chips held or disposed there priced at ``osat_hold_price`` x v_k, so the plan packages
    them as the simulator does and ships the output instead of letting it overflow;
  - fabs, week 1: every fab on a grid starts the same share rho of p-hat = min(capacity, wafers on hand), the
    simulator's split; when the plan both sheds a grid's base load and powers its fabs, a second solve
    (``align_fab_second_solve``) bounds that week's shed (if the fabs' energy would have covered it) or its fabs
    (otherwise): base load first;
  - fabs, weeks with headroom (G_bar > y_bar at the fab's grid): wafers held or disposed there priced at
    ``wafer_hold_price`` x v_wafer, so the plan starts the wafers it has, as the simulator does.
  A solve that fails with the alignment is repeated once without it.
- Incidence lists from ``network.Network`` keep the build linear in the LP's size; the solve has a time limit.
"""

import time

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog


def floor_taper(t: int, H: int, T: int, N: int) -> np.ndarray:
    """Per window week h (absolute week t + h): the share of the safety floor kept, min(1, (T - t - h) / N), never
    below 0; all ones when N <= 0."""
    if N <= 0:
        return np.ones(H)
    left = T - (t + np.arange(H))  # weeks after week t + h: 0 in the episode's last week
    return np.clip(left / N, 0.0, 1.0)


def package(raw, thr: float) -> list[float]:
    """The simulator's OSAT rule (19): every raw chip on hand up to the throughput, pro rata across the products."""
    tot = float(sum(raw))
    if tot <= thr:
        return [float(r) for r in raw]
    return [thr * float(r) / tot for r in raw]


class CompactLP:
    def __init__(self, net, params: dict):
        self.net = net
        self.p = params
        self.rule_x = None  # this week's base-stock flows of wafers and chips (S,), kept when the LP fails
        self.status = None  # the last solve's status (diagnostics)
        self.last_plan = None  # the last solved window: solution, column offsets and sizes (offline analysis only)
        self.last_align = {}  # what the production alignment did this week (diagnostics)

    def align_switches(self) -> dict | None:
        """The production alignment's parts in force (None when ``align_chip_production`` is off)."""
        P_, net = self.p, self.net
        if float(P_["align_chip_production"]) < 0.5:
            return None
        return {
            "osat_week1": float(P_["align_osat_week1"]) >= 0.5 and not net.rule_on,
            "fab_week1": float(P_["align_fab_week1"]) >= 0.5,
            "fab_second_solve": float(P_["align_fab_week1"]) >= 0.5 and float(P_["align_fab_second_solve"]) >= 0.5,
            "osat_price": 0.0 if net.rule_on else max(0.0, float(P_["osat_hold_price"])),
            "wafer_price": max(0.0, float(P_["wafer_hold_price"])),
        }

    # ------------------------------------------------------------------------------------------ route costs

    def _route_costs(self, c_e, tariff, war, queue_cost):
        """Per unit cost of every route, override slot and queue pair in one week: freight, tariffs, war risk."""
        net, v = self.net, self.net.v
        cost = np.zeros(net.S)
        for s, (es, cs, k) in enumerate(zip(net.cap_edges, net.cap_chk, net.k_s)):
            if net.tk_dest[s] >= 0:
                cs = [net.edge_head[es[-1]]]
            cost[s] = sum(c_e[e] + tariff[e, k] * v[k] for e in es)
            for c in cs:
                cls = int(war[net.chk_row[c]])
                wr = net.chk_params[c].get("war_risk_cost", {}).get(net.k_id[k])
                if wr:
                    cost[s] += wr[min(cls, len(wr) - 1)]
                cost[s] += queue_cost.get((c, k), 0.0)
        ov_cost = np.zeros(net.NO)
        for i, (p_from, _e_out, path, _sp, _sj, passed) in enumerate(net.ov[: net.NO]):
            k = net.pairs[p_from][1] if p_from >= 0 else 0
            ov_cost[i] = sum(c_e[x] + tariff[x, k] * v[k] for x in path)
            for c in passed:
                cls = int(war[net.chk_row[c]])
                wr = net.chk_params[c].get("war_risk_cost", {}).get(net.k_id[k])
                if wr:
                    ov_cost[i] += wr[min(cls, len(wr) - 1)]
        q_hold = np.zeros(net.P)  # queue holding per unit-week at each pair
        for i, (c, k) in enumerate(net.pairs[: net.P]):
            qh = net.chk_params[c].get("queue_holding", {}).get(net.k_id[k])
            if qh:
                q_hold[i] = qh[min(int(war[net.chk_row[c]]), len(qh) - 1)]
        return cost, ov_cost, q_hold

    # ------------------------------------------------------------------------------------- wafers and chips

    def _rule_flows(self, o, t, H, win, lead, blocked_from, barred, arr):
        """Base-stock shipments of wafers and chips (``mpc_nobuf``'s rule). Each destination stock slot j uses d_j a
        week (a fab its expected capacity, an OSAT its throughput share of the demand for what it packages, a sink its
        forecast demand) and is kept at min(d_j (L_j + 1 + cover), storage_j + d_j L_j), L_j its fastest open route's
        lead; what is on hand or on its way counts. Stock at the origins goes, half a week of use at a time, to the
        destination with the fewest weeks of cover, over its fastest open route with capacity left. Routes through a
        closed strait or barred by the maritime controller are not used. Returns the flows and, per fab wafer slot, its
        lead and weekly use (the wafers the LP may count on)."""
        net, cover = self.net, float(self.p["rule_cover"])
        x = np.zeros(net.S)
        I0 = o["stock.qty"].astype(float)
        u_left = np.nan_to_num(np.asarray(win["u"][0], dtype=float), nan=0.0, posinf=1e18).copy()
        ct_left = {c: float(win["chk_cap"]["ct"][0, row]) for c, row in net.chk_row.items()}
        closed = win["closed"]
        usable = {}  # destination j -> its open rule slots
        for s in np.flatnonzero(net.rule_slot):
            j_out, j_in = net.j_out[s], net.j_in[s]
            if j_out < 0 or j_in < 0 or j_in not in net.dest or blocked_from[s] <= 0 or t + lead[s] > net.T:
                continue
            if barred[s] or any(c in closed for c in net.route_chk[s]):
                continue
            usable.setdefault(j_in, []).append(s)

        dem_k = {}  # packaged chip -> its forecast demand per week, all sinks
        for d, k in enumerate(net.demand_k):
            dem_k[k] = dem_k.get(k, 0.0) + float(np.mean(win["demand"][:, d]))
        thr = np.asarray(win["osat_thr"][0], dtype=float)
        share = {}  # (osat, packaged k) -> share of its throughput
        for osat, pks in net.osat_pk.items():
            tot = sum(dem_k.get(k, 0.0) for k in pks)
            for k in pks:
                share[(osat, k)] = dem_k.get(k, 0.0) / tot if tot > 0 else 1.0 / len(pks)
        cap_k = {}  # packaged chip -> OSAT throughput the shares give it, all OSATs
        for (osat, k), sh in share.items():
            cap_k[k] = cap_k.get(k, 0.0) + thr[osat] * sh
        IP = I0 + arr.sum(axis=0)
        d_of, target, fab_feed = {}, {}, {}
        for j, kind in net.dest.items():
            L = int(min(lead[s] for s in usable[j])) if j in usable else 0
            a, b = min(L, H - 1), min(H, L + int(cover) + 1)
            if kind[0] == "fab":
                d = float(np.mean(win["fab_cap"][a:b, kind[1]]))
                if j in usable:
                    fab_feed[j] = (L, np.asarray(win["fab_cap"][:, kind[1]], dtype=float))
            elif kind[0] == "osat":
                osat, k = kind[1], kind[2]
                scale = min(1.0, dem_k.get(k, 0.0) / cap_k[k]) if cap_k.get(k, 0.0) > 0 else 0.0
                d = thr[osat] * share[(osat, k)] * scale
            else:
                d = float(np.mean(win["demand"][a:b, kind[1]]))
            d_of[j] = max(d, 0.0)
            target[j] = min(d * (L + 1 + cover), net.storage[j] + d * L)
        if not usable:
            return x, fab_feed

        avail = I0.copy()
        sent = {j: 0.0 for j in usable}
        order = sorted(usable, key=lambda j: -max((pi for jj, pi in net.demands if jj == j), default=0.0))

        def slot_cap(s):
            cap = min(u_left[e] for e in net.cap_edges[s])
            for c in net.cap_chk[s]:
                cap = min(cap, ct_left[c])
            return max(cap, 0.0)

        for _step in range(4000):
            best, best_cov = None, np.inf
            for j in order:
                d = d_of[j]
                if d <= 0 or IP[j] + sent[j] >= target[j] - 1e-6:
                    continue
                cov = (IP[j] + sent[j]) / d
                if cov < best_cov and any(avail[net.j_out[s]] > 1e-6 and slot_cap(s) > 1e-6 for s in usable[j]):
                    best, best_cov = j, cov
            if best is None:
                break
            j = best
            s = min(
                (s for s in usable[j] if avail[net.j_out[s]] > 1e-6 and slot_cap(s) > 1e-6),
                key=lambda s: (lead[s], len(net.route_chk[s])),
            )
            q = min(target[j] - IP[j] - sent[j], avail[net.j_out[s]], slot_cap(s), max(0.5 * d_of[j], 1.0))
            x[s] += q
            sent[j] += q
            avail[net.j_out[s]] -= q
            for e in net.cap_edges[s]:
                u_left[e] -= q
            for c in net.cap_chk[s]:
                ct_left[c] -= q
        return x, fab_feed

    # ---------------------------------------------------------------------------------------------- the solve

    def solve(self, o, t, H, win, bounds, floors, time_limit):
        """The week's plan: (flows (S,), (override_qty, release_mode) or None), or None without an optimum in time.

        With the production alignment on, a failed solve is repeated once without it.
        """
        start = time.process_time()
        align = self.align_switches()
        out = self._solve(o, t, H, win, bounds, floors, time_limit, align)
        if out is None and align is not None:
            left = float(time_limit) - (time.process_time() - start)
            if left > 0.05:
                out = self._solve(o, t, H, win, bounds, floors, left, None)
                self.last_align = {"retried_without_alignment": True, "solved": out is not None}
        return out

    def _solve(self, o, t, H, win, bounds, floors, time_limit, align):
        """One build and solve (two with the week-1 base-load-first pass); ``align``: ``align_switches()`` or None."""
        start = time.process_time()
        net, P_ = self.net, self.p
        S, J, P, NO, NG = net.S, net.J, net.P, net.NO, net.NG
        tau = o["graph_now.tau"]
        v = net.v
        open_, war0 = o["graph_now.open"], o["graph_now.war_risk"]
        end = t + H - 1 >= net.T  # the window holds the episode's last week: what is left is worth its salvage
        tf = 0.0 if end else net.terminal_frac
        value = net.salvage if end else net.value
        strand = float(P_["strand_credit"])
        closed, reopen = win["closed"], win["reopen"]
        closed_below = float(P_["closed_below"])
        self.rule_x = None

        # straits: the weeks a partly open one delays cargo, and the queue holding that costs
        delay, queue_cost = {}, {}
        for c, row in net.chk_row.items():
            op = float(open_[row])
            if closed_below < op < 1.0:
                extra = (1.0 / op - 1.0) * float(P_["queue_scale"])
                delay[c] = int(np.ceil(extra))
                for k, name in enumerate(net.k_id):
                    qh = net.chk_params[c].get("queue_holding", {}).get(name)
                    if qh:
                        queue_cost[(c, k)] = qh[min(int(war0[row]), len(qh) - 1)] * extra
        use_pc = float(P_["pipeline_closure"]) >= 0.5

        # routes: lead (this week's), first prohibited window week; costs per window week where they change
        lead = np.array([int(sum(tau[e] for e in es)) for es in net.route_edges])
        blocked_from = np.full(S, H)
        proh = win["proh"]
        for s, (es, cs, k) in enumerate(zip(net.cap_edges, net.cap_chk, net.k_s)):
            if net.tk_dest[s] >= 0:  # a tanker lane: the dispatch only runs to the first strait
                lead[s] = int(sum(tau[e] for e in es))
                cs = [net.edge_head[es[-1]]]
            for c in cs:
                lead[s] += delay.get(c, 0)
            hit = np.flatnonzero(proh[:, es, k].any(axis=1))
            blocked_from[s] = hit[0] if hit.size else H
        if o["action_mask.observed"][0]:
            blocked_from = np.where(o["action_mask"] == 0, np.minimum(blocked_from, 0), blocked_from)
        ov_lead, ov_blocked = np.zeros(NO, dtype=int), np.full(NO, H)
        for i, (p_from, _e_out, path, _sp, _sj, _passed) in enumerate(net.ov[:NO]):
            k = net.pairs[p_from][1] if p_from >= 0 else 0
            ov_lead[i] = int(sum(tau[x] for x in path))
            hit = np.flatnonzero(proh[:, path, k].any(axis=1))
            ov_blocked[i] = 0 if p_from < 0 else (hit[0] if hit.size else H)
            if o["override_mask.observed"][0] and o["override_mask"][i] == 0:
                ov_blocked[i] = min(ov_blocked[i], 0)
        costs = []
        for h in range(H):
            same = h > 0 and all(np.array_equal(win[f][h], win[f][h - 1]) for f in ("c", "tariff", "war"))
            if not same:
                week_cost = self._route_costs(win["c"][h], win["tariff"][h], win["war"][h], queue_cost)
            costs.append(week_cost)

        # known arrivals into stock slots and tanker queues per window week
        arr = np.zeros((H, J))
        arrQ = np.zeros((H, max(P, 1)))

        def add(j, h, q):
            if j >= 0 and 0 <= h < H and q > 0:
                arr[h, j] += q

        def add_q(p, h, q):
            if p >= 0 and 0 <= h < H and q > 0:
                arrQ[h, p] += q

        for e, k, lane, lane_ok, q, w, ok in zip(
            o["pipeline.edge"],
            o["pipeline.k"],
            o["pipeline.lane"],
            o["pipeline.lane.observed"],
            o["pipeline.qty"],
            o["pipeline.arrival_week"],
            o["pipeline.qty.observed"],
        ):
            if not ok or q <= 0:
                continue
            e, k = int(e), int(k)
            if net.tk_on and net.tanker[k]:  # tanker cargo: it stops at the next strait, else at the edge's head
                head = net.edge_head[e]
                if (head, k) in net.pair_of:
                    add_q(net.pair_of[(head, k)], int(w) - t, float(q))
                else:
                    add(net.j_of.get((head, k), -1), int(w) - t, float(q))
            elif lane_ok and 0 <= int(lane) < len(net.lane_edges) and e in net.lane_edges[int(lane)]:
                es = net.lane_edges[int(lane)]
                i0 = es.index(e)
                h_arr = int(w) - t + int(sum(tau[x] for x in es[i0 + 1 :]))
                lost = False
                if use_pc:  # a closed strait still ahead holds the cargo until its announced reopening
                    for idx in range(i0, len(es) - 1):
                        c = int(net.edge_head[es[idx]])
                        if c in closed:
                            if c in reopen:
                                h_arr = max(h_arr, reopen[c] + int(sum(tau[x] for x in es[idx + 1 :])))
                            else:
                                lost = True
                if not lost:
                    add(net.j_of.get((net.edge_head[es[-1]], k), -1), h_arr, float(q))
            else:
                add(net.j_of.get((net.edge_head[e], k), -1), int(w) - t, float(q))
        Q0 = np.zeros(max(P, 1))
        if net.tk_on and not net.lot_keys and "queue_lots.chokepoint" in o:  # Tiny: one entry per lot
            for c, k, q, ok in zip(
                o["queue_lots.chokepoint"], o["queue_lots.k"], o["queue_lots.qty"], o["queue_lots.qty.observed"]
            ):
                if ok and q > 0 and (int(c), int(k)) in net.pair_of:
                    Q0[net.pair_of[(int(c), int(k))]] += float(q)
        if net.lot_keys and "queue_lots.qty" in o:
            qty = o["queue_lots.qty"].sum(axis=1)
            for (c, k, lane, nxt), q in zip(net.lot_keys, qty):
                if q <= 0:
                    continue
                if net.tk_on and (c, k) in net.pair_of:
                    Q0[net.pair_of[(c, k)]] += float(q)
                    continue
                es = net.lane_edges[lane]
                rest = es[es.index(nxt) :] if nxt in es else [nxt]
                h_arr = int(sum(tau[x] for x in rest)) + delay.get(c, 0)
                if c in closed:
                    if not (use_pc and c in reopen):
                        continue
                    h_arr += reopen[c]
                add(net.j_of.get((net.edge_head[es[-1]], k), -1), h_arr, float(q))
        for n, k, q, w, ok in zip(o["wip.node"], o["wip.k"], o["wip.qty"], o["wip.out_week"], o["wip.qty.observed"]):
            if ok and q > 0:
                add(net.j_of.get((int(n), int(k)), -1), int(w) - t, float(q))

        # the rule ships wafers and chips; the LP sees the wafers reach the fabs (this week's shipments at their lead,
        # then each fab's weekly use), and plans the rest
        if net.rule_on:
            self.rule_x, fab_feed = self._rule_flows(o, t, H, win, lead, blocked_from, bounds.barred_now, arr)
            for s in np.flatnonzero(self.rule_x > 0):
                add(net.j_in[s], int(lead[s]), float(self.rule_x[s]))
            for j, (L, use) in fab_feed.items():
                for h in range(1, H):
                    add(j, h + L, use[min(h + L, H - 1)])

        # columns
        F, M, D, G = len(net.fabs), len(net.osats), len(net.demands), len(net.grid_burn)
        R, U_ = len(net.ration), len(net.supply_j)
        sizes = {"x": S, "I": J, "O": J, "lift": U_, "p": F, "q": M, "U": D, "buf": R, "z": NG}
        sizes |= {"gk": G, "short": G, "lam": NG, "sh": NG, "Q": P, "y": NO, "safe": len(floors)}
        fab_w1 = align is not None and align["fab_week1"]
        sizes |= {"rho": NG if fab_w1 else 0}  # week 1: the share of p-hat every fab of the grid starts
        off, n = {}, 0
        for name, size in sizes.items():
            off[name] = n
            n += H * size

        def col(name, h, i):
            return off[name] + h * sizes[name] + i

        lb, ub = np.zeros(n), np.full(n, np.inf)
        cost_vec = np.zeros(n)
        I0 = o["stock.qty"].astype(float)
        rows_eq, cols_eq, vals_eq, b_eq = [], [], [], []
        rows_ub, cols_ub, vals_ub, b_ub = [], [], [], []

        def eq(entries, rhs_):
            r = len(b_eq)
            for cidx, val in entries:
                rows_eq.append(r)
                cols_eq.append(cidx)
                vals_eq.append(val)
            b_eq.append(rhs_)

        def le(entries, rhs_):
            r = len(b_ub)
            for cidx, val in entries:
                rows_ub.append(r)
                cols_ub.append(cidx)
                vals_ub.append(val)
            b_ub.append(rhs_)

        balance = [[[] for _ in range(J)] for _ in range(H)]  # entries per (h, j)
        rhs = arr.copy()
        qbal = [[[] for _ in range(P)] for _ in range(H)]  # tanker queue balance entries per (h, pair)
        qrhs = arrQ[:, :P].copy()
        pair_credit = np.array([tf * v[net.pairs[p][1]] * (strand if p in bounds.stranded else 1.0) for p in range(P)])
        base_first = float(P_["base_first"]) >= 0.5
        taper = floor_taper(t, H, net.T, int(P_["floor_taper_weeks"]))
        hold_extra, disp_extra = np.zeros((H, J)), np.zeros((H, J))  # the alignment's prices on stock and disposal
        q0, phat = {}, {}  # week 1: packaging fixed per OSAT pair, p-hat per fab
        info = {"aligned": align is not None}
        if align is not None:
            if align["osat_week1"]:
                for osat in range(net.n_osat):
                    ms = [m for m in net.osat_ms[osat] if net.osats[m][0] >= 0]
                    raw = [max(0.0, float(I0[net.osats[m][0]] + arr[0, net.osats[m][0]])) for m in ms]
                    for m, q in zip(ms, package(raw, max(0.0, float(win["osat_thr"][0, osat])))):
                        q0[m] = q
            if align["osat_price"] > 0:
                for j_raw, _j_pk, _tau, _osat in net.osats:
                    if j_raw >= 0:
                        price = align["osat_price"] * v[net.stock[j_raw][1]]
                        hold_extra[:, j_raw] += price
                        disp_extra[:, j_raw] += price
            for f, (j_in, _j_out, _tau) in enumerate(net.fabs):
                if j_in < 0:
                    continue
                if fab_w1:
                    W = max(0.0, float(I0[j_in] + arr[0, j_in]))
                    phat[f] = min(max(0.0, float(win["fab_cap"][0, f])), W)
                if align["wafer_price"] > 0:
                    g = net.fab_grid[f]
                    for h in range(H):
                        headroom = float(win["G_bar"][h, g] - win["y_bar"][h, g]) if g >= 0 else 1.0
                        if headroom > 0 and float(win["fab_cap"][h, f]) > 0:
                            price = align["wafer_price"] * v[net.stock[j_in][1]]
                            hold_extra[h, j_in] += price
                            disp_extra[h, j_in] += price
            info |= {"osat_week1": dict(q0), "fab_week1_phat": dict(phat)}
        for h in range(H):
            for p in range(P):
                cq = col("Q", h, p)
                qbal[h][p].append((cq, 1.0))
                if h > 0:
                    qbal[h][p].append((col("Q", h - 1, p), -1.0))
                else:
                    qrhs[0, p] += Q0[p]
                cost_vec[cq] = costs[h][2][p]
                if h == H - 1:
                    cost_vec[cq] -= pair_credit[p]
            for i, (p_from, _e_out, _path, stop_p, stop_j, _passed) in enumerate(net.ov[:NO]):
                cy = col("y", h, i)
                if h >= ov_blocked[i] or p_from < 0 or h < bounds.y_until[i]:
                    ub[cy] = 0.0
                cost_vec[cy] = costs[h][1][i]
                if p_from >= 0:
                    qbal[h][p_from].append((cy, 1.0))
                ha = h + ov_lead[i]
                if stop_p >= 0:
                    if ha < H:
                        qbal[ha][stop_p].append((cy, -1.0))
                    else:
                        cost_vec[cy] -= pair_credit[stop_p]
                elif stop_j >= 0:
                    if ha < H:
                        balance[ha][stop_j].append((cy, -1.0))
                    else:
                        cost_vec[cy] -= value[stop_j]
            for j in range(J):
                ci, co = col("I", h, j), col("O", h, j)
                balance[h][j].append((ci, 1.0))
                if h > 0:
                    balance[h][j].append((col("I", h - 1, j), -1.0))
                else:
                    rhs[0, j] += I0[j]
                balance[h][j].append((co, 1.0))
                ub[ci] = net.storage[j]
                cost_vec[ci] = net.holding[j] + hold_extra[h, j]
                cost_vec[co] = net.disposal[net.stock[j][1]] + disp_extra[h, j]
            for s in range(S):
                cx = col("x", h, s)
                if h >= blocked_from[s] or net.j_out[s] < 0 or net.rule_slot[s] or h < bounds.x_until[s]:
                    ub[cx] = 0.0
                cost_vec[cx] = costs[h][0][s]
                if net.j_out[s] >= 0:
                    balance[h][net.j_out[s]].append((cx, 1.0))
                ha = h + lead[s]
                if net.tk_dest[s] >= 0:
                    if ha < H:
                        qbal[ha][net.tk_dest[s]].append((cx, -1.0))
                    else:
                        cost_vec[cx] -= pair_credit[net.tk_dest[s]]
                elif net.j_in[s] >= 0:
                    if ha < H:
                        balance[ha][net.j_in[s]].append((cx, -1.0))
                    else:
                        cost_vec[cx] -= value[net.j_in[s]]
            for i, j in enumerate(net.supply_j):
                if j >= 0:
                    ub[col("lift", h, i)] = max(0.0, float(win["supply"][h, i]))
                    balance[h][j].append((col("lift", h, i), -1.0))
            for f, (j_in, j_out, tau_f) in enumerate(net.fabs):
                cp = col("p", h, f)
                ub[cp] = max(0.0, float(win["fab_cap"][h, f])) if j_in >= 0 else 0.0
                if h == 0 and f in phat:  # the simulator's week: p = rho_g p-hat (no grid or no energy use: p-hat)
                    g = net.fab_grid[f]
                    if g >= 0 and net.fab_e[f] > 0:
                        eq([(cp, 1.0), (col("rho", 0, g), -phat[f])], 0.0)
                    else:
                        lb[cp] = ub[cp] = phat[f]
                if j_in >= 0:
                    balance[h][j_in].append((cp, 1.0))
                if net.rule_on:  # the chips' way to the sinks is the rule's: a lot is worth its share of pi
                    cost_vec[cp] -= net.lot_value[f]
                elif j_out >= 0:
                    if h + tau_f < H:
                        balance[h + tau_f][j_out].append((cp, -1.0))
                    else:
                        cost_vec[cp] -= value[j_out]
            for m, (j_raw, j_pk, tau_o, _osat) in enumerate(net.osats):
                cq = col("q", h, m)
                if j_raw < 0 or net.rule_on:
                    ub[cq] = 0.0
                    continue
                if h == 0 and m in q0:  # the simulator packages every raw chip on hand, up to the throughput
                    lb[cq] = ub[cq] = q0[m]
                balance[h][j_raw].append((cq, 1.0))
                if j_pk >= 0:
                    if h + tau_o < H:
                        balance[h + tau_o][j_pk].append((cq, -1.0))
                    else:
                        cost_vec[cq] -= value[j_pk]
            if not net.rule_on:
                for osat in range(net.n_osat):
                    le([(col("q", h, m), 1.0) for m in net.osat_ms[osat]], max(0.0, float(win["osat_thr"][h, osat])))
                for d, (j, pi) in enumerate(net.demands):
                    dem = max(0.0, float(win["demand"][h, d]))
                    cu = col("U", h, d)
                    ub[cu] = dem
                    cost_vec[cu] = pi
                    if j >= 0:
                        rhs[h, j] -= dem
                        balance[h][j].append((cu, -1.0))
            # the simulator's grid (segments): fuel k yields up to share_k G_bar (rationed below psi ibar), the
            # unmodelled share needs no fuel, and every segment runs at the grid's one load factor lam:
            #   sum_k G_k + lam share_0 G_bar + shed - sum_f e_f p_f = y_bar,  G_k <= lam share_k G_bar,
            #   G_k >= lam share_k G_bar - short_k (short priced at short_price x v_k)
            for g, (grid, j, share, _voll) in enumerate(net.grid_burn):
                cap = share * max(0.0, float(win["G_bar"][h, grid]))
                cg, cs_, cl = col("gk", h, g), col("short", h, g), col("lam", h, grid)
                balance[h][j].append((cg, 1.0))
                le([(cg, 1.0), (cl, -cap)], 0.0)
                le([(cl, cap), (cg, -1.0), (cs_, -1.0)], 0.0)
                cost_vec[cs_] = v[net.stock[j][1]] * float(P_["short_price"])
                if j in net.ration_thr and cap > 0:  # G_k <= share_k G_bar I^{t-1} / (psi ibar)
                    r = cap / net.ration_thr[j]
                    if h > 0:
                        le([(cg, 1.0), (col("I", h - 1, j), -r)], 0.0)
                    else:
                        le([(cg, 1.0)], r * I0[j])
            for grid in range(NG):
                if fab_w1:
                    ub[col("rho", h, grid)] = 1.0 if h == 0 and net.grid_fabs[grid] else 0.0
                cl, csd, cz = col("lam", h, grid), col("sh", h, grid), col("z", h, grid)
                y_bar = max(0.0, float(win["y_bar"][h, grid]))
                G_bar = max(0.0, float(win["G_bar"][h, grid]))
                ub[cl], ub[csd] = 1.0, y_bar
                cost_vec[csd] = net.grid_voll[grid] * float(P_["shed_weight"])
                ent = [(col("gk", h, g), 1.0) for g in net.grid_gk[grid]]
                ent += [(cl, net.grid_null[grid] * G_bar), (csd, 1.0)]
                ent += [(col("p", h, f), -e_f) for f, e_f in net.grid_fabs[grid]]
                eq(ent, y_bar)
                # base load first, continuous: shed <= y_bar (1 - z), fab energy <= E_max z, z in [0, 1]
                if not base_first or not net.grid_fabs[grid]:
                    ub[cz] = 0.0
                    continue
                ub[cz] = 1.0
                le([(csd, 1.0), (cz, y_bar)], y_bar)
                e_max = sum(e_f * max(0.0, float(win["fab_cap"][h, f])) for f, e_f in net.grid_fabs[grid])
                le([(col("p", h, f), e_f) for f, e_f in net.grid_fabs[grid]] + [(cz, -e_max)], 0.0)
            for i, (j, threshold, marginal) in enumerate(net.ration):  # I[h, j] + buf >= threshold
                cb = col("buf", h, i)
                cost_vec[cb] = marginal * float(P_["buffer_weight"])
                le([(col("I", h, j), -1.0), (cb, -1.0)], -threshold)
            for i, (j, floor, price) in enumerate(floors):  # I[h, j] + safe >= floor x taper[h]
                cs_ = col("safe", h, i)
                cost_vec[cs_] = price
                le([(col("I", h, j), -1.0), (cs_, -1.0)], -floor * taper[h])
            # dispatch draws on the stock on hand at the start of the week
            for j, slots in net.jout_x.items():
                ent = [(col("x", h, s), 1.0) for s in slots]
                if h > 0:
                    le(ent + [(col("I", h - 1, j), -1.0)], 0.0)
                else:
                    le(ent, I0[j])
            for e in net.used_edges:
                ent = [(col("x", h, s), 1.0) for s in net.edge_x[e]] + [(col("y", h, i), 1.0) for i in net.edge_y[e]]
                cap_e = max(0.0, float(win["u"][h, e]))
                if float(P_["recover_weeks"]) > 0 and cap_e < net.u0[e]:  # a cut recovers toward nominal
                    cap_e += (net.u0[e] - cap_e) * (1.0 - np.exp(-h / float(P_["recover_weeks"])))
                le(ent, cap_e)
            for c, pool in net.chk_pairs:
                cap = float(win["chk_cap"][pool][h, net.chk_row[c]])
                ent = [(col("x", h, s), 1.0) for s in net.chk_x[(c, pool)]]
                ent += [(col("y", h, i), 1.0) for i in net.chk_y[(c, pool)]]
                le(ent, cap)
        for h in range(H):
            for j in range(J):
                eq(balance[h][j], rhs[h, j])
            for p in range(P):
                eq(qbal[h][p], qrhs[h, p])
        for j in range(J):
            cost_vec[col("I", H - 1, j)] -= value[j]

        A_eq = sp.csr_matrix((vals_eq, (rows_eq, cols_eq)), shape=(len(b_eq), n))
        A_ub = sp.csr_matrix((vals_ub, (rows_ub, cols_ub)), shape=(len(b_ub), n))
        ub = np.maximum(ub, lb)
        b_ub_, b_eq_ = np.array(b_ub), np.array(b_eq)

        def run(upper):
            left = float(time_limit) - (time.process_time() - start)
            return linprog(
                cost_vec,
                A_ub=A_ub,
                b_ub=b_ub_,
                A_eq=A_eq,
                b_eq=b_eq_,
                bounds=np.c_[lb, upper],
                method="highs",
                options={"time_limit": max(0.05, left)},
            )

        res = run(ub)
        self.status = int(res.status)
        if res.x is None or res.status != 0:
            self.last_align = info
            return None
        if fab_w1 and align["fab_second_solve"]:  # base load first in week 1: a grid that sheds powers no fab
            ub2, changed = ub.copy(), 0
            for grid in range(NG):
                if not net.grid_fabs[grid]:
                    continue
                shed = float(res.x[col("sh", 0, grid)])
                fab_energy = sum(e_f * float(res.x[col("p", 0, f)]) for f, e_f in net.grid_fabs[grid])
                tol = 1e-6 * max(1.0, float(win["y_bar"][0, grid]))
                if shed > tol and fab_energy > tol:
                    ub2[col("sh" if fab_energy >= shed else "rho", 0, grid)] = 0.0
                    changed += 1
            info |= {"week1_base_first_bounds": changed}
            if changed:
                res2 = run(ub2)
                info["week1_second_solve"] = bool(res2.x is not None and res2.status == 0)
                if info["week1_second_solve"]:
                    res = res2
        self.last_align = info
        self.last_plan = {"x": res.x, "cost": cost_vec, "off": off, "sizes": sizes, "H": H, "objective": float(res.fun)}
        self.last_plan["align"] = dict(info)
        flows = res.x[off["x"] : off["x"] + S].copy()
        if self.rule_x is not None:
            flows = np.where(net.rule_slot, self.rule_x, flows)
        if not net.tk_on:
            return flows, None
        # the week's tanker releases: release_mode 1 on every pair with a sendable override slot, its quantities from
        # the plan (0 holds: cargo at a strait the controller bars stays put); other pairs keep the default release
        qty = np.zeros(net.n_override)
        qty[:NO] = res.x[off["y"] : off["y"] + NO]
        mode = np.zeros(net.n_pairs, dtype=np.int64)
        for i, (p_from, *_rest) in enumerate(net.ov[:NO]):
            if p_from >= 0 and ov_blocked[i] > 0:
                mode[p_from] = 1
        return flows, (qty, mode)
