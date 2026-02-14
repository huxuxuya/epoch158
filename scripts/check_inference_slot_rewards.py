#!/usr/bin/env python3
import argparse
import csv
import json
import os
from decimal import Decimal, getcontext
import urllib.parse
import urllib.request
from typing import Any

getcontext().prec = 50


def api_get(base: str, path: str, timeout: float = 40.0) -> dict[str, Any]:
    url = urllib.parse.urljoin(base.rstrip("/") + "/", path.lstrip("/"))
    req = urllib.request.Request(url=url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        charset = resp.headers.get_content_charset() or "utf-8"
        return json.loads(resp.read().decode(charset))


def fetch_all_epoch_groups(base_node: str, timeout: float) -> list[dict[str, Any]]:
    all_items: list[dict[str, Any]] = []
    next_key = ""
    while True:
        path = "/chain-api/productscience/inference/inference/epoch_group_data?pagination.limit=200"
        if next_key:
            path += "&pagination.key=" + urllib.parse.quote(next_key, safe="")
        page = api_get(base_node, path, timeout)
        all_items.extend(page.get("epoch_group_data", []) or [])
        next_key = ((page.get("pagination") or {}).get("next_key") or "")
        if not next_key:
            break
    return all_items


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check whether participants with ML nodes in inference slot received rewards."
    )
    parser.add_argument("--node", required=True, help="Validator node URL, e.g. http://node1.gonka.ai:8000")
    parser.add_argument("--epoch", type=int, required=True, help="Epoch number")
    parser.add_argument("--out-dir", default="artifacts", help="Output dir")
    parser.add_argument("--timeout-sec", type=float, default=40.0, help="HTTP timeout")
    parser.add_argument(
        "--inference-slot-index",
        type=int,
        default=0,
        help="Index in timeslot_allocation treated as inference slot (default: 0)",
    )
    args = parser.parse_args()

    base = args.node.rstrip("/")
    epoch = args.epoch
    timeout = args.timeout_sec

    participants_payload = api_get(base, f"/api/v1/epochs/{epoch}/participants", timeout)
    participants = (participants_payload.get("active_participants") or {}).get("participants", []) or []
    excluded = participants_payload.get("excluded_participants", []) or []
    excluded_by_addr = {str(e.get("address", "")): str(e.get("reason", "")) for e in excluded}

    perf_payload = api_get(
        base,
        f"/chain-api/productscience/inference/inference/epoch_performance_summary/{epoch}",
        timeout,
    )
    perf_rows = perf_payload.get("epochPerformanceSummary", []) or []
    rewarded_by_addr = {
        str(r.get("participant_id", "")): int(str(r.get("rewarded_coins", "0")))
        for r in perf_rows
        if str(r.get("participant_id", ""))
    }
    parent_epoch_group = api_get(
        base, f"/chain-api/productscience/inference/inference/epoch_group_data/{epoch}", timeout
    ).get("epoch_group_data", {})
    parent_vw = {
        str(vw.get("member_address", "")): vw for vw in (parent_epoch_group.get("validation_weights", []) or [])
    }

    all_groups = fetch_all_epoch_groups(base, timeout)

    # Aggregate slot weights from model subgroups (model_id != "") for this epoch.
    agg: dict[str, dict[str, int]] = {}
    for p in participants:
        addr = str(p.get("index", ""))
        agg[addr] = {
            "ml_nodes_total": 0,
            "inference_slot_nodes": 0,
            "verification_slot_nodes": 0,
            "inference_slot_weight": 0,
            "verification_slot_weight": 0,
            "all_mlnode_weight": 0,
        }

    for grp in all_groups:
        if int(grp.get("epoch_index", 0)) != epoch:
            continue
        model_id = str(grp.get("model_id", "") or "")
        if not model_id:
            continue

        for vw in grp.get("validation_weights", []) or []:
            addr = str(vw.get("member_address", ""))
            if addr not in agg:
                continue
            for node in vw.get("ml_nodes", []) or []:
                ts = node.get("timeslot_allocation", []) or []
                w = int(str(node.get("poc_weight", "0")))
                agg[addr]["ml_nodes_total"] += 1
                agg[addr]["all_mlnode_weight"] += w
                if len(ts) > 0 and bool(ts[0]):
                    agg[addr]["verification_slot_nodes"] += 1
                    agg[addr]["verification_slot_weight"] += w
                if len(ts) > 1 and bool(ts[1]):
                    agg[addr]["inference_slot_nodes"] += 1
                    agg[addr]["inference_slot_weight"] += w

    inference_slot_index = args.inference_slot_index
    for p in participants:
        addr = str(p.get("index", ""))
        if addr not in agg:
            continue
        # Recompute "inference slot" according to configured index.
        agg[addr]["inference_slot_nodes"] = 0
        agg[addr]["inference_slot_weight"] = 0

    for grp in all_groups:
        if int(grp.get("epoch_index", 0)) != epoch:
            continue
        model_id = str(grp.get("model_id", "") or "")
        if not model_id:
            continue
        for vw in grp.get("validation_weights", []) or []:
            addr = str(vw.get("member_address", ""))
            if addr not in agg:
                continue
            for node in vw.get("ml_nodes", []) or []:
                ts = node.get("timeslot_allocation", []) or []
                w = int(str(node.get("poc_weight", "0")))
                if len(ts) > inference_slot_index and bool(ts[inference_slot_index]):
                    agg[addr]["inference_slot_nodes"] += 1
                    agg[addr]["inference_slot_weight"] += w

    rows: list[dict[str, Any]] = []
    missed = 0
    for p in sorted(participants, key=lambda x: str(x.get("index", ""))):
        addr = str(p.get("index", ""))
        reward = rewarded_by_addr.get(addr, 0)
        slot = agg.get(addr, {})
        inf_w = int(slot.get("inference_slot_weight", 0))
        likely_missed = inf_w > 0 and reward == 0
        if likely_missed:
            missed += 1
        rows.append(
            {
                "index": addr,
                "rewarded_coins": reward,
                "inference_slot_nodes": int(slot.get("inference_slot_nodes", 0)),
                "inference_slot_weight": inf_w,
                "verification_slot_nodes": int(slot.get("verification_slot_nodes", 0)),
                "verification_slot_weight": int(slot.get("verification_slot_weight", 0)),
                "all_mlnode_weight": int(slot.get("all_mlnode_weight", 0)),
                "excluded_reason": excluded_by_addr.get(addr, ""),
                "parent_base_weight": int(str((parent_vw.get(addr) or {}).get("weight", "0"))),
                "parent_confirmation_weight": int(
                    str((parent_vw.get(addr) or {}).get("confirmation_weight", "0"))
                ),
                "likely_missed_reward_for_inference_slot": likely_missed,
            }
        )

    out_dir = os.path.join(args.out_dir, f"epoch_{epoch}")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "inference_slot_reward_check.csv")
    split_csv_path = os.path.join(out_dir, "inference_slot_reward_columns.csv")
    json_path = os.path.join(out_dir, "inference_slot_reward_check_summary.json")
    sim_csv_path = os.path.join(out_dir, "reward_simulation_comparison.csv")

    simulated_by_addr: dict[str, int] = {}
    if os.path.exists(sim_csv_path):
        with open(sim_csv_path, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                simulated_by_addr[row["index"]] = int(row["simulated_rewarded_coins"])

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "rewarded_coins",
                "inference_slot_nodes",
                "inference_slot_weight",
                "verification_slot_nodes",
                "verification_slot_weight",
                "all_mlnode_weight",
                "excluded_reason",
                "parent_base_weight",
                "parent_confirmation_weight",
                "likely_missed_reward_for_inference_slot",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    # Requested column format per participant:
    # 1) real reward from chain
    # 2) calculated reward for ML nodes in inference slot
    # 3) calculated reward for ML nodes not in inference slot
    # 4) ML node weight in inference slot
    # 5) ML node weight not in inference slot
    # Extra column `participant_index` is added for traceability.
    split_rows = []
    for r in rows:
        all_w = int(r["all_mlnode_weight"])
        inf_w = int(r["inference_slot_weight"])
        non_inf_w = int(r["all_mlnode_weight"]) - int(r["inference_slot_weight"])
        real_reward = int(r["rewarded_coins"])
        if all_w > 0:
            inf_calc = Decimal(real_reward) * (Decimal(inf_w) / Decimal(all_w))
            non_inf_calc = Decimal(real_reward) * (Decimal(non_inf_w) / Decimal(all_w))
        else:
            inf_calc = Decimal(0)
            non_inf_calc = Decimal(0)
        split_rows.append(
            {
                "real_reward_chain": str(real_reward),
                "calculated_reward_inference_slot_nodes": str(inf_calc),
                "calculated_reward_non_inference_slot_nodes": str(non_inf_calc),
                "ml_nodes_weight_inference_slot": str(inf_w),
                "ml_nodes_weight_non_inference_slot": str(non_inf_w),
                "participant_index": r["index"],
            }
        )

    with open(split_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "real_reward_chain",
                "calculated_reward_inference_slot_nodes",
                "calculated_reward_non_inference_slot_nodes",
                "ml_nodes_weight_inference_slot",
                "ml_nodes_weight_non_inference_slot",
                "participant_index",
            ],
        )
        writer.writeheader()
        writer.writerows(split_rows)

    participants_with_inf_slot = sum(1 for r in rows if int(r["inference_slot_weight"]) > 0)
    rewarded_with_inf_slot = sum(
        1 for r in rows if int(r["inference_slot_weight"]) > 0 and int(r["rewarded_coins"]) > 0
    )
    zero_with_inf_slot = [r for r in rows if int(r["inference_slot_weight"]) > 0 and int(r["rewarded_coins"]) == 0]
    zero_excluded = sum(1 for r in zero_with_inf_slot if r["excluded_reason"])
    zero_nonexcluded = [r for r in zero_with_inf_slot if not r["excluded_reason"]]
    zero_nonexcluded_conf0 = sum(1 for r in zero_nonexcluded if int(r["parent_confirmation_weight"]) == 0)
    zero_nonexcluded_conf_pos = sum(1 for r in zero_nonexcluded if int(r["parent_confirmation_weight"]) > 0)

    real_sum_inf = sum(int(r["rewarded_coins"]) for r in rows if int(r["inference_slot_weight"]) > 0)
    real_sum_non_inf = sum(int(r["rewarded_coins"]) for r in rows if int(r["inference_slot_weight"]) == 0)
    sim_sum_inf = (
        sum(simulated_by_addr.get(r["index"], 0) for r in rows if int(r["inference_slot_weight"]) > 0)
        if simulated_by_addr
        else None
    )
    sim_sum_non_inf = (
        sum(simulated_by_addr.get(r["index"], 0) for r in rows if int(r["inference_slot_weight"]) == 0)
        if simulated_by_addr
        else None
    )

    # Reward split by ML-node slot weights (not by participant buckets):
    # For each participant: reward * (inference_slot_weight / all_mlnode_weight)
    # and the remainder goes to non-inference-slot ML nodes.
    real_mlnode_inf_split = Decimal(0)
    real_mlnode_non_inf_split = Decimal(0)
    sim_mlnode_inf_split = Decimal(0)
    sim_mlnode_non_inf_split = Decimal(0)

    for r in rows:
        all_w = int(r["all_mlnode_weight"])
        inf_w = int(r["inference_slot_weight"])
        non_inf_w = max(all_w - inf_w, 0)
        real_reward = int(r["rewarded_coins"])
        sim_reward = simulated_by_addr.get(r["index"], 0) if simulated_by_addr else 0

        if all_w <= 0:
            continue

        inf_ratio = Decimal(inf_w) / Decimal(all_w)
        non_inf_ratio = Decimal(non_inf_w) / Decimal(all_w)

        real_mlnode_inf_split += Decimal(real_reward) * inf_ratio
        real_mlnode_non_inf_split += Decimal(real_reward) * non_inf_ratio
        if simulated_by_addr:
            sim_mlnode_inf_split += Decimal(sim_reward) * inf_ratio
            sim_mlnode_non_inf_split += Decimal(sim_reward) * non_inf_ratio

    summary = {
        "epoch": epoch,
        "inference_slot_index_used": inference_slot_index,
        "participants_total": len(rows),
        "participants_with_inference_slot_weight": participants_with_inf_slot,
        "participants_with_inference_slot_weight_and_rewarded": rewarded_with_inf_slot,
        "participants_with_inference_slot_weight_but_zero_reward": missed,
        "zero_reward_with_inference_slot_excluded": zero_excluded,
        "zero_reward_with_inference_slot_nonexcluded": len(zero_nonexcluded),
        "zero_reward_with_inference_slot_nonexcluded_confirmation_weight_zero": zero_nonexcluded_conf0,
        "zero_reward_with_inference_slot_nonexcluded_confirmation_weight_positive": zero_nonexcluded_conf_pos,
        "real_reward_sum_inference_slot_participants": real_sum_inf,
        "real_reward_sum_non_inference_slot_participants": real_sum_non_inf,
        "simulated_reward_sum_inference_slot_participants": sim_sum_inf,
        "simulated_reward_sum_non_inference_slot_participants": sim_sum_non_inf,
        "real_reward_calculated_by_inference_slot_ml_nodes": str(real_mlnode_inf_split),
        "real_reward_calculated_by_non_inference_slot_ml_nodes": str(real_mlnode_non_inf_split),
        "simulated_reward_calculated_by_inference_slot_ml_nodes": (
            str(sim_mlnode_inf_split) if simulated_by_addr else None
        ),
        "simulated_reward_calculated_by_non_inference_slot_ml_nodes": (
            str(sim_mlnode_non_inf_split) if simulated_by_addr else None
        ),
        "note": "zero reward with inference slot weight can be expected for excluded participants",
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
