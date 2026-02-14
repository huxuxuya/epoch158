#!/usr/bin/env python3
import argparse
import csv
import json
import os
from decimal import Decimal, ROUND_DOWN, ROUND_HALF_UP, getcontext
import urllib.parse
import urllib.request
from typing import Any

getcontext().prec = 80

PROPOSAL_AUTHOR_ADDRESS = 'gonka1t7mcnc8zjkkvhwmfmst54sasulj68e5zsv4yzu'
PROPOSAL_AUTHOR_FEE_NGONKA = 500 * 1_000_000_000


def dec_to_plain_str(x: Decimal) -> str:
    s = format(x, 'f')
    if '.' in s:
        s = s.rstrip('0').rstrip('.')
    return s if s else '0'


def dec_to_int_str(x: Decimal) -> str:
    return str(int(x.quantize(Decimal('1'), rounding=ROUND_HALF_UP)))


def ngnk_to_gnk_str(ngnk_value: str) -> str:
    gnk = Decimal(ngnk_value) / Decimal(1_000_000_000)
    return dec_to_plain_str(gnk)


def api_get(base: str, path: str, timeout: float = 40.0, headers: dict[str, str] | None = None) -> dict[str, Any]:
    url = urllib.parse.urljoin(base.rstrip('/') + '/', path.lstrip('/'))
    req = urllib.request.Request(url=url, method='GET', headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        charset = resp.headers.get_content_charset() or 'utf-8'
        return json.loads(resp.read().decode(charset))


def fetch_all_epoch_groups(base_node: str, timeout: float, at_height: int | None = None) -> list[dict[str, Any]]:
    all_items: list[dict[str, Any]] = []
    next_key = ''
    headers = {'x-cosmos-block-height': str(at_height)} if at_height is not None else None
    while True:
        path = '/chain-api/productscience/inference/inference/epoch_group_data?pagination.limit=200'
        if next_key:
            path += '&pagination.key=' + urllib.parse.quote(next_key, safe='')
        page = api_get(base_node, path, timeout, headers=headers)
        all_items.extend(page.get('epoch_group_data', []) or [])
        next_key = ((page.get('pagination') or {}).get('next_key') or '')
        if not next_key:
            break
    return all_items


def aggregate_slot_stats(
    all_groups: list[dict[str, Any]],
    epoch: int,
    participant_ids: set[str],
    inference_slot_index: int,
) -> dict[str, dict[str, int]]:
    agg: dict[str, dict[str, int]] = {}
    for addr in participant_ids:
        agg[addr] = {
            'ml_nodes_total': 0,
            'inference_slot_nodes': 0,
            'verification_slot_nodes': 0,
            'inference_slot_weight': 0,
            'verification_slot_weight': 0,
            'all_mlnode_weight': 0,
        }

    for grp in all_groups:
        if int(grp.get('epoch_index', 0)) != epoch:
            continue
        model_id = str(grp.get('model_id', '') or '')
        if not model_id:
            continue
        for vw in grp.get('validation_weights', []) or []:
            addr = str(vw.get('member_address', ''))
            if addr not in agg:
                continue
            for node in vw.get('ml_nodes', []) or []:
                ts = node.get('timeslot_allocation', []) or []
                w = int(str(node.get('poc_weight', '0')))
                agg[addr]['ml_nodes_total'] += 1
                agg[addr]['all_mlnode_weight'] += w
                if len(ts) > 0 and bool(ts[0]):
                    agg[addr]['verification_slot_nodes'] += 1
                    agg[addr]['verification_slot_weight'] += w
                if len(ts) > inference_slot_index and bool(ts[inference_slot_index]):
                    agg[addr]['inference_slot_nodes'] += 1
                    agg[addr]['inference_slot_weight'] += w
    return agg


def chain_decimal_to_decimal(v: Any) -> Decimal:
    if v is None:
        return Decimal(0)
    if isinstance(v, dict) and 'value' in v and 'exponent' in v:
        return Decimal(int(str(v.get('value', '0')))) * (Decimal(10) ** int(v.get('exponent', 0)))
    return Decimal(str(v))


def decimal_pow_int(base: Decimal, exp: int) -> Decimal:
    if exp < 0:
        raise ValueError('negative exponent not supported')
    result = Decimal(1)
    b = base
    e = exp
    while e > 0:
        if e & 1:
            result *= b
        b *= b
        e >>= 1
    return result


def get_exponent_for_decay(decay_rate: Decimal) -> Decimal:
    if decay_rate == Decimal('-0.000475'):
        return Decimal('0.9995251127946402')
    if decay_rate == Decimal('-0.000001'):
        return Decimal('0.9999990000005')
    if decay_rate == Decimal('0.0001'):
        return Decimal('1.0001000050001667')
    if decay_rate == Decimal('0'):
        return Decimal('1')
    raise ValueError(f'unsupported decay rate: {decay_rate}')


def calculate_fixed_epoch_reward(epoch: int, genesis_epoch: int, initial_epoch_reward: int, decay_rate: Decimal) -> int:
    if initial_epoch_reward <= 0:
        return 0
    epochs_since_genesis = max(epoch - genesis_epoch, 0)
    if epochs_since_genesis == 0:
        return initial_epoch_reward
    exponent = get_exponent_for_decay(decay_rate)
    reward = Decimal(initial_epoch_reward) * decimal_pow_int(exponent, epochs_since_genesis)
    return int(reward.to_integral_value(rounding=ROUND_DOWN))


def missed_stat_test(n_missed: int, n_total: int, p0: Decimal) -> bool:
    if n_total == 0:
        return True
    if n_missed < 0 or n_total < 0 or n_missed > n_total:
        return False
    # Chain uses lookup tables for these p0 values. For large n the lookup falls back to a ratio.
    permille = int((p0 * Decimal(1000)).to_integral_value(rounding=ROUND_DOWN))
    large_n_multipliers = {
        50: (1, 20),
        100: (1, 10),
        200: (1, 5),
        300: (3, 10),
        400: (2, 5),
        500: (1, 2),
    }
    if n_total > 1000 and permille in large_n_multipliers:
        num, den = large_n_multipliers[permille]
        return n_missed * den <= n_total * num

    # Exact one-sided binomial p-value P(X >= k), matching chain fallback behavior.
    if p0 <= 0 or p0 >= 1:
        return False
    if n_missed == 0:
        return True

    one = Decimal(1)
    q0 = one - p0

    # PMF at k: C(n,k) * p^k * q^(n-k)
    coeff = Decimal(1)
    for i in range(n_missed):
        coeff = coeff * Decimal(n_total - i) / Decimal(i + 1)
    prob = coeff * decimal_pow_int(p0, n_missed) * decimal_pow_int(q0, n_total - n_missed)
    total = prob
    ratio = p0 / q0
    for i in range(n_missed, n_total):
        factor = Decimal(n_total - i) / Decimal(i + 1)
        prob = prob * factor * ratio
        total += prob

    return total >= Decimal('0.05')


def apply_power_capping(weights_by_addr: dict[str, int]) -> dict[str, int]:
    items = [{'addr': a, 'weight': max(int(w), 0)} for a, w in weights_by_addr.items()]
    n = len(items)
    if n <= 1:
        return {x['addr']: x['weight'] for x in items}

    total_weight = sum(x['weight'] for x in items)
    if total_weight <= 0:
        return {x['addr']: 0 for x in items}

    max_pct = Decimal('0.30')
    if n == 1:
        max_pct = Decimal('1.0')
    elif n == 2:
        max_pct = Decimal('0.50')
    elif n == 3:
        max_pct = Decimal('0.40')

    powers = sorted(items, key=lambda x: x['weight'])
    cap = None
    sum_prev = 0
    for k in range(n):
        current_power = powers[k]['weight']
        weighted_total = sum_prev + current_power * (n - k)
        threshold = max_pct * Decimal(weighted_total)
        if Decimal(current_power) > threshold:
            numerator = max_pct * Decimal(sum_prev)
            denominator = Decimal(1) - max_pct * Decimal(n - k)
            if denominator <= 0:
                cap = current_power
            else:
                cap = int((numerator / denominator).to_integral_value(rounding=ROUND_DOWN))
            break
        sum_prev += current_power

    if cap is None:
        return {x['addr']: x['weight'] for x in items}
    return {x['addr']: min(x['weight'], cap) for x in items}


def render_upgrade_compensation_go(epoch: int, rows: list[dict[str, Any]]) -> str:
    amounts_by_addr: dict[str, int] = {}
    for r in rows:
        address = str(r['participant_index'])
        amount = int(str(r['expected_lost_reward_ngnk']))
        if amount <= 0:
            continue
        amounts_by_addr[address] = amounts_by_addr.get(address, 0) + amount

    # Fixed fee to proposal author (500 GNK) added to the same upgrade payout list.
    amounts_by_addr[PROPOSAL_AUTHOR_ADDRESS] = (
        amounts_by_addr.get(PROPOSAL_AUTHOR_ADDRESS, 0) + PROPOSAL_AUTHOR_FEE_NGONKA
    )

    positive = [{'address': addr, 'amount_ngonka': amt} for addr, amt in amounts_by_addr.items() if amt > 0]
    positive.sort(key=lambda x: x['address'])
    total = sum(x['amount_ngonka'] for x in positive)

    lines = []
    lines.append('// Auto-generated by scripts/check_inference_slot_rewards.py')
    lines.append(f'// Epoch: {epoch}')
    lines.append('// Compensation source: expected_lost_reward_ngnk')
    lines.append(f'// Includes proposal author fee: {PROPOSAL_AUTHOR_FEE_NGONKA:_} ngonka (500 GNK)')
    lines.append('// Denom: ngonka (1 GNK = 1_000_000_000 ngonka)')
    lines.append('')
    lines.append('type BountyReward struct {')
    lines.append('\tAddress string')
    lines.append('\tAmount  int64')
    lines.append('}')
    lines.append('')
    lines.append(f'var epoch{epoch}CompensationRewards = []BountyReward{{')
    for x in positive:
        amount_ngonka = x['amount_ngonka']
        amount_expr = f'{amount_ngonka:_}'
        amount_gnk = ngnk_to_gnk_str(str(amount_ngonka))
        lines.append(f'\t{{Address: "{x["address"]}", Amount: {amount_expr}}}, // {amount_gnk} GNK')
    lines.append('}')
    lines.append('')
    lines.append(f'// Total recipients: {len(positive)}')
    lines.append(f'// Total compensation: {total:_} ngonka')
    lines.append(f'// Total compensation: {dec_to_plain_str(Decimal(total) / Decimal(1_000_000_000))} GNK')
    lines.append('')
    return '\n'.join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Check inference-slot rewards and include lost-slot data from historical snapshot.'
    )
    parser.add_argument('--node', required=True, help='Validator node URL, e.g. http://node1.gonka.ai:8000')
    parser.add_argument('--epoch', type=int, required=True, help='Epoch number')
    parser.add_argument('--out-dir', default='artifacts', help='Output dir')
    parser.add_argument('--timeout-sec', type=float, default=40.0, help='HTTP timeout')
    parser.add_argument(
        '--inference-slot-index',
        type=int,
        default=1,
        help='Index in timeslot_allocation treated as inference slot (default: 1)',
    )
    args = parser.parse_args()

    base = args.node.rstrip('/')
    epoch = args.epoch
    timeout = args.timeout_sec
    inference_slot_index = args.inference_slot_index

    participants_payload = api_get(base, f'/api/v1/epochs/{epoch}/participants', timeout)
    participants = (participants_payload.get('active_participants') or {}).get('participants', []) or []
    excluded = participants_payload.get('excluded_participants', []) or []
    excluded_by_addr = {str(e.get('address', '')): str(e.get('reason', '')) for e in excluded}

    perf_payload = api_get(
        base,
        f'/chain-api/productscience/inference/inference/epoch_performance_summary/{epoch}',
        timeout,
    )
    perf_rows = perf_payload.get('epochPerformanceSummary', []) or []
    rewarded_by_addr = {
        str(r.get('participant_id', '')): int(str(r.get('rewarded_coins', '0')))
        for r in perf_rows
        if str(r.get('participant_id', ''))
    }
    perf_by_addr = {
        str(r.get('participant_id', '')): {
            'inference_count': int(str(r.get('inference_count', '0'))),
            'missed_requests': int(str(r.get('missed_requests', '0'))),
        }
        for r in perf_rows
        if str(r.get('participant_id', ''))
    }

    parent_epoch_group = api_get(
        base,
        f'/chain-api/productscience/inference/inference/epoch_group_data/{epoch}',
        timeout,
    ).get('epoch_group_data', {})
    parent_vw = {
        str(vw.get('member_address', '')): vw for vw in (parent_epoch_group.get('validation_weights', []) or [])
    }
    historical_parent_vw: dict[str, dict[str, Any]] = {}
    params_payload = api_get(base, '/chain-api/productscience/inference/inference/params', timeout)
    params = params_payload.get('params', {}) or {}
    bitcoin_params = params.get('bitcoin_reward_params', {}) or {}
    validation_params = params.get('validation_params', {}) or {}

    participant_ids = {str(p.get('index', '')) for p in participants}

    current_groups = fetch_all_epoch_groups(base, timeout)
    current_agg = aggregate_slot_stats(current_groups, epoch, participant_ids, inference_slot_index)

    effective_h = int(str(parent_epoch_group.get('effective_block_height', '0')))
    historical_agg = {
        addr: {'inference_slot_nodes': 0, 'inference_slot_weight': 0, 'all_mlnode_weight': 0} for addr in participant_ids
    }
    if effective_h > 0:
        historical_groups = fetch_all_epoch_groups(base, timeout, at_height=effective_h)
        hist_full = aggregate_slot_stats(historical_groups, epoch, participant_ids, inference_slot_index)
        historical_parent_epoch_group = api_get(
            base,
            f'/chain-api/productscience/inference/inference/epoch_group_data/{epoch}',
            timeout,
            headers={'x-cosmos-block-height': str(effective_h)},
        ).get('epoch_group_data', {})
        historical_parent_vw = {
            str(vw.get('member_address', '')): vw
            for vw in (historical_parent_epoch_group.get('validation_weights', []) or [])
        }
        for addr in participant_ids:
            historical_agg[addr] = {
                'inference_slot_nodes': int(hist_full.get(addr, {}).get('inference_slot_nodes', 0)),
                'inference_slot_weight': int(hist_full.get(addr, {}).get('inference_slot_weight', 0)),
                'all_mlnode_weight': int(hist_full.get(addr, {}).get('all_mlnode_weight', 0)),
            }

    rows: list[dict[str, Any]] = []
    for p in sorted(participants, key=lambda x: str(x.get('index', ''))):
        addr = str(p.get('index', ''))
        reward = rewarded_by_addr.get(addr, 0)
        cur = current_agg.get(addr, {})
        hist = historical_agg.get(addr, {})

        inf_w = int(cur.get('inference_slot_weight', 0))
        inf_n = int(cur.get('inference_slot_nodes', 0))
        hist_inf_w = int(hist.get('inference_slot_weight', 0))
        hist_inf_n = int(hist.get('inference_slot_nodes', 0))
        perf = perf_by_addr.get(addr, {})

        rows.append(
            {
                'index': addr,
                'rewarded_coins': reward,
                'inference_count': int(perf.get('inference_count', 0)),
                'missed_requests': int(perf.get('missed_requests', 0)),
                'inference_slot_nodes': inf_n,
                'inference_slot_weight': inf_w,
                'historical_inference_slot_nodes': hist_inf_n,
                'historical_inference_slot_weight': hist_inf_w,
                'lost_inference_slot_nodes': max(hist_inf_n - inf_n, 0),
                'lost_inference_slot_weight': max(hist_inf_w - inf_w, 0),
                'verification_slot_nodes': int(cur.get('verification_slot_nodes', 0)),
                'verification_slot_weight': int(cur.get('verification_slot_weight', 0)),
                'all_mlnode_weight': int(cur.get('all_mlnode_weight', 0)),
                'excluded_reason': excluded_by_addr.get(addr, ''),
                'parent_base_weight': int(str((parent_vw.get(addr) or {}).get('weight', '0'))),
                'parent_confirmation_weight': int(str((parent_vw.get(addr) or {}).get('confirmation_weight', '0'))),
                'historical_parent_confirmation_weight': int(
                    str((historical_parent_vw.get(addr) or {}).get('confirmation_weight', '0'))
                ),
                'likely_missed_reward_for_inference_slot': inf_w > 0 and reward == 0,
            }
        )

    out_dir = os.path.join(args.out_dir, f'epoch_{epoch}')
    os.makedirs(out_dir, exist_ok=True)
    split_csv_path = os.path.join(out_dir, 'inference_slot_reward_columns.csv')
    upgrade_go_path = os.path.join(out_dir, f'epoch_{epoch}_upgrade_compensation_rewards.go.txt')

    # Simulate chain Bitcoin reward distribution formula.
    initial_epoch_reward = int(str(bitcoin_params.get('initial_epoch_reward', '0')))
    genesis_epoch = int(str(bitcoin_params.get('genesis_epoch', '1')))
    decay_rate = chain_decimal_to_decimal(bitcoin_params.get('decay_rate'))
    fixed_epoch_reward = calculate_fixed_epoch_reward(epoch, genesis_epoch, initial_epoch_reward, decay_rate)
    p0 = chain_decimal_to_decimal(validation_params.get('binom_test_p0'))
    if p0 == 0:
        p0 = Decimal('0.1')

    full_weight_by_addr: dict[str, int] = {}
    effective_weight_by_addr: dict[str, int] = {}
    for r in rows:
        addr = str(r['index'])
        vw = parent_vw.get(addr) or {}
        full_weight = max(int(str(vw.get('weight', '0'))), 0)
        confirmation_weight = max(int(str(vw.get('confirmation_weight', '0'))), 0)
        preserved_weight = max(int(r['inference_slot_weight']), 0)
        full_weight_by_addr[addr] = full_weight
        if str(r.get('excluded_reason', '')):
            # Mimic chain invalid/inactive participant handling:
            # full weight remains in denominator, but participant receives zero distribution.
            effective_weight_by_addr[addr] = 0
        else:
            effective_weight_by_addr[addr] = preserved_weight + confirmation_weight

    capped_weight_by_addr = apply_power_capping(effective_weight_by_addr)
    downtime_weight_by_addr: dict[str, int] = {}
    for r in rows:
        addr = str(r['index'])
        n_inf = int(r['inference_count'])
        n_missed = int(r['missed_requests'])
        n_total = n_inf + n_missed
        w = capped_weight_by_addr.get(addr, 0)
        downtime_weight_by_addr[addr] = w if missed_stat_test(n_missed, n_total, p0) else 0

    total_full_weight = sum(full_weight_by_addr.values())
    simulated_reward_by_addr: dict[str, int] = {}
    for r in rows:
        addr = str(r['index'])
        w = downtime_weight_by_addr.get(addr, 0)
        if total_full_weight > 0 and w > 0:
            simulated_reward_by_addr[addr] = int((w * fixed_epoch_reward) // total_full_weight)
        else:
            simulated_reward_by_addr[addr] = 0

    split_rows_raw = []
    for r in rows:
        addr = str(r['index'])
        all_w = int(r['all_mlnode_weight'])
        # Display historical inference-slot weight (at epoch effective height),
        # because current slot assignment for epoch 158 was reset.
        inf_w = int(r['historical_inference_slot_weight'])
        current_inf_w = int(r['inference_slot_weight'])
        non_inf_w = int(r['parent_confirmation_weight'])
        hist_inf_w = int(r['historical_inference_slot_weight'])
        hist_non_inf_w = int(r['historical_parent_confirmation_weight'])
        lost_inf_w = max(hist_inf_w - current_inf_w, 0)
        real_reward = int(r['rewarded_coins'])
        simulated_reward = int(simulated_reward_by_addr.get(addr, 0))

        split_rows_raw.append(
            {
                'real_reward_chain': str(real_reward),
                'simulated_reward_chain_formula': str(simulated_reward),
                'calculated_reward_inference_slot_nodes': '0',
                'calculated_reward_non_inference_slot_nodes': '0',
                'ml_nodes_weight_inference_slot': str(inf_w),
                'ml_nodes_weight_non_inference_slot': str(non_inf_w),
                'historical_ml_nodes_weight_inference_slot': str(hist_inf_w),
                'historical_ml_nodes_weight_non_inference_slot': str(hist_non_inf_w),
                'lost_ml_nodes_weight_inference_slot': str(lost_inf_w),
                'historical_calculated_reward_inference_slot_nodes': '0',
                'participant_index': addr,
            }
        )

    # Single global reward-per-weight coefficient used for all nodes.
    total_simulated_reward = sum(Decimal(x['simulated_reward_chain_formula']) for x in split_rows_raw)
    total_non_inf_weight = sum(Decimal(x['ml_nodes_weight_non_inference_slot']) for x in split_rows_raw)
    global_coeff = (total_simulated_reward / total_non_inf_weight) if total_non_inf_weight > 0 else Decimal(0)

    split_rows = []
    for x in split_rows_raw:
        inf_w = Decimal(x['ml_nodes_weight_inference_slot'])
        non_inf_w = Decimal(x['ml_nodes_weight_non_inference_slot'])
        hist_inf_w = Decimal(x['historical_ml_nodes_weight_inference_slot'])
        lost_w = Decimal(x['lost_ml_nodes_weight_inference_slot'])
        inf_calc = inf_w * global_coeff
        non_inf_calc = non_inf_w * global_coeff
        hist_inf_calc = hist_inf_w * global_coeff
        estimated_lost = lost_w * global_coeff
        row = dict(x)
        row['calculated_reward_inference_slot_nodes'] = dec_to_int_str(inf_calc)
        row['calculated_reward_non_inference_slot_nodes'] = dec_to_int_str(non_inf_calc)
        row['historical_calculated_reward_inference_slot_nodes'] = dec_to_int_str(hist_inf_calc)
        row['estimated_lost_reward_due_to_lost_inference_slots'] = dec_to_int_str(estimated_lost)
        row['calculated_reward_inference_slot_nodes_gnk'] = ngnk_to_gnk_str(
            row['calculated_reward_inference_slot_nodes']
        )
        row['estimated_lost_reward_due_to_lost_inference_slots_gnk'] = ngnk_to_gnk_str(
            row['estimated_lost_reward_due_to_lost_inference_slots']
        )
        row['expected_lost_reward_ngnk'] = row['estimated_lost_reward_due_to_lost_inference_slots']
        row['expected_lost_reward_gnk'] = row['estimated_lost_reward_due_to_lost_inference_slots_gnk']
        split_rows.append(row)

    with open(split_csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(
            f,
            extrasaction='ignore',
            fieldnames=[
                'real_reward_chain',
                'simulated_reward_chain_formula',
                'calculated_reward_non_inference_slot_nodes',
                'ml_nodes_weight_non_inference_slot',
                'ml_nodes_weight_inference_slot',
                'expected_lost_reward_ngnk',
                'expected_lost_reward_gnk',
                'participant_index',
            ],
        )
        writer.writeheader()
        writer.writerows(split_rows)

    with open(upgrade_go_path, 'w', encoding='utf-8') as f:
        f.write(render_upgrade_compensation_go(epoch, split_rows))

    participants_with_inf_slot = sum(1 for r in rows if int(r['inference_slot_weight']) > 0)
    rewarded_with_inf_slot = sum(1 for r in rows if int(r['inference_slot_weight']) > 0 and int(r['rewarded_coins']) > 0)
    missed = sum(1 for r in rows if int(r['inference_slot_weight']) > 0 and int(r['rewarded_coins']) == 0)

    summary = {
        'epoch': epoch,
        'inference_slot_index_used': inference_slot_index,
        'historical_effective_block_height_used': effective_h,
        'global_reward_per_weight_coefficient': dec_to_plain_str(global_coeff),
        'fixed_epoch_reward_formula': str(fixed_epoch_reward),
        'simulated_distributed_total': str(sum(int(r['simulated_reward_chain_formula']) for r in split_rows)),
        'real_distributed_total': str(sum(int(r['real_reward_chain']) for r in split_rows)),
        'distribution_gap_real_minus_simulated': str(
            sum(int(r['real_reward_chain']) for r in split_rows) - sum(int(r['simulated_reward_chain_formula']) for r in split_rows)
        ),
        'participants_reward_mismatch_count': sum(
            1 for r in split_rows if int(r['real_reward_chain']) != int(r['simulated_reward_chain_formula'])
        ),
        'participants_total': len(rows),
        'participants_with_inference_slot_weight': participants_with_inf_slot,
        'participants_with_inference_slot_weight_and_rewarded': rewarded_with_inf_slot,
        'participants_with_inference_slot_weight_but_zero_reward': missed,
        'participants_with_lost_inference_slot_weight': sum(1 for r in rows if int(r['lost_inference_slot_weight']) > 0),
        'total_lost_inference_slot_weight': sum(int(r['lost_inference_slot_weight']) for r in rows),
        'real_reward_calculated_by_inference_slot_ml_nodes': dec_to_plain_str(
            sum(Decimal(r['calculated_reward_inference_slot_nodes']) for r in split_rows)
        ),
        'real_reward_calculated_by_non_inference_slot_ml_nodes': dec_to_plain_str(
            sum(Decimal(r['calculated_reward_non_inference_slot_nodes']) for r in split_rows)
        ),
        'upgrade_compensation_go_file': upgrade_go_path,
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
