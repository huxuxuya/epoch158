# Coin Amount Computation Logic (Epoch 158 Tool)

This document describes exactly how `scripts/check_inference_slot_rewards.py` computes coin amounts.

## Simple Reverse Explanation

For epoch 158, we first take the total amount distributed by the reward simulation for that epoch: **252,286,759,171,835 ngonka**.
Then we divide it by the total non-preserved (confirmation) weight used for epoch 158: **4,981,565**.
This gives a single conversion rate: **50,644,076.544586889446 ngonka per 1 weight unit**.
In simple terms, this is the average value of one weight unit in epoch 158.

After that, we calculate lost preserved weight for each participant.
We restore historical preserved weight from the chain snapshot at block **2,443,438** (the effective block of epoch 158), and compare it to the preserved weight of epoch 158 in current-state epoch data (after reset).
If historical value is higher, the difference is the lost preserved weight; otherwise loss is zero.
Compensation is computed as this lost preserved weight multiplied by the single per-weight rate, then rounded to whole ngonka.

Example: if lost preserved weight is **4,663**, then
**4,663 × 50,644,076.544586889446 = 236,153,328,927.39...**,
so final compensation is **236,153,328,927 ngonka** (**236.153328927 GNK**).
If lost preserved weight is zero, compensation is zero.

Finally, after all participant compensations are calculated, an additional fixed payment of **500 GNK** is added to the proposal author.

## Scope

The script computes three related amounts:

1. Chain-formula simulated reward per participant (`simulated_reward_chain_formula`).
2. Estimated lost reward due to lost inference-slot weight (`expected_lost_reward_ngnk`).
3. Final upgrade payout amounts in Go output (`Amount` in `[]BountyReward`).

## Units

- Base unit: `ngonka` (integer).
- Display unit: `GNK`.
- Conversion: `1 GNK = 1_000_000_000 ngonka`.

## Inputs Used

For each participant address:

- `full_weight` = `validation_weights[].weight` (parent epoch group).
- `confirmation_weight` = `validation_weights[].confirmation_weight` (parent epoch group).
- `inference_slot_weight` from `ml_nodes[].poc_weight` where `timeslot_allocation[inference_slot_index] == true`.
- `historical_inference_slot_weight` from the same query at `x-cosmos-block-height = effective_block_height`.
- `rewarded_coins`, `inference_count`, `missed_requests` from epoch performance summary.
- `excluded_reason` from epoch participants endpoint.

Global params:

- `initial_epoch_reward`, `genesis_epoch`, `decay_rate`.
- `binom_test_p0` (fallback to `0.1` if zero).

## Step 1: Fixed Epoch Reward

The script computes:

- `epochs_since_genesis = max(epoch - genesis_epoch, 0)`
- `fixed_epoch_reward = floor(initial_epoch_reward * exponent^epochs_since_genesis)`

where `exponent` is selected by decay-rate mapping in `get_exponent_for_decay`.

Rounding: floor (`ROUND_DOWN`) to integer `ngonka`.

## Step 2: Simulated Chain Reward (Per Participant)

For each participant:

1. `preserved_weight = max(inference_slot_weight, 0)`
2. If excluded (`excluded_reason != ""`): `effective_weight = 0`
3. Else: `effective_weight = preserved_weight + max(confirmation_weight, 0)`

Then:

1. Apply power capping to all `effective_weight` values (`apply_power_capping`).
2. Apply missed-request statistical filter (`missed_stat_test`):
   - pass -> keep capped weight
   - fail -> set weight to `0`
3. Let resulting weight be `w_i`.
4. Let `total_full_weight = sum(max(full_weight, 0))` across all participants.
5. Simulated reward:
   - if `total_full_weight > 0` and `w_i > 0`:
     - `simulated_reward_i = floor(w_i * fixed_epoch_reward / total_full_weight)`
   - else `0`.

Rounding: integer floor because Python `//` is used on integers.

## Step 3: Global Reward-Per-Weight Coefficient

Build raw table values:

- `non_inf_weight_i = parent_confirmation_weight`
- `hist_inf_weight_i = historical_inference_slot_weight`
- `current_inf_weight_i = inference_slot_weight`
- `lost_inf_weight_i = max(hist_inf_weight_i - current_inf_weight_i, 0)`

Compute:

- `total_simulated_reward = sum(simulated_reward_i)`
- `total_non_inf_weight = sum(non_inf_weight_i)`
- `global_coeff = total_simulated_reward / total_non_inf_weight` (or `0` if denominator is `0`)

## Step 4: Estimated Lost Reward (Compensation Basis)

For each participant:

- `estimated_lost_i = lost_inf_weight_i * global_coeff`
- `expected_lost_reward_ngnk_i = round_half_up(estimated_lost_i)` to integer

Implementation detail:

- `dec_to_int_str` uses `Decimal.quantize(..., ROUND_HALF_UP)`.

Also computed (for audit):

- `calculated_reward_non_inference_slot_nodes = round_half_up(non_inf_weight_i * global_coeff)`
- `calculated_reward_inference_slot_nodes = round_half_up(hist_inf_weight_i * global_coeff)`

## Step 5: Final Upgrade Payout Amount

When generating `epoch_158_upgrade_compensation_rewards.go.txt`:

1. Start with `amount_i = expected_lost_reward_ngnk_i`.
2. Exclude non-positive amounts (`<= 0`).
3. Aggregate by address (sum if duplicates exist).
4. Add fixed proposal-author fee:
   - address: `gonka1t7mcnc8zjkkvhwmfmst54sasulj68e5zsv4yzu`
   - amount: `500 * 1_000_000_000 = 500_000_000_000 ngonka`
5. Sort by address and emit:
   - `Amount` in integer `ngonka`
   - GNK comment for readability only.

## Output Fields That Represent Coin Amount

- CSV (`inference_slot_reward_columns.csv`):
  - `real_reward_chain`
  - `simulated_reward_chain_formula`
  - `calculated_reward_non_inference_slot_nodes`
  - `expected_lost_reward_ngnk`
  - `expected_lost_reward_gnk`

- Go payout file (`epoch_158_upgrade_compensation_rewards.go.txt`):
  - `BountyReward.Amount` is the exact on-chain payout amount in `ngonka`.
