# Epoch 158: Reward Distribution Issue After `v0.2.9` Upgrade

## Context

During upgrade `v0.2.9`, `POC_SLOT` (inference slot) was reset for the effective epoch:

- `resetPocSlotsForEffectiveEpoch`:
  - sets `TimeslotAllocation[1] = false` in `ActiveParticipants`
  - see `upgrades.go#L268`
- `resetPocSlotsInEpochGroupData`:
  - sets `TimeslotAllocation[1] = false` in model subgroup `EpochGroupData`
  - see `upgrades.go#L343-L344`

Reference:  
`https://github.com/gonka-ai/gonka/blob/upgrade-v0.2.9/inference-chain/app/upgrades/v0_2_9/upgrades.go#L268`

## Problem

Settlement reward logic uses preserved weight derived from `TimeslotAllocation[1]`:

- `preservedWeight` is computed from ML nodes with `POC_SLOT=true`
- `effectiveWeight = preservedWeight + confirmationWeight`
- see `x/inference/keeper/bitcoin_rewards.go`

After the reset, preserved weight for those nodes became zero in effective-epoch data.  
For validators whose weight was mainly in inference-slot (`preserved`) nodes, this reduced their effective reward share (in some cases to zero).

## Root Cause

The reset was introduced as a technical migration step for PoC V2 (to force all nodes into the first V2 PoC round).  
However, it was applied to effective-epoch structures that also affected reward calculation for that period.

Result: the original historical inference-slot allocation for epoch 158 was overwritten in current-state views.

## What This Repository Provides

The script `scripts/check_inference_slot_rewards.py`:

1. Pulls epoch 158 data directly from validator nodes.
2. Reconstructs historical inference-slot state at `effective_block_height`.
3. Matches real chain rewards vs formula simulation.
4. Calculates expected lost reward caused by lost inference-slot preserved weight.
5. Generates:
   - `artifacts/epoch_158/inference_slot_reward_columns.csv` (audit table)
   - `artifacts/epoch_158/epoch_158_upgrade_compensation_rewards.go.txt` (upgrade-ready payout list)

## Why This Matters

This approach provides:

- reproducible evidence of who was underpaid and by how much;
- a transparent address/amount list for on-chain compensation via a new upgrade;
- output format aligned with existing upgrade payout pattern (`[]BountyReward`).

## Amount Format

- Base chain denom: `ngonka`
- `1 GNK = 1_000_000_000 ngonka`
- In `epoch_158_upgrade_compensation_rewards.go.txt`:
  - `Amount` is in `ngonka` (exact on-chain unit)
  - each line also includes a `GNK` comment for readability

## How To Run

### Prerequisites

- Python 3.10+ (standard library only, no external packages required)
- Access to at least one validator node API (for example: `http://node1.gonka.ai:8000`)

### Command

Run from repository root:

```bash
python3 scripts/check_inference_slot_rewards.py \
  --node http://node1.gonka.ai:8000 \
  --epoch 158 \
  --out-dir artifacts
```

Optional flag:

- `--inference-slot-index 1` (default is `1`, which corresponds to `POC_SLOT`)

### What The Script Produces

- `artifacts/epoch_158/inference_slot_reward_columns.csv`
  - audit table with real chain reward, formula simulation, non-inference-slot calculation, inference/non-inference weights, and expected lost reward (`ngonka` + `GNK`)
- `artifacts/epoch_158/epoch_158_upgrade_compensation_rewards.go.txt`
  - upgrade-ready `[]BountyReward` list for compensation distribution

### Quick Validation

After run, the script prints JSON summary to stdout.  
Key checks:

- `participants_reward_mismatch_count` should be `0` when simulation matches real chain distribution.
- `distribution_gap_real_minus_simulated` should be `0` when totals match.
