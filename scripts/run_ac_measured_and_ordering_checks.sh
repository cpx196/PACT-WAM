#!/usr/bin/env bash
set -euo pipefail
cd /data/pxchen/PACT-WAM
exec 9>artifacts/libero10_ac_measured_condition_diagnostic_20261007/worker.lock
flock -n 9
/data/pxchen/envs/jepa-wam/bin/python -u scripts/diagnose_libero10_ac_measured_conditions.py >>artifacts/libero10_ac_measured_condition_diagnostic_20261007/run.log 2>&1
test ! -e artifacts/libero10_jepa_action_ordering_gripper_fixed_20261007
mkdir artifacts/libero10_jepa_action_ordering_gripper_fixed_20261007
/data/pxchen/envs/jepa-wam/bin/python -u scripts/diagnose_libero10_jepa_action_ordering.py >artifacts/libero10_jepa_action_ordering_gripper_fixed_20261007/run.log 2>&1
