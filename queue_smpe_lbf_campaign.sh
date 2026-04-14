#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

SBATCH_FILE="${SCRIPT_DIR}/train_smpe_lbf_campaign.sbatch"
MODE="${1:-all}"
DRY_RUN="${DRY_RUN:-0}"
REPLICATES="${REPLICATES:-10}"
CAMPAIGN_SEED="${CAMPAIGN_SEED:-$(( (10#$(date +%s) ^ (RANDOM << 16) ^ RANDOM) & 2147483647 ))}"
RUNS_PER_JOB="${RUNS_PER_JOB:-1}"

ALGO_COUNT=1
TASK_COUNT=9
RUNS_PER_REPLICATE="$((ALGO_COUNT * TASK_COUNT))"
TOTAL_RUNS="$((RUNS_PER_REPLICATE * REPLICATES))"
JOB_COUNT="$(((TOTAL_RUNS + RUNS_PER_JOB - 1) / RUNS_PER_JOB))"

CPU_ACCOUNT="huytran1-ic"
CPU_PARTITION="IllinoisComputes"
CPU_TIME="${CPU_TIME:-18:00:00}"
CPU_CONCURRENCY="${CPU_CONCURRENCY:-40}"

# Current known hard cap from association for niketnp2+huytran1-ic:
# MaxJobs=40, so keep array throttle <= 40.
CPU_CONCURRENCY_HARD_CAP=40

if [[ ! -f "${SBATCH_FILE}" ]]; then
  echo "Could not find ${SBATCH_FILE}"
  exit 1
fi

if ! [[ "${CPU_CONCURRENCY}" =~ ^[0-9]+$ && "${DRY_RUN}" =~ ^[0-9]+$ && "${REPLICATES}" =~ ^[1-9][0-9]*$ && "${CAMPAIGN_SEED}" =~ ^[0-9]+$ && "${RUNS_PER_JOB}" =~ ^[1-9][0-9]*$ ]]; then
  echo "CPU_CONCURRENCY, DRY_RUN, REPLICATES, CAMPAIGN_SEED, and RUNS_PER_JOB must be valid integers."
  exit 1
fi

if (( CPU_CONCURRENCY < 1 )); then
  echo "CPU_CONCURRENCY must be >= 1."
  exit 1
fi

if (( CPU_CONCURRENCY > CPU_CONCURRENCY_HARD_CAP )); then
  echo "CPU_CONCURRENCY=${CPU_CONCURRENCY} exceeds cap ${CPU_CONCURRENCY_HARD_CAP}; clamping."
  CPU_CONCURRENCY="${CPU_CONCURRENCY_HARD_CAP}"
fi

mkdir -p "${SCRIPT_DIR}/train_sbatch_outputs"

submit_array() {
  local account="$1"
  local partition="$2"
  local walltime="$3"
  local array_spec="$4"
  local export_vars="ALL,REPLICATES=${REPLICATES},CAMPAIGN_SEED=${CAMPAIGN_SEED},RUNS_PER_JOB=${RUNS_PER_JOB}"

  if (( DRY_RUN == 1 )); then
    sbatch --test-only --account="${account}" --partition="${partition}" --time="${walltime}" --array="${array_spec}" --export="${export_vars}" "${SBATCH_FILE}"
  else
    sbatch --parsable --account="${account}" --partition="${partition}" --time="${walltime}" --array="${array_spec}" --export="${export_vars}" "${SBATCH_FILE}"
  fi
}

case "${MODE}" in
  all|cpu-only|ic-only)
    array_spec="0-$((JOB_COUNT - 1))%${CPU_CONCURRENCY}"
    result="$(submit_array "${CPU_ACCOUNT}" "${CPU_PARTITION}" "${CPU_TIME}" "${array_spec}")"
    echo "Submitted SMPE LBF campaign to ${CPU_PARTITION}: ${result}"
    echo "  replicates=${REPLICATES}, campaign_seed=${CAMPAIGN_SEED}, runs_per_job=${RUNS_PER_JOB}"
    echo "  total_runs=${TOTAL_RUNS}, total_jobs=${JOB_COUNT}, concurrency=${CPU_CONCURRENCY}"
    ;;
  first5)
    if (( RUNS_PER_JOB != 1 )); then
      echo "first5 mode requires RUNS_PER_JOB=1 to avoid boundary overlap."
      exit 1
    fi
    first_rep_count=$((REPLICATES < 5 ? REPLICATES : 5))
    array_spec="0-$((RUNS_PER_REPLICATE * first_rep_count - 1))%${CPU_CONCURRENCY}"
    result="$(submit_array "${CPU_ACCOUNT}" "${CPU_PARTITION}" "${CPU_TIME}" "${array_spec}")"
    echo "Submitted first ${first_rep_count} replicates to ${CPU_PARTITION}: ${result} (array ${array_spec}, campaign_seed=${CAMPAIGN_SEED})"
    ;;
  second5)
    if (( RUNS_PER_JOB != 1 )); then
      echo "second5 mode requires RUNS_PER_JOB=1 to avoid boundary overlap."
      exit 1
    fi
    if (( REPLICATES <= 5 )); then
      echo "No second-5 segment available because REPLICATES=${REPLICATES}."
      exit 1
    fi
    second_rep_end=$((REPLICATES < 10 ? REPLICATES : 10))
    array_start="$((RUNS_PER_REPLICATE * 5))"
    array_end="$((RUNS_PER_REPLICATE * second_rep_end - 1))"
    array_spec="${array_start}-${array_end}%${CPU_CONCURRENCY}"
    result="$(submit_array "${CPU_ACCOUNT}" "${CPU_PARTITION}" "${CPU_TIME}" "${array_spec}")"
    echo "Submitted replicates 6-${second_rep_end} to ${CPU_PARTITION}: ${result} (array ${array_spec}, campaign_seed=${CAMPAIGN_SEED})"
    ;;
  *)
    echo "Usage: $0 [all|cpu-only|ic-only|first5|second5]"
    echo "Optional env overrides: REPLICATES, CAMPAIGN_SEED, RUNS_PER_JOB, CPU_CONCURRENCY, CPU_TIME, DRY_RUN=1"
    exit 1
    ;;
esac
