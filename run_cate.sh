#!/bin/bash
# =============================================================================
# run_cate.sh -- ONE (J, alpha) cell: all cases and all m, 200 replications.
#
#   sbatch run_cate.sh --J 50 --alpha 0.80 --cases 1,2,3 --ms 4,5,6,J
#
# The SLURM array runs over REPLICATIONS (0-199).  Inside each array task the
# script loops over every case and every m, so a single submission fills the
# whole cell:  200 reps x 3 cases x 4 m = 2400 fits, 12 per task.
#
# Each configuration writes its own CSV, tagged with case and m, so
# combine_cate.py picks them all up afterwards.
#
# Already-finished configurations are skipped, so a timed-out or failed job
# can be resubmitted unchanged and will only redo what is missing.  Pass
# --force to recompute everything.
# =============================================================================
#SBATCH --job-name=cate_cell
#SBATCH --output=logs/cate_%A_%a.out
#SBATCH --error=logs/cate_%A_%a.err
#SBATCH --array=0-199
#SBATCH --time=08:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --partition=standard
#SBATCH --account=yili2
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ghosal@umich.edu

set -uo pipefail
mkdir -p logs

module load python/3.9
source .DPCM/bin/activate

# ---- defaults ---------------------------------------------------------------
J=50                 # number of evaluation times   (paper: 25, 50, 100)
ALPHA=0.80           # subsample exponent, r = n^alpha
CASES="1,2,3"        # 1: correct S / correct e   2: wrong e   3: wrong S
MS="4,5,6,J"         # basis dimensions; the literal J means m = J, the
                     # cardinal basis, i.e. an unconstrained J-output network
B=1000               # bags
DEGREE=3             # B-spline degree
H1=128
H2=64
N_JOBS=8             # must match --cpus-per-task
OUT_DIR="results"
FORCE=0

# ---- parse ------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case $1 in
    --J)       J="$2";       shift 2 ;;
    --alpha)   ALPHA="$2";   shift 2 ;;
    --cases)   CASES="$2";   shift 2 ;;
    --ms)      MS="$2";      shift 2 ;;
    --B)       B="$2";       shift 2 ;;
    --degree)  DEGREE="$2";  shift 2 ;;
    --h1)      H1="$2";      shift 2 ;;
    --h2)      H2="$2";      shift 2 ;;
    --n_jobs)  N_JOBS="$2";  shift 2 ;;
    --out_dir) OUT_DIR="$2"; shift 2 ;;
    --force)   FORCE=1;      shift 1 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

REP_ID=${SLURM_ARRAY_TASK_ID:-0}
mkdir -p "${OUT_DIR}"

IFS=',' read -ra CASE_ARR <<< "${CASES}"
IFS=',' read -ra M_ARR    <<< "${MS}"

echo "=============================================================="
echo "  Job ${SLURM_JOB_ID:-local}   task ${REP_ID}   $(hostname)"
echo "  start   : $(date)"
echo "  J       = ${J}        alpha = ${ALPHA}"
echo "  cases   = ${CASES}"
echo "  m       = ${MS}       (J -> cardinal basis, m = ${J})"
echo "  B       = ${B}        width = ${H1}x${H2}"
echo "  out_dir = ${OUT_DIR}"
echo "=============================================================="

N_OK=0; N_SKIP=0; N_FAIL=0
ALPHA_FMT=$(printf "%.2f" "${ALPHA}")

for CASE in "${CASE_ARR[@]}"; do
  for MRAW in "${M_ARR[@]}"; do

    # the literal token J means "use the cardinal basis"
    if [[ "${MRAW}" == "J" || "${MRAW}" == "j" ]]; then M="${J}"; else M="${MRAW}"; fi

    if (( M > J )); then
      echo "  [skip] m=${M} > J=${J}: readout would be rank deficient"
      N_SKIP=$((N_SKIP+1)); continue
    fi

    TAG="c${CASE}_a${ALPHA_FMT}_J${J}_m${M}_h${H1}x${H2}_B${B}_rep${REP_ID}"
    FILE="${OUT_DIR}/cate_${TAG}.csv"

    if [[ -s "${FILE}" && ${FORCE} -eq 0 ]]; then
      echo "  [have] case=${CASE} m=${M}  ->  $(basename "${FILE}")"
      N_SKIP=$((N_SKIP+1)); continue
    fi

    echo "  ------------------------------------------------------------"
    echo "  [run ] case=${CASE}  m=${M}  rep=${REP_ID}   $(date +%H:%M:%S)"
    python cate_esm.py \
      --rep_id "${REP_ID}" --case "${CASE}" --alpha "${ALPHA}" \
      --J "${J}" --m "${M}" --B "${B}" --degree "${DEGREE}" \
      --h1 "${H1}" --h2 "${H2}" --n_jobs "${N_JOBS}" \
      --out_dir "${OUT_DIR}"

    if [[ $? -eq 0 ]]; then N_OK=$((N_OK+1)); else
      echo "  [FAIL] case=${CASE} m=${M}"; N_FAIL=$((N_FAIL+1)); fi
  done
done

echo "=============================================================="
echo "  task ${REP_ID} done: ${N_OK} run, ${N_SKIP} skipped, ${N_FAIL} failed"
echo "  finish  : $(date)"
echo "=============================================================="
exit $(( N_FAIL > 0 ? 1 : 0 ))
