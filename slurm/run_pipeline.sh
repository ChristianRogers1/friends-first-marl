#!/bin/bash
#SBATCH --account=rai
#SBATCH --partition=rai-gpu-grn
#SBATCH --qos=rai-gpu-grn
#SBATCH --time=20:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
#SBATCH -o slurmjob-%j.out-%N
#SBATCH -e slurmjob-%j.err-%N


# ============================================================
#  Full experimental pipeline SLURM job
#
#  Supports multi-GPU parallelism: if you request N GPUs with
#  --gres=gpu:N, the pipeline distributes independent training
#  runs across all N GPUs automatically.
#
#  Runs the entire screen-and-confirm procedure:
#    1. Screening  -- 4 ratios x 5 seeds  = 20 training runs
#    2. Confirmation -- best ratio x 12 seeds + baseline x 12 = 24 runs
#    3. Evaluation -- H1/H2/H3 metrics on all confirmation agents
#    4. Analysis   -- Generate figures with matplotlib
#
#  Usage (single GPU, sequential):
#    sbatch slurm/run_pipeline.sh
#
#  Usage (multi-GPU, parallel):
#    sbatch --gres=gpu:4 slurm/run_pipeline.sh
#    sbatch --gres=gpu:2 --export=TOTAL_EPISODES=10000 slurm/run_pipeline.sh
# ============================================================

echo "============================================"
echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $SLURMD_NODENAME"
echo "Start time  : $(date)"
echo "Working dir : $SLURM_SUBMIT_DIR"
echo "============================================"

# ---------- tunables (override via --export) ----------
TOTAL_EPISODES=${TOTAL_EPISODES:-50000}
CONFIG=${CONFIG:-configs/decoupled_sweep.yaml}
OUTPUT_DIR=${OUTPUT_DIR:-results}

# Auto-detect GPU count from SLURM allocation (0 = auto-detect in Python)
NUM_GPUS=${NUM_GPUS:-0}

echo "TOTAL_EPISODES = $TOTAL_EPISODES"
echo "CONFIG         = $CONFIG"
echo "OUTPUT_DIR     = $OUTPUT_DIR"
echo "NUM_GPUS       = $NUM_GPUS  (0 = auto-detect)"

# ---------- scratch directory ----------
SCRDIR=/scratch/general/vast/$USER/$SLURM_JOB_ID
mkdir -p $SCRDIR

# Copy the full project to scratch
cp -r $SLURM_SUBMIT_DIR/src        $SCRDIR/
cp -r $SLURM_SUBMIT_DIR/configs    $SCRDIR/
cp -r $SLURM_SUBMIT_DIR/scripts    $SCRDIR/
cp -r $SLURM_SUBMIT_DIR/analysis   $SCRDIR/
cp    $SLURM_SUBMIT_DIR/requirements.txt $SCRDIR/
cp    $SLURM_SUBMIT_DIR/pyproject.toml   $SCRDIR/

cd $SCRDIR

# ---------- python environment ----------
module purge 2>/dev/null || true
module load python/3.12.4
module load cuda/12.4

python -m venv venv
source venv/bin/activate

pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet

# ---------- GPU check ----------
python - <<'PYCHECK'
import torch, sys
print(f"PyTorch  : {torch.__version__}")
print(f"CUDA     : {torch.version.cuda}")
if not torch.cuda.is_available():
    print("ERROR: CUDA not available, aborting job")
    sys.exit(1)
n = torch.cuda.device_count()
print(f"GPUs     : {n}")
for i in range(n):
    name = torch.cuda.get_device_name(i)
    vram = torch.cuda.get_device_properties(i).total_memory / 1e9
    print(f"  cuda:{i} : {name}  ({vram:.1f} GB)")
PYCHECK

if [ $? -ne 0 ]; then
    echo "GPU check failed, see error above"
    exit 1
fi

# ============================================================
#  RUN FULL PIPELINE
# ============================================================
echo ""
echo "============================================"
echo "Starting full pipeline at $(date)"
echo "============================================"

python scripts/run_full_pipeline.py \
    --config "$CONFIG" \
    --device cuda \
    --num-gpus "$NUM_GPUS" \
    --total-episodes "$TOTAL_EPISODES" \
    --output-dir "$OUTPUT_DIR"

PIPELINE_EXIT=$?

echo ""
echo "Pipeline finished at $(date) with exit code $PIPELINE_EXIT"

# ============================================================
#  GENERATE ANALYSIS FIGURES
# ============================================================
if [ $PIPELINE_EXIT -eq 0 ]; then
    echo ""
    echo "============================================"
    echo "Running matplotlib analysis at $(date)"
    echo "============================================"

    python analysis/analyze_results.py \
        --results-dir "$OUTPUT_DIR" \
        --figures-dir "$OUTPUT_DIR/figures"

    ANALYSIS_EXIT=$?
    echo "Analysis finished at $(date) with exit code $ANALYSIS_EXIT"
else
    echo "Skipping analysis due to pipeline failure"
fi

# ============================================================
#  COPY OUTPUTS
# ============================================================
echo ""
echo "Copying outputs to $SLURM_SUBMIT_DIR ..."

# Copy results (CSVs, JSONs, figures)
if [ -d "$OUTPUT_DIR" ]; then
    mkdir -p "$SLURM_SUBMIT_DIR/$OUTPUT_DIR"
    cp -r $OUTPUT_DIR/* "$SLURM_SUBMIT_DIR/$OUTPUT_DIR/" 2>/dev/null && \
        echo "  $OUTPUT_DIR/ copied"
fi

# Copy checkpoints
if [ -d "checkpoints" ]; then
    mkdir -p "$SLURM_SUBMIT_DIR/checkpoints"
    cp -r checkpoints/* "$SLURM_SUBMIT_DIR/checkpoints/" 2>/dev/null && \
        echo "  checkpoints/ copied"
fi

# Copy tensorboard logs
if [ -d "runs" ]; then
    mkdir -p "$SLURM_SUBMIT_DIR/runs"
    cp -r runs/* "$SLURM_SUBMIT_DIR/runs/" 2>/dev/null && \
        echo "  runs/ copied"
fi

# Copy SLURM log files
cp slurmjob-*.out-* $SLURM_SUBMIT_DIR/ 2>/dev/null || true
cp slurmjob-*.err-* $SLURM_SUBMIT_DIR/ 2>/dev/null || true

# ============================================================
#  CLEANUP
# ============================================================
echo "Cleaning up scratch at $SCRDIR ..."
cd $HOME
rm -rf $SCRDIR

echo "============================================"
echo "Job complete : $(date)"
echo "============================================"

exit $PIPELINE_EXIT
