#!/bin/bash
#SBATCH --account=dbrown
#SBATCH --partition=dbrown-gpu-grn
#SBATCH --qos=dbrown-gpu-grn
#SBATCH --time=23:59:59
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -o slurmjob-%j.out-%N
#SBATCH -e slurmjob-%j.err-%N


# ============================================================
#  Full experimental pipeline SLURM job
#
#  Runs the entire screen-and-confirm procedure:
#    1. Screening  -- 4 ratios x 5 seeds  = 20 training runs
#    2. Confirmation -- best ratio x 12 seeds + baseline x 12 seeds = 24 runs
#    3. Evaluation -- H1/H2/H3 metrics on all confirmation agents
#    4. Analysis   -- Generate figures with matplotlib
#
#  Usage:
#    sbatch slurm/run_pipeline.sh
#    sbatch --export=TOTAL_EPISODES=50000 slurm/run_pipeline.sh
# ============================================================

echo "============================================"
echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $SLURMD_NODENAME"
echo "Start time  : $(date)"
echo "Working dir : $SLURM_SUBMIT_DIR"
echo "============================================"

# ---------- tunables (override via --export) ----------
TOTAL_EPISODES=${TOTAL_EPISODES:-50000}
CONFIG=${CONFIG:-configs/default.yaml}
OUTPUT_DIR=${OUTPUT_DIR:-results}

echo "TOTAL_EPISODES = $TOTAL_EPISODES"
echo "CONFIG         = $CONFIG"
echo "OUTPUT_DIR     = $OUTPUT_DIR"

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
print(f"GPU      : {torch.cuda.get_device_name(0)}")
print(f"VRAM     : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
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