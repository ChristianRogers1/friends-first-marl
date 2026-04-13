#!/bin/bash
#SBATCH --account=dbrown
#SBATCH --partition=dbrown-gpu-grn
#SBATCH --qos=dbrown-gpu-grn
#SBATCH --time=20:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH -o slurmjob-%j.out-%N
#SBATCH -e slurmjob-%j.err-%N

# ============================================================
#  Single training + evaluation run for cooperative pretraining
#  MARL experiments.
#
#  Usage:
#    sbatch slurm/train_single.sh                     # defaults
#    sbatch --export=SEED=7,RATIO=0.5 slurm/train_single.sh
#    sbatch --export=SEED=0,RATIO=0.0 slurm/train_single.sh   # baseline
# ============================================================

echo "============================================"
echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $SLURMD_NODENAME"
echo "Start time  : $(date)"
echo "Working dir : $SLURM_SUBMIT_DIR"
echo "============================================"

# ---------- tunables (override via --export) ----------
SEED=${SEED:-42}
RATIO=${RATIO:-0.25}
CONFIG=${CONFIG:-configs/default.yaml}
EVAL_EPISODES=${EVAL_EPISODES:-500}

echo "SEED   = $SEED"
echo "RATIO  = $RATIO"
echo "CONFIG = $CONFIG"

# ---------- scratch directory ----------
SCRDIR=/scratch/general/vast/$USER/$SLURM_JOB_ID
mkdir -p $SCRDIR

# Copy project to scratch
cp -r $SLURM_SUBMIT_DIR/src        $SCRDIR/
cp -r $SLURM_SUBMIT_DIR/configs    $SCRDIR/
cp -r $SLURM_SUBMIT_DIR/scripts    $SCRDIR/
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

# ---------- training ----------
echo ""
echo "Starting training at $(date)"
echo "  seed=$SEED  ratio=$RATIO  config=$CONFIG"

python scripts/train.py \
    --config "$CONFIG" \
    --seed "$SEED" \
    --pretraining-ratio "$RATIO" \
    --device cuda

TRAIN_EXIT=$?
echo "Training finished at $(date) with exit code $TRAIN_EXIT"

# ---------- evaluation ----------
if [ $TRAIN_EXIT -eq 0 ]; then
    CKPT_DIR="checkpoints/seed_${SEED}_ratio${RATIO}"

    echo ""
    echo "Starting evaluation at $(date)"

    python scripts/evaluate.py \
        --config "$CONFIG" \
        --checkpoint-dir "$CKPT_DIR" \
        --num-episodes "$EVAL_EPISODES"

    EVAL_EXIT=$?
    echo "Evaluation finished at $(date) with exit code $EVAL_EXIT"
else
    echo "Skipping evaluation due to training failure"
fi

# ---------- copy outputs ----------
echo ""
echo "Copying outputs to $SLURM_SUBMIT_DIR ..."

# Results, checkpoints, tensorboard logs
for dir in results checkpoints runs; do
    if [ -d "$dir" ]; then
        mkdir -p "$SLURM_SUBMIT_DIR/$dir"
        cp -r $dir/* "$SLURM_SUBMIT_DIR/$dir/" 2>/dev/null && \
            echo "  $dir/ copied"
    fi
done

# Copy SLURM log files
cp slurmjob-*.out-* $SLURM_SUBMIT_DIR/ 2>/dev/null || true
cp slurmjob-*.err-* $SLURM_SUBMIT_DIR/ 2>/dev/null || true

# ---------- cleanup ----------
echo "Cleaning up scratch at $SCRDIR ..."
cd $HOME
rm -rf $SCRDIR

echo "============================================"
echo "Job complete : $(date)"
echo "============================================"

exit $TRAIN_EXIT