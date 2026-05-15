$RUN_DIR = "outputs/experiments/train2000_test500_fixed_2p56_1p28"
$CKPT = "checkpoints/r3d18_train2000_e3/best.pth"
$MAX = 500

New-Item -ItemType Directory -Force -Path "$RUN_DIR/manifest" | Out-Null
New-Item -ItemType Directory -Force -Path "$RUN_DIR/predictions" | Out-Null
New-Item -ItemType Directory -Force -Path "$RUN_DIR/metrics" | Out-Null
New-Item -ItemType Directory -Force -Path "$RUN_DIR/figures" | Out-Null

python dataset.py `
  --annotations Charades/Charades_v1_test.csv `
  --video-dir Charades_v1_480 `
  --mode fixed `
  --clip-seconds 2.56 `
  --stride-seconds 1.28 `
  --max-videos $MAX `
  --require-video `
  --out-manifest "$RUN_DIR/manifest/manifest.csv"

python run_inference.py `
  --manifest "$RUN_DIR/manifest/manifest.csv" `
  --checkpoint $CKPT `
  --arch r3d_18 `
  --num-frames 8 `
  --resize 112 `
  --batch-size 16 `
  --print-topk 5 `
  --print-every 100 `
  --out "$RUN_DIR/predictions/clip_results.csv"

python aggregate_metrics.py `
  --clip-results "$RUN_DIR/predictions/clip_results.csv" `
  --out-dir "$RUN_DIR/metrics"

python plot_results.py `
  --clip-results "$RUN_DIR/predictions/clip_results.csv" `
  --summary "$RUN_DIR/metrics/action_summary.csv" `
  --run-summary "$RUN_DIR/metrics/run_summary.json" `
  --out-dir "$RUN_DIR/figures"
