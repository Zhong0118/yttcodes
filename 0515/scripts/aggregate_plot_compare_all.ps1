$ErrorActionPreference = "Stop"

$RUNS = @(
  @{ Name = "action_delay0p5"; Dir = "outputs/experiments/trainall_testall_action_delay0p5" },
  @{ Name = "fixed_1p28_0p64"; Dir = "outputs/experiments/trainall_testall_fixed_1p28_0p64" },
  @{ Name = "fixed_2p56_1p28"; Dir = "outputs/experiments/trainall_testall_fixed_2p56_1p28" },
  @{ Name = "fixed_5p12_2p56"; Dir = "outputs/experiments/trainall_testall_fixed_5p12_2p56" },
  @{ Name = "fixed_7p68_3p84"; Dir = "outputs/experiments/trainall_testall_fixed_7p68_3p84" }
)

foreach ($run in $RUNS) {
  $dir = $run.Dir
  $clipResults = "$dir/predictions/clip_results.csv"
  $metricsDir = "$dir/metrics"
  $figuresDir = "$dir/figures"

  if (!(Test-Path $clipResults)) {
    throw "Missing clip results: $clipResults"
  }

  New-Item -ItemType Directory -Force -Path $metricsDir | Out-Null
  New-Item -ItemType Directory -Force -Path $figuresDir | Out-Null

  Write-Host ""
  Write-Host "Aggregating $($run.Name) ..."
  python aggregate_metrics.py `
    --clip-results $clipResults `
    --out-dir $metricsDir

  Write-Host "Plotting $($run.Name) ..."
  python plot_results.py `
    --clip-results $clipResults `
    --summary "$metricsDir/action_summary.csv" `
    --run-summary "$metricsDir/run_summary.json" `
    --out-dir $figuresDir
}

Write-Host ""
Write-Host "Comparing all experiments ..."
python compare_experiments.py `
  --run action_delay0p5 outputs/experiments/trainall_testall_action_delay0p5 `
  --run fixed_1p28_0p64 outputs/experiments/trainall_testall_fixed_1p28_0p64 `
  --run fixed_2p56_1p28 outputs/experiments/trainall_testall_fixed_2p56_1p28 `
  --run fixed_5p12_2p56 outputs/experiments/trainall_testall_fixed_5p12_2p56 `
  --run fixed_7p68_3p84 outputs/experiments/trainall_testall_fixed_7p68_3p84 `
  --out-dir outputs/experiments/comparison_trainall_testall_all5

Write-Host ""
Write-Host "Done. Main comparison outputs:"
Write-Host "  outputs/experiments/comparison_trainall_testall_all5/comparison_summary.csv"
Write-Host "  outputs/experiments/comparison_trainall_testall_all5/performance_comparison.png"
Write-Host "  outputs/experiments/comparison_trainall_testall_all5/latency_comparison.png"
Write-Host "  outputs/experiments/comparison_trainall_testall_all5/delay_comparison.png"
