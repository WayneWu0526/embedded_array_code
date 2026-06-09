# Tools

Small shell wrappers for repeatable local workflows. These scripts should stay
thin: package-specific ROS nodes belong under the package's `scripts/`
directory.

## Manual MagGrad Recording

Terminal 1 launches the manual recorder from this workspace:

```bash
./tools/run_stm32_manual_migels.sh
```

Terminal 2 controls start/stop with Enter:

```bash
./tools/manual_record_enter.sh
```

Both scripts support `--help`.

## Scripts

| Script | Purpose |
| --- | --- |
| `run_stm32_manual_migels.sh` | Sources ROS, `zlab_robots`, and this workspace; verifies `sensor_data_collection` resolves to this worktree; launches `stm32_manual.launch` |
| `manual_record_enter.sh` | Publishes `std_msgs/Bool` start/stop triggers to `/maggrad_manual_record/record_trigger` |

## Environment

| Variable | Default | Used by |
| --- | --- | --- |
| `ROS_DISTRO` | `noetic` | Both scripts |
| `ZLAB_ROBOTS_WS` | `$HOME/zlab_robots` | Both scripts |
| `TOPIC` | `/maggrad_manual_record/record_trigger` | `manual_record_enter.sh` |
