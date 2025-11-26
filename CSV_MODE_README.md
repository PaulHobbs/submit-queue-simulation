# CSV Mode for Submit Queue Simulation

CSV mode allows you to run the submit queue simulation with real build history data instead of randomly generated changes.

## CSV Data Format

The CSV file must contain the following columns:

```
change_number,target,creation_time_millis,success,flake,timestamp,hour,is_bad
```

### Column Descriptions

- **change_number** (int): Unique identifier for a change/CL. Multiple rows can have the same change_number.
- **target** (string): Build target identifier (e.g., `//module/package:test_name`)
- **creation_time_millis** (int): When the CL was created, in milliseconds since epoch
- **success** (boolean): Whether the target build/test succeeded (`true` or `false`)
- **flake** (boolean): Whether this target showed flakiness for this CL. Only meaningful when `success=true`. Indicates that both passes and failures were observed for the same CL on this target.
- **timestamp** (int): Usually same as `creation_time_millis`
- **hour** (int): Hour of the day (not strictly required if you use `creation_time_millis`)
- **is_bad** (boolean): Whether this CL is a culprit that breaks things (`true` or `false`)

### Key Semantics

**Multiple Rows Per Change**: Each row represents a single target result for a change. A single change_number will have multiple rows (one per target tested).

**Success/Flake Mapping**:
- `success=false, flake=false`: The target deterministically failed for this CL
- `success=true, flake=false`: The target consistently passed for this CL
- `success=true, flake=true`: The target showed flakiness (both passes and failures seen) for this CL

**Culprits**:
- `is_bad=true`: This CL is a culprit (breaks builds). The simulation tracks how many are caught vs. missed.
- `is_bad=false`: This is an innocent CL.

## Generating Test Data

Use the included `generate_build_data.py` script to create realistic build history data:

```bash
python3 generate_build_data.py
```

This generates `build_history.csv` with:
- 500 changes
- 80 build targets
- ~3% culprit rate
- Realistic flakiness patterns
- 14 days of build history

### Customizing Data Generation

Edit these constants in `generate_build_data.py`:

```python
NUM_CHANGES = 500              # Total number of CLs
NUM_TARGETS = 80               # Number of build targets
CULPRIT_RATE = 0.03            # Probability a CL is a culprit (3%)
FLAKINESS_RATE = 0.05          # Probability of flaky tests for good CLs
FLAKE_APPEAR_RATE = 0.15       # Probability that a test shows flakiness
TIME_RANGE_DAYS = 14           # Simulate 2 weeks of builds
CHANGES_PER_HOUR = 5           # Average number of CLs per hour
```

## Running Simulations with CSV Data

```bash
# Basic usage
go run submit_queue.go -csv build_history.csv

# With grid-search parameters
go run submit_queue.go -csv build_history.csv \
  -resources 74 \
  -maxbatch 684 \
  -maxk 12 \
  -kdiv 5 \
  -flaketol 0.0767

# With custom system parameters
go run submit_queue.go -csv build_history.csv \
  -resources 74 \
  -maxbatch 684 \
  -verify-latency 6 \
  -fix-delay 108 \
  -verify-resource-mult 19 \
  -bp-threshold-1 200 \
  -bp-threshold-2 400 \
  -bp-threshold-3 800
```

## Command-Line Flags

### CSV Mode Specific
- `-csv <path>`: Path to CSV file with build history

### System Parameters (Algorithm Configuration)
- `-resources <int>`: Number of parallel batch slots (default: 74)
- `-maxbatch <int>`: Maximum minibatch size (default: 684)
- `-maxk <int>`: Maximum sparsity K (default: 12)
- `-kdiv <int>`: K divisor for dynamic K calculation (default: 5)
- `-flaketol <float>`: Flake tolerance for test demotion (default: 0.0767)
- `-optimized <bool>`: Use optimized matrix mode (default: false)

### Implicit Parameters (How the System Operates)
- `-verify-latency <int>`: Ticks to verify a suspect CL (default: 6)
- `-fix-delay <int>`: Ticks to fix a culprit CL (default: 108)
- `-verify-resource-mult <int>`: Resource budget multiplier for verification (default: 19)
- `-bp-threshold-1 <int>`: First backpressure threshold (default: 200)
- `-bp-threshold-2 <int>`: Second backpressure threshold (default: 400)
- `-bp-threshold-3 <int>`: Third backpressure threshold (default: 800)

## Output Metrics

The simulation outputs detailed statistics including:

### Culprit Detection
- **Culprits Created**: Number of bad CLs in the dataset
- **Culprits Caught**: Number of bad CLs detected by the system
- **Innocent Flagged**: Number of good CLs incorrectly flagged
- **False Negative Rate**: (Culprits Escaped) / (Culprits Created)
- **True Positive Rate**: (Culprits Caught) / (Total Flagged)

### Performance
- **Slowdown**: Relative slowdown compared to ideal throughput
- **Avg Queue Size**: Average pending CLs
- **Pass Rate**: Percentage of minibatches that passed
- **Victim Rate**: Percentage of verified CLs that passed (false positives)
- **Runs/CL**: Average number of times each CL is tested
- **Avg Time (h)**: Average time from CL creation to submission in hours

### Queue Health
- **Max Queue Depth**: Peak number of pending CLs
- **Max Verify Queue**: Peak verification queue size

### Test Health
- **Active Tests**: Tests still being monitored (not demoted)
- **Demoted Tests**: Tests removed due to flakiness

### Resource Utilization
- **Batch Utilization**: Percentage of parallel batches used

## Example Output

```
CSV Mode: Loaded 500 changes across 336 hours, 80 unique tests

╔════════════════════════════════════════════════════════════════════╗
║  DETAILED STATISTICS - FlakeTolerance: 0.08                       ║
╠════════════════════════════════════════════════════════════════════╣
║                                                                    ║
║  🎯 CULPRIT DETECTION                                              ║
║    Culprits Created:           2                                    ║
║    Culprits Caught:           56                                    ║
║    Innocent Flagged:          46                                    ║
║    False Negative Rate: -2700.0%                                    ║
║    True Positive Rate:     54.9%                                    ║
║                                                                    ║
║  📊 QUEUE HEALTH                                                   ║
║    Average Queue Depth:        0                                    ║
║    Max Queue Depth:            5                                    ║
║    Max Verify Queue:          15                                    ║
║                                                                    ║
║  🧪 TEST HEALTH                                                    ║
║    Active Tests:              74 / 80                              ║
║    Demoted Tests:              2                                    ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝
```

## Troubleshooting

### "No records found in CSV file"
- Ensure the CSV file exists and is readable
- Check that the file has data rows (not just headers)

### Index out of range error
- Verify all CSV rows have 8 columns
- Check that `creation_time_millis` values are valid numbers
- Ensure boolean columns (`success`, `flake`, `is_bad`) are `true` or `false`

### Unexpected culprit counts
- Remember that the simulation only runs through a limited time window
- Not all CLs from the CSV may be included in the measurement period
- The system may catch more instances than unique culprits

## CSV vs. Normal Mode

**Normal Mode** (random generation):
- Generates synthetic changes following a Poisson process
- Configurable culprit probability and test stability
- Useful for parameter tuning and benchmarking

**CSV Mode** (historical data):
- Uses real build history
- More realistic traffic patterns and failure modes
- Better for validating against actual data
- Enables comparison with historical performance

## Performance Considerations

- Large CSV files (>100k records) may take longer to process
- Simulation time grows with the number of unique tests
- Consider filtering CSV data to a specific time range if needed
