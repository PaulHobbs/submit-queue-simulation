#!/usr/bin/env python3
"""
Generate realistic build history data for CSV mode testing.
"""

import csv
import random
import math
from datetime import datetime, timedelta
from typing import List, Tuple

# Configuration
NUM_CHANGES = 500  # Total number of CLs
NUM_TARGETS = 80  # Number of build targets
CULPRIT_RATE = 0.03  # Probability a CL is a culprit (3%)
FLAKINESS_RATE = 0.05  # Probability of flaky tests for good CLs
FLAKE_APPEAR_RATE = 0.15  # Probability that a test shows flakiness
TIME_RANGE_DAYS = 14  # Simulate 2 weeks of builds
CHANGES_PER_HOUR = 5  # Average number of CLs per hour

# Target names (simulating real build targets)
TARGET_TEMPLATES = [
    "//core/authentication:unit_tests",
    "//core/authentication:integration_tests",
    "//api/v1:service_tests",
    "//api/v1:client_tests",
    "//database:migration_tests",
    "//database:query_tests",
    "//frontend/components:unit_tests",
    "//frontend/pages:e2e_tests",
    "//backend/handlers:handler_tests",
    "//backend/middleware:middleware_tests",
    "//cache/redis:redis_tests",
    "//cache/memcached:memcached_tests",
    "//monitoring/metrics:metrics_tests",
    "//logging:logger_tests",
    "//security/crypto:crypto_tests",
    "//security/auth:auth_tests",
    "//networking/http:http_tests",
    "//networking/grpc:grpc_tests",
    "//storage/s3:s3_tests",
    "//storage/gcs:gcs_tests",
    "//search/elasticsearch:search_tests",
    "//search/indexing:indexing_tests",
    "//config/loader:config_tests",
    "//config/validation:validation_tests",
    "//utils/string:string_tests",
    "//utils/math:math_tests",
    "//utils/time:time_tests",
    "//utils/json:json_tests",
    "//performance/benchmarks:benchmark_tests",
    "//performance/profiling:profiling_tests",
    "//deploy/kubernetes:k8s_tests",
    "//deploy/docker:docker_tests",
    "//deploy/terraform:terraform_tests",
    "//ci/scripts:integration_tests",
    "//proto/messages:proto_tests",
    "//proto/services:service_tests",
    "//thirdparty/vendors:vendor_tests",
    "//thirdparty/licensing:license_tests",
    "//docs/api:doc_tests",
    "//docs/guides:guide_tests",
]

# Expand targets to reach NUM_TARGETS
def generate_targets() -> List[str]:
    """Generate a list of build targets."""
    targets = []
    # Use templates multiple times with variations
    for i in range(NUM_TARGETS):
        template = TARGET_TEMPLATES[i % len(TARGET_TEMPLATES)]
        # Add variant suffixes for duplicate templates
        if i >= len(TARGET_TEMPLATES):
            variant = i // len(TARGET_TEMPLATES)
            target = f"{template}_v{variant}"
        else:
            target = template
        targets.append(target)
    return targets

class BuildHistoryGenerator:
    def __init__(self, num_changes: int, num_targets: int, seed: int = 42):
        self.num_changes = num_changes
        self.num_targets = num_targets
        self.targets = generate_targets()
        self.random = random.Random(seed)
        self.now = datetime.now()

        # Track which tests are flaky (consistent across runs)
        self.flaky_tests = set()
        for i in range(len(self.targets)):
            if self.random.random() < FLAKINESS_RATE:
                self.flaky_tests.add(self.targets[i])

        # Track which CLs are culprits
        self.culprit_cids = set()
        for i in range(int(num_changes * CULPRIT_RATE)):
            self.culprit_cids.add(self.random.randint(0, num_changes - 1))

    def generate_records(self) -> List[Tuple]:
        """Generate CSV records for build history."""
        records = []

        # Generate CLs spread out over time
        time_range_seconds = TIME_RANGE_DAYS * 24 * 3600

        for cid in range(self.num_changes):
            # Randomize time (roughly following CHANGES_PER_HOUR distribution)
            # Use Poisson-like distribution
            random_offset = self.random.expovariate(1.0 / (3600 / CHANGES_PER_HOUR))
            if cid == 0:
                relative_time = 0
            else:
                relative_time = (cid - 1) * time_range_seconds / self.num_changes
                relative_time += random_offset

            # Ensure it stays within bounds
            relative_time = min(relative_time, time_range_seconds - 1)
            relative_time = max(relative_time, 0)

            creation_time_millis = int((self.now - timedelta(seconds=time_range_seconds - relative_time)).timestamp() * 1000)
            hour = int(creation_time_millis // 3600000)
            is_bad = cid in self.culprit_cids

            # Determine which targets to build (not all targets built for every CL)
            num_targets_for_cl = self.random.randint(int(0.3 * len(self.targets)), len(self.targets))
            targets_for_cl = self.random.sample(self.targets, num_targets_for_cl)

            for target in targets_for_cl:
                # Determine success/failure
                if is_bad:
                    # Culprits: deterministically break certain tests
                    # 30-70% of their tests fail
                    if self.random.random() < self.random.uniform(0.3, 0.7):
                        success = False
                        flake = False
                    else:
                        success = True
                        flake = False
                else:
                    # Innocent CLs: mostly pass
                    if target in self.flaky_tests:
                        # Flaky tests
                        success = self.random.random() > 0.1  # 90% pass rate for flaky tests
                        # Flake indicator: we saw both pass and fail for this CL
                        # This is only meaningful if success=True
                        if success:
                            flake = self.random.random() < FLAKE_APPEAR_RATE
                        else:
                            flake = False
                    else:
                        # Regular tests: 99% pass rate
                        success = self.random.random() > 0.01
                        flake = False

                record = (
                    cid,
                    target,
                    creation_time_millis,
                    "true" if success else "false",
                    "true" if flake else "false",
                    creation_time_millis,  # timestamp same as creation_time
                    hour,
                    "true" if is_bad else "false",
                )
                records.append(record)

        return records

def main():
    """Generate and write build history CSV."""
    print(f"Generating {NUM_CHANGES} changes with {NUM_TARGETS} targets...")

    generator = BuildHistoryGenerator(NUM_CHANGES, NUM_TARGETS)
    records = generator.generate_records()

    print(f"Generated {len(records)} records")
    print(f"Culprits: {len(generator.culprit_cids)}")
    print(f"Flaky tests: {len(generator.flaky_tests)}")

    # Write CSV file
    output_file = "build_history.csv"
    print(f"\nWriting to {output_file}...")

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'change_number',
            'target',
            'creation_time_millis',
            'success',
            'flake',
            'timestamp',
            'hour',
            'is_bad'
        ])
        writer.writerows(records)

    print(f"✓ Written {len(records)} records to {output_file}")

    # Print statistics
    print("\nStatistics:")
    culprit_records = sum(1 for r in records if r[7] == "true")
    innocent_records = len(records) - culprit_records
    culprit_success = sum(1 for r in records if r[7] == "true" and r[3] == "true")
    innocent_success = sum(1 for r in records if r[7] == "false" and r[3] == "true")

    print(f"  Total records: {len(records)}")
    print(f"  Culprit records: {culprit_records} ({culprit_records/len(records)*100:.1f}%)")
    print(f"  Innocent records: {innocent_records} ({innocent_records/len(records)*100:.1f}%)")
    print(f"  Culprit success rate: {culprit_success/max(1, culprit_records)*100:.1f}%")
    print(f"  Innocent success rate: {innocent_success/max(1, innocent_records)*100:.1f}%")

    # Show time range
    times = [int(r[2]) for r in records]
    min_time = min(times)
    max_time = max(times)
    min_dt = datetime.fromtimestamp(min_time / 1000)
    max_dt = datetime.fromtimestamp(max_time / 1000)
    days = (max_dt - min_dt).days
    print(f"  Time range: {days} days ({min_dt.strftime('%Y-%m-%d %H:%M')} to {max_dt.strftime('%Y-%m-%d %H:%M')})")

if __name__ == '__main__':
    main()
