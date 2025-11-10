import dataclasses
import math
from numpy import random

Distribution = list[tuple[float, float]]


def sample(d: Distribution) -> float:
    s = random.rand()
    for p, result in d:
        if s < p:
            return result
    return d[-1][1]


@dataclasses.dataclass
class TestDefinition:
    id: int
    p_affected: float
    pass_rates: Distribution


@dataclasses.dataclass
class RepositoryState:
    base_p_pass: dict[int, float] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class Change:
    id: int
    # Map from CL to tests that it affects.
    effects: dict[int, float] = dataclasses.field(default_factory=dict)

    @staticmethod
    def create(change_id: int, test_defs: list[TestDefinition]) -> "Change":
        c = Change(id=change_id)
        for t_def in test_defs:
            if random.rand() < t_def.p_affected:
                c.effects[t_def.id] = sample(t_def.pass_rates)
        return c

    def is_hard_break(self) -> bool:
        return any(e == 0.0 for e in self.effects.values())

    def fix_hard_breaks(self, test_defs: list[TestDefinition]):
        for tid, effect in self.effects.items():
            if effect == 0.0:
                t_def = next(td for td in test_defs if td.id == tid)
                new_effect = 0.0
                for _ in range(10):
                    new_effect = sample(t_def.pass_rates)
                    if new_effect > 0.0:
                        break
                if new_effect == 0.0:
                    new_effect = 1.0
                self.effects[tid] = new_effect


@dataclasses.dataclass
class Minibatch:
    changes: list[Change]  # Cumulative list for execution
    new_changes: list[Change]  # The specific delta this batch adds

    def evaluate(
        self, repo: RepositoryState, all_test_ids: list[int]
    ) -> tuple[bool, bool]:
        passed = True
        hard_failure = False
        for tid in all_test_ids:
            eff_p = repo.base_p_pass.get(tid, 1.0)
            for cl in self.changes:
                eff_p = min(eff_p, cl.effects.get(tid, 1.0))

            if eff_p == 0.0:
                hard_failure = True
            if eff_p < 1.0 and random.rand() >= eff_p:
                passed = False
                if hard_failure:
                    break
        return passed, hard_failure


class PipelinedSubmitQueue:
    def __init__(
        self,
        test_defs: list[TestDefinition],
        pipeline_depth: int,
        max_minibatch_size: int,
    ):
        self.test_defs = test_defs
        self.all_test_ids = [t.id for t in test_defs]
        self.repo = RepositoryState()
        for t in test_defs:
            self.repo.base_p_pass[t.id] = 1.0
        self.pipeline_depth = pipeline_depth
        self.max_minibatch_size = max_minibatch_size
        self.pending_changes: list[Change] = []
        self.change_id_counter = 0

    def add_changes(self, n=1):
        for _ in range(n):
            self.pending_changes.append(
                Change.create(self.change_id_counter, self.test_defs)
            )
            self.change_id_counter += 1

    def step(self) -> int:
        pending_count = len(self.pending_changes)
        if pending_count == 0:
            return 0

        # --- 1. Dynamic Batch Sizing ---
        # Spread pending CLs across the pipeline depth, but cap at max_minibatch_size.
        raw_size = math.ceil(pending_count / self.pipeline_depth)
        base_batch_size = min(max(raw_size, 1), self.max_minibatch_size)

        minibatches = []
        cumulative_cls = []

        for i in range(self.pipeline_depth):
            start = i * base_batch_size
            end = start + base_batch_size
            if start >= pending_count:
                break

            # Slice specifically for this pipeline level
            new_cls_for_level = self.pending_changes[start:end]
            cumulative_cls.extend(new_cls_for_level)

            minibatches.append(
                Minibatch(changes=list(cumulative_cls), new_changes=new_cls_for_level)
            )

        # --- 2. Speculative Execution ---
        results = [mb.evaluate(self.repo, self.all_test_ids) for mb in minibatches]

        # --- 3. Greedy Merging from Head ---
        submitted_count = 0
        pass_streak_broken = False

        for i, (passed, hard_fail) in enumerate(results):
            if pass_streak_broken:
                break

            if passed:
                # Merge all CLs specific to this minibatch
                for cl in minibatches[i].new_changes:
                    for tid, effect in cl.effects.items():
                        self.repo.base_p_pass[tid] = min(
                            self.repo.base_p_pass[tid], effect
                        )
                submitted_count += len(minibatches[i].new_changes)
            else:
                pass_streak_broken = True
                if hard_fail:
                    # The hard failure MUST be in 'new_changes', because if it were
                    # in the previous batch, that batch would have also hard-failed
                    # and we wouldn't have reached here (or it's the first batch).
                    for cl in minibatches[i].new_changes:
                        if cl.is_hard_break():
                            cl.fix_hard_breaks(self.test_defs)
                    # We don't merge this batch, but we fixed the culprit.
                    # Next tick will retry.

        if submitted_count > 0:
            self.pending_changes = self.pending_changes[submitted_count:]

        return submitted_count


if __name__ == "__main__":
    N_ITER = 10000
    N_CHANGES_PER_HOUR = 50
    FIXED_DEPTH = 8  # Fixing depth to isolate effect of batch size

    print(f"Pipeline Depth fixed at: {FIXED_DEPTH}")
    print(f"{'Max Batch':<10} | {'Throughput (CLs/tick)':<22} | {'Average Queue Size'}")
    print("-" * 55)

    for n_tests in [8, 16, 32]:
        print(f"n_tests: {n_tests}")
        for max_mb in [16, 32, 64, 128]:
            total_q = 0
            test_defs = [
                TestDefinition(
                    id=i,
                    p_affected=0.001,
                    # 80% of regressions are to flaky state
                    pass_rates=[(0.50, 1.0), (0.90, 0.99), (1.0, 0.0)],
                )
                for i in range(n_tests)
            ]
            sq = PipelinedSubmitQueue(
                test_defs, pipeline_depth=FIXED_DEPTH, max_minibatch_size=max_mb
            )

            submitted_total = 0
            sq.add_changes(N_CHANGES_PER_HOUR)

            for _ in range(N_ITER):
                submitted_total += sq.step()
                total_q += len(sq.pending_changes)
                if len(sq.pending_changes) < 500:
                    sq.add_changes(N_CHANGES_PER_HOUR)
                elif len(sq.pending_changes) < 1000:
                    # When developers have pending work, they upload less work per hour
                    sq.add_changes(N_CHANGES_PER_HOUR // 2)
            else:
                throughput = submitted_total / N_ITER
                print(f"{max_mb:<10} | {throughput:<22.2f} | {total_q / N_ITER:.0f}")
