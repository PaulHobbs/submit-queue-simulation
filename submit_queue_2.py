import dataclasses
import copy
from numpy import random

Distribution = list[tuple[float, float]]

def sample(d: Distribution) -> float:
    s = random.rand()
    for p, result in d:
        if s < p:
            return result
    # Fallback for the last element if it doesn't strictly cover up to 1.0 
    # though typical definitions usually do.
    return d[-1][1]

@dataclasses.dataclass
class TestDefinition:
    """Static definition of a test's behavior under change."""
    id: int
    p_affected: float
    pass_rates: Distribution

@dataclasses.dataclass
class RepositoryState:
    """Current permanent state of the repository's tests."""
    # base_p_pass[i] is the pass rate of Test i in the repo before any pending CLs.
    base_p_pass: dict[int, float] = dataclasses.field(default_factory=dict)

@dataclasses.dataclass
class Change:
    """A concrete CL that might affect tests."""
    id: int
    # effects[test_id] = max_pass_rate imposed by this CL. 
    # 1.0 = neutral, 0.0 = broken, 0.995 = flaky.
    effects: dict[int, float] = dataclasses.field(default_factory=dict)

    @staticmethod
    def create(change_id: int, test_defs: list[TestDefinition]) -> 'Change':
        c = Change(id=change_id)
        for t_def in test_defs:
             # Roll to see if this CL affects this specific test
             if random.rand() < t_def.p_affected:
                 c.effects[t_def.id] = sample(t_def.pass_rates)
        return c

    def is_hard_break(self) -> bool:
        return any(e == 0.0 for e in self.effects.values())

    def fix_hard_breaks(self, test_defs: list[TestDefinition]):
        """Converts 0.0 (hard break) effects into non-zero (maybe flaky) effects."""
        for tid, effect in self.effects.items():
            if effect == 0.0:
                # Find the definition for this test to resample
                t_def = next(td for td in test_defs if td.id == tid)
                new_effect = 0.0
                # Resample until we get a non-breaking result (simulating a fix)
                # We limit iterations just in case a test is 100% broken by design.
                for _ in range(10): 
                    new_effect = sample(t_def.pass_rates)
                    if new_effect > 0.0:
                        break
                
                if new_effect == 0.0:
                     # Fallback if resampling keeps hitting 0: force it to neutral 
                     # to ensure we don't get stuck forever in simulation.
                     new_effect = 1.0
                self.effects[tid] = new_effect

@dataclasses.dataclass
class Minibatch:
    """A speculative run of a sequence of CLs."""
    changes: list[Change]
    
    def evaluate(self, repo: RepositoryState, all_test_ids: list[int]) -> tuple[bool, bool]:
        """
        Returns: (passed: bool, hard_failure: bool)
        """
        # Calculate effective pass rate for every test in this minibatch
        passed = True
        hard_failure = False

        for tid in all_test_ids:
            # Start with repo baseline health
            eff_p = repo.base_p_pass.get(tid, 1.0)
            
            # Combine with effects of all CLs in this batch. 
            # We use min() to model that one broken CL breaks the whole test.
            for cl in self.changes:
                eff_p = min(eff_p, cl.effects.get(tid, 1.0))
            
            if eff_p == 0.0:
                hard_failure = True

            # Perform the actual test run (roll the dice)
            if eff_p < 1.0 and random.rand() >= eff_p:
                passed = False
                # In a real scenario, we might stop early here, but 
                # we need to know if it was a hard_failure for culprit finding.
                if hard_failure: 
                     break

        return passed, hard_failure

class PipelinedSubmitQueue:
    def __init__(self, test_defs: list[TestDefinition], pipeline_depth: int = 3):
        self.test_defs = test_defs
        self.all_test_ids = [t.id for t in test_defs]
        self.repo = RepositoryState()
        # Initialize all tests as 100% passing baseline
        for t in test_defs:
            self.repo.base_p_pass[t.id] = 1.0
            
        self.pipeline_depth = pipeline_depth
        self.pending_changes: list[Change] = []
        self.change_id_counter = 0

    def add_changes(self, n=1):
        for _ in range(n):
            self.pending_changes.append(
                Change.create(self.change_id_counter, self.test_defs)
            )
            self.change_id_counter += 1

    def step(self) -> int:
        """
        Runs one pipeline tick.
        Returns number of merged CLs.
        """
        # 1. Fill pipeline (creating speculative minibatches)
        # The pipeline is a list of Minibatches. 
        # If we have CLs [A, B, C], batches are [A], [A,B], [A,B,C].
        
        current_pipeline_cls = self.pending_changes[:self.pipeline_depth]
        if not current_pipeline_cls:
            return 0

        minibatches = []
        cumulative_cls = []
        for cl in current_pipeline_cls:
            cumulative_cls.append(cl)
            # Important: Must copy the list so batches don't share mutable references
            minibatches.append(Minibatch(changes=list(cumulative_cls)))

        # 2. Run all minibatches in parallel (conceptually)
        results = [mb.evaluate(self.repo, self.all_test_ids) for mb in minibatches]
        # results is list of (passed, hard_failure)

        # 3. Process results from the head (oldest) down
        submitted_count = 0
        
        # We can greedy merge. If batch 0 passes, we merge CL 0.
        # If batch 1 ALSO passed, we know CL 1 is also good to go.
        
        for i, (passed, hard_fail) in enumerate(results):
            cls_in_batch = minibatches[i].changes
            
            # If the previous batch failed, we can't merge this one 
            # (it was built on a broken foundation).
            # In this greedy loop, if we didn't merge the previous CLs, we must stop.
            if i > submitted_count:
                break

            if passed:
                # Success! Merge the SPECIFIC new CL that this batch added.
                cl_to_merge = cls_in_batch[-1]
                
                # Persist any flakiness this CL introduced into the repo baseline
                for tid, effect in cl_to_merge.effects.items():
                     self.repo.base_p_pass[tid] = min(self.repo.base_p_pass[tid], effect)
                     
                submitted_count += 1
            else:
                # This batch failed.
                if hard_fail:
                    # Culprit Finding Logic:
                    # We assume we can perfectly identify the hard breaker in this batch.
                    # For simplicity in simulation, we just check the CLs in this batch.
                    for cl in cls_in_batch:
                        if cl.is_hard_break():
                            cl.fix_hard_breaks(self.test_defs)
                            # In a real system, this CL might be kicked out.
                            # Here we "convert to non-breakage" and keep it in pending
                            # to be retried next tick.
                    # Once we found a hard failure, all subsequent speculative batches 
                    # are invalid. Stop processing.
                    break
                else:
                    # Flaky failure. 
                    # We don't merge this. We stop greedy merging.
                    # Next tick will retry this batch (and likely pass if it's just flaky).
                    break
                    
        # Remove merged CLs from pending
        if submitted_count > 0:
            self.pending_changes = self.pending_changes[submitted_count:]
            
        return submitted_count

if __name__ == "__main__":
    N_ITER = 50000 
    N_TESTS = 20
    
    # Define tests once.
    test_defs = [
        TestDefinition(
            id=i,
            p_affected=0.001,
            pass_rates=[(0.50, 1.0), (0.90, 0.995), (1.0, 0.0)],
        )
        for i in range(N_TESTS)
    ]

    for pipeline_depth in [1, 2, 4, 8, 16, 32]:
        sq = PipelinedSubmitQueue(test_defs, pipeline_depth=pipeline_depth)
        
        pending_target = pipeline_depth * 2 # Keep queue reasonably full to utilize pipeline
        submitted_total = 0
        
        # Pre-fill some changes
        sq.add_changes(pending_target)

        for _ in range(N_ITER):
            # Top up the queue if it's getting low, to ensure pipeline stays busy
            if len(sq.pending_changes) < pending_target:
                 sq.add_changes(pipeline_depth)
            
            # Check for "stuck" queue (head CL failing repeatedly? or just too many broken)
            if len(sq.pending_changes) > 1000:
                 print("...submit queue got stuck.")
                 break

            merged = sq.step()
            submitted_total += merged

        # Throughput = CLs merged per tick
        throughput = submitted_total / N_ITER
        print(f"pipeline_depth={pipeline_depth:2}: throughput {throughput:.2f} CLs/tick")
