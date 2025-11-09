import dataclasses
from numpy import random
from tests import *


@dataclasses.dataclass
class SubmitQueue:
    tests: list[Test]
    build_id: int = 0

    def step(self, n=1) -> Run:
        r = Run(build_id=self.build_id, n=n, success=True, tests=self.tests)
        self.build_id += 1
        return r.evaluate()


if __name__ == "__main__":

    PRIMING_ITERATIONS = 1000
    N_ITER = 100000
    N_TESTS = 20
    pending = 0

    for batch_size in [
        1,
        2,
        4,
        8,
        16,
        32,
        64,
    ]:
        tests = [
            Test(
                p_pass=1,
                p_affected=0.001,
                # Per change:
                #   0.04% chance of transitioning to 0.5% flaky
                #   0.01% chance of being broken by a CL (would never be submitted by design)
                pass_rates=[(0.50, 1), (0.90, 0.995), (1, 0)],
            )
            for _ in range(N_TESTS)
        ]

        sq = SubmitQueue(tests=tests)
        submitted = 0
        for _ in range(PRIMING_ITERATIONS):  # Prime for a while without backing up
            sq.step(n=batch_size)
        print(f"batch_size={batch_size}: ")
        for _ in range(N_ITER):
            pending += batch_size
            if pending > 1000:
                print("...submit queue got stuck.")
                break
            if sq.step(n=pending).success:
                submitted += 1
                pending = 0

        print(f"  failed { 100 * (N_ITER-submitted )/ N_ITER}%")
        print(f"  throughput {(batch_size * submitted) / N_ITER:2f}")
