import dataclasses
from numpy import random


Distribution = list[tuple[float, float]]


@dataclasses.dataclass
class Test:
    p_pass: float
    p_affected: float
    pass_rates: Distribution
    new_p: float = 1

    def new_pass_rate(self, n=1):
        n_affected = random.binomial(n, self.p_affected)
        if n_affected:
            self.new_p = min(sample(self.pass_rates) for _ in range(n_affected))

    def transition(self):
        self.p_pass = self.new_p


@dataclasses.dataclass
class Run:
    build_id: int
    success: bool
    tests: list[Test]
    n: int

    def evaluate(self):
        for t in self.tests:
            t.new_pass_rate(n=self.n)
        self.success = all(random.rand() < t.new_p for t in self.tests if t.new_p < 1)
        if self.success:
            for t in self.tests:
                t.transition()
        return self


def sample(d: Distribution) -> float:
    s = random.rand()
    for p, result in d:
        if s < p:
            return result
    return d[-1][1]
