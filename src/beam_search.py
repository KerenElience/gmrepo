import heapq
import random


class BeamSearch:
    def __init__(self,
                 init_solution: list,
                 estimator,
                 beam_width: int = 5,
                 expand_size: int = 10,
                 max_iter: int = 50,
                 min_size: int = 2,
                 max_size: int = 5):

        self.init_solution = init_solution
        self.estimator = estimator

        self.beam_width = beam_width
        self.expand_size = expand_size
        self.max_iter = max_iter

        self.min_size = min_size
        self.max_size = max_size

    def _score(self, groups):
        total = 0
        for g in groups:
            total += self._group_score(g)
        return total / len(groups)

    def _group_score(self, group):
        penalty = 0
        if len(group) < self.min_size:
            penalty += 0.8
        elif len(group) > self.max_size:
            penalty += 0.8 * (len(group) - self.max_size)

        metrics = self.estimator.get_metrics(group)

        f1 = metrics["f1-score"]["macro avg"]
        m = metrics.iloc[:-3, :]
        mean_r = m["recall"].mean()
        std_r = m["recall"].std()

        return -(f1 + mean_r)/2 + std_r + penalty

    # ====== 邻域操作 ======

    def _swap(self, g1, g2):
        if not g1 or not g2:
            return
        i = random.randrange(len(g1))
        j = random.randrange(len(g2))
        g1[i], g2[j] = g2[j], g1[i]

    def _shift(self, g1, g2):
        if not g1:
            return
        d = random.choice(g1)
        if d not in g2:
            g1.remove(d)
            g2.append(d)

    def _split(self, group):
        if len(group) <= self.min_size:
            return [group]
        random.shuffle(group)
        mid = len(group) // 2
        return [group[:mid], group[mid:]]

    def _merge(self, g1, g2):
        return g1 + g2

    def _generate_neighbors(self, solution):
        neighbors = []

        for _ in range(self.expand_size):
            new_sol = [g.copy() for g in solution]

            op = random.random()

            if op < 0.3 and len(new_sol) >= 2:
                g1, g2 = random.sample(new_sol, 2)
                self._swap(g1, g2)

            elif op < 0.6 and len(new_sol) >= 2:
                g1, g2 = random.sample(new_sol, 2)
                self._shift(g1, g2)

            elif op < 0.8:
                g = random.choice(new_sol)
                new_sol.remove(g)
                new_sol.extend(self._split(g))

            else:
                if len(new_sol) >= 2:
                    g1, g2 = random.sample(new_sol, 2)
                    new_sol.remove(g1)
                    new_sol.remove(g2)
                    new_sol.append(self._merge(g1, g2))

            new_sol = [g for g in new_sol if len(g) > 0]
            neighbors.append(new_sol)

        return neighbors

    # ====== 主流程 ======
    def solve(self):

        # beam: [(score, solution)]
        beam = [(self._score(self.init_solution), self.init_solution)]

        best_solution = self.init_solution
        best_score = beam[0][0]

        for step in range(self.max_iter):

            candidates = []

            for score, sol in beam:
                neighbors = self._generate_neighbors(sol)

                for n in neighbors:
                    s = self._score(n)
                    candidates.append((s, n))

                    if s > best_score:
                        best_score = s
                        best_solution = n

            # 取 top-K
            beam = heapq.nlargest(self.beam_width, candidates, key=lambda x: x[0])

            print(f"Step {step}: best_score={best_score:.4f}")

        return best_solution, best_score
    