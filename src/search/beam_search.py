import heapq
import random
from copy import deepcopy
from joblib import Parallel, delayed
from gmrepo.src.evaluator import DIestimator
from gmrepo.src.utils import encode_solution, random_partition

class BeamSearch:
    def __init__(self,
                 disease: list,
                 estimator: DIestimator,
                 beam_width: int = 5,
                 expand_size: int = 10,
                 max_iter: int = 50,
                 min_size: int = 2,
                 max_size: int = 5):

        self.disease = disease
        self.estimator = estimator

        self.beam_width = beam_width
        self.expand_size = expand_size
        self.max_iter = max_iter

        self.min_size = min_size
        self.max_size = max_size

        self.cache = {}

        self.diversity_threshold = 0.3
        self.min_diversity = 0.05
        self.max_diversity = 0.6

        self.no_improve_steps = 0
        self.prev_best_score = -1e9

    def _score(self, groups):
        key = encode_solution(groups)
        if key in self.cache:
            return self.cache[key]
        
        total = 0
        for g in groups:
            total += self._group_score(g)
        score = total / len(groups) if groups else 0
        self.cache[key] = score
        return score

    def _group_score(self, group):
        score = self.estimator.get_metrics(group)
        return score

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
            new_sol = deepcopy(solution)

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

    def init_population(self):
        groups = [random_partition(self.disease) for _ in range(self.beam_width * 4)]
        scored = self.parallel_score(groups)
        return heapq.nlargest(self.beam_width, scored, key = lambda x: x[0])

    def local_search(self, groups):
        best = groups
        best_score = self._score(groups)
        for _ in range(5):
            neighbors = self._generate_neighbors(groups)
            scored = self.parallel_score(neighbors)
            s, candidate = max(scored, key=lambda x: x[0])

            if s > best_score:
                best = candidate
                best_score = s
        return best, best_score
    
    def select_diverse(self, candidates):
        candidates.sort(key=lambda x: x[0], reverse=True)
        selected = []

        for score, sol in candidates:
            if not selected:
                selected.append((score, sol))
                continue

            distances = [
                self.solution_distance(sol, s[1]) for s in selected
            ]

            if min(distances) > self.diversity_threshold:
                selected.append((score, sol))

            if len(selected) >= self.beam_width:
                break

        # 不够就补（防止过严）
        if len(selected) < self.beam_width:
            selected += candidates[:self.beam_width - len(selected)]

        return selected
    
    def update_diversity(self, current_best):
        if current_best > self.prev_best_score:
            self.no_improve_steps = 0
            self.prev_best_score = current_best

            self.diversity_threshold *= 0.9
        else:
            self.no_improve_steps += 1
            if self.no_improve_steps >= 3:
                self.diversity_threshold *= 1.2
        
        self.diversity_threshold = min(max(self.diversity_threshold, self.min_diversity), self.max_diversity)

    def solve(self):

        # beam: [(score, solution)]
        beam = self.init_population()

        best_solution = beam[0][1]
        best_score = beam[0][0]

        for step in range(self.max_iter):

            candidates = []
            seen = set()

            for score, sol in beam:
                neighbors = self._generate_neighbors(sol)

                for n in neighbors:
                    key = encode_solution(n)
                    if key not in seen:
                        seen.add(key)
                        candidates.append(n)
            scored = self.parallel_score(candidates)
            # 约束取值
            beam = self.select_diverse(scored)

            self.update_diversity(beam[0][0])
            for i in range(min(2, len(beam))):
                sols = beam[i][1]
                improved_sol, improved_score = self.local_search(sol)
                beam[i] = (improved_score, improved_sol)
            
            if beam[0][0] > best_score:
                best_solution = beam[0][1]
                best_score = beam[0][0]

            print(f"[Step {step}] best_score={best_score:.4f}, cache={len(self.cache)}")

        return best_solution, best_score
    
    def parallel_score(self, solutions):
        scores = Parallel(n_jobs = -1)(
            delayed(self._score)(sol) for sol in solutions
        )
        return list(zip(scores, solutions))
    
    @staticmethod
    def solution_distance(sol1, sol2):
        s1 = set(tuple(sorted(g)) for g in sol1)
        s2 = set(tuple(sorted(g)) for g in sol2)
        return 1 - len(s1 & s2) / len(s1 | s2)