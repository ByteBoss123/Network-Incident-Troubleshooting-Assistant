"""Core graph and stream algorithms, written from scratch (standard library only).

Used for incident correlation and blast-radius queries; each one is checked against NetworkX / sorting
on the project's real data in src/algo_bench.py and tests/test_pipeline.py.

    UnionFind               near-constant amortized union/find (path halving + union by size)
    correlate_incidents     group alerts into incidents: two alerts join if they share a source IP or a
                            destination IP and are at most `window_s` apart. Sort-and-sweep per key, so
                            O(n log n) instead of comparing every pair (O(n^2)).
    k_hop_neighbors         blast radius: BFS from a host, all hosts within k hops. O(V + E)
    dijkstra                weighted shortest paths with a binary heap. O((V + E) log V)
    top_k                   streaming top-k with a size-k min-heap. O(n log k), O(k) memory
"""
import heapq
from collections import deque
from itertools import count


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1] * n
        self.sets = n

    def find(self, x):
        p = self.parent
        while p[x] != x:
            p[x] = p[p[x]]  # path halving
            x = p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        self.sets -= 1
        return True

    def groups(self):
        out = {}
        for i in range(len(self.parent)):
            out.setdefault(self.find(i), []).append(i)
        return list(out.values())


def correlate_incidents(events, window_s=1800, keys=("src", "dst")):
    """events: list of dicts with 't' (epoch seconds) and the key fields. Returns a list of index lists."""
    uf = UnionFind(len(events))
    for k in keys:
        by_key = {}
        for i, e in enumerate(events):
            by_key.setdefault(e[k], []).append(i)
        for idx in by_key.values():
            idx.sort(key=lambda i: events[i]["t"])
            # Within one key, sorted by time, "within window_s" is transitive along neighbours, so joining
            # each event to the next one when the gap is <= window_s yields the same groups as all pairs.
            for a, b in zip(idx, idx[1:]):
                if events[b]["t"] - events[a]["t"] <= window_s:
                    uf.union(a, b)
    return uf.groups()


def correlate_incidents_naive(events, window_s=1800, keys=("src", "dst")):
    """Reference: compare every pair, O(n^2). Returns the edge list for an independent component check."""
    edges = []
    for i in range(len(events)):
        for j in range(i + 1, len(events)):
            if abs(events[i]["t"] - events[j]["t"]) <= window_s and any(events[i][k] == events[j][k] for k in keys):
                edges.append((i, j))
    return edges


def k_hop_neighbors(adj, source, k):
    """adj: {node: iterable of neighbours}. Returns {node: hops} for every node within k hops (source = 0)."""
    dist = {source: 0}
    q = deque([source])
    while q:
        u = q.popleft()
        if dist[u] == k:
            continue
        for v in adj[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def dijkstra(adj, source):
    """adj: {node: {neighbour: weight >= 0}}. Returns ({node: distance}, {node: predecessor})."""
    dist, prev = {source: 0.0}, {source: None}
    tie = count()
    heap = [(0.0, next(tie), source)]
    done = set()
    while heap:
        d, _, u = heapq.heappop(heap)
        if u in done:
            continue  # stale entry (lazy deletion instead of decrease-key)
        done.add(u)
        for v, w in adj[u].items():
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v], prev[v] = nd, u
                heapq.heappush(heap, (nd, next(tie), v))
    return dist, prev


def path_to(prev, target):
    out = []
    while target is not None:
        out.append(target)
        target = prev[target]
    return out[::-1]


def top_k(items, k, key):
    """Largest k items by key from any iterable, one pass; ties broken by first occurrence."""
    heap = []
    for i, it in enumerate(items):
        entry = (key(it), -i, it)
        if len(heap) < k:
            heapq.heappush(heap, entry)
        elif entry[:2] > heap[0][:2]:
            heapq.heapreplace(heap, entry)
    return [e[2] for e in sorted(heap, key=lambda e: e[:2], reverse=True)]
