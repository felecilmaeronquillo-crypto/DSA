# ==========================================
# 0. SAFE IMPORT OF MATPLOTLIB
# ==========================================
try:
    import matplotlib
    try:
        matplotlib.use("TkAgg")
    except Exception:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    print("matplotlib loaded successfully ✅")
except ImportError:
    plt = None
    print("matplotlib is NOT installed ❌ — run: pip install matplotlib")

import heapq
import math


# ==========================================
# 1. HAVERSINE DISTANCE
# ==========================================
def haversine(v1, v2):
    R = 6371000
    lat1, lon1 = math.radians(v1.y), math.radians(v1.x)
    lat2, lon2 = math.radians(v2.y), math.radians(v2.x)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = (math.sin(dlat / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


# ==========================================
# 2. VERTEX CLASS
# ==========================================
class Vertex:
    X = 0.0
    Y = 0.0

    def __init__(self, name, x=0.0, y=0.0):
        self.name = name
        self.x = x
        self.y = y

    def __repr__(self):
        return f"{self.name} ({self.x}, {self.y})"


# ==========================================
# 3. GRAPH CLASS
# ==========================================
class Graph:
    def __init__(self):
        self.vertices = {}
        self.edges = []

    def add_vertex(self, vertex):
        self.vertices[vertex.name] = vertex

    def add_edge(self, name1, name2):
        self.edges.append((name1, name2))

    # -------- Fuzzy name matching --------
    def find_name(self, query):
        query = query.lower().strip()
        for name in self.vertices:
            if name.lower() == query:
                return name
        matches = [n for n in self.vertices if query in n.lower()]
        if not matches:
            raise ValueError(f"❌ No place found matching '{query}'")
        if len(matches) > 1:
            print(f"⚠️  Multiple matches for '{query}':")
            for m in matches:
                print(f"    - {m}")
            print(f"   → Using first match: {matches[0]}")
        return matches[0]

    # -------- SHORTEST PATH (Dijkstra) --------
    def shortest_path(self, start_name, end_name):
        start_name = self.find_name(start_name)
        end_name = self.find_name(end_name)

        adj = {n: [] for n in self.vertices}
        for a, b in self.edges:
            d = haversine(self.vertices[a], self.vertices[b])
            adj[a].append((b, d))
            adj[b].append((a, d))

        distances = {n: float("inf") for n in self.vertices}
        distances[start_name] = 0
        previous = {n: None for n in self.vertices}
        visited = set()
        pq = [(0, start_name)]

        while pq:
            d, u = heapq.heappop(pq)
            if u in visited:
                continue
            visited.add(u)
            if u == end_name:
                break
            for v, w in adj[u]:
                nd = d + w
                if nd < distances[v]:
                    distances[v] = nd
                    previous[v] = u
                    heapq.heappush(pq, (nd, v))

        if distances[end_name] == float("inf"):
            raise ValueError(f"❌ No path from '{start_name}' to '{end_name}'")

        path = []
        node = end_name
        while node is not None:
            path.append(node)
            node = previous[node]
        path.reverse()

        return path, distances[end_name], start_name, end_name

    # -------- DRAW MAP (with connected route dots) --------
    def draw_map(self, path=None, filename="myeongdong_map.png"):
        if plt is None:
            print("\n--- Cannot draw map (matplotlib missing) ---")
            return

        fig, ax = plt.subplots(figsize=(16, 11))

        # 1. Draw ALL streets in light gray (background)
        for a, b in self.edges:
            v1, v2 = self.vertices[a], self.vertices[b]
            ax.plot([v1.x, v2.x], [v1.y, v2.y],
                    color="lightgray", linewidth=1.2, zorder=1)

        # 2. Draw the path in BLUE with connected lines
        if path:
            for i in range(len(path) - 1):
                v1 = self.vertices[path[i]]
                v2 = self.vertices[path[i + 1]]
                ax.plot([v1.x, v2.x], [v1.y, v2.y],
                        color="blue", linewidth=4.5, zorder=2,
                        label="Route" if i == 0 else "")

        # 3. Draw ALL vertices — but path vertices get BLUE & bigger
        green_labels = {"A", "B", "C", "D", "E", "F", "G", "H"}
        path_set = set(path) if path else set()

        for v in self.vertices.values():
            on_path = v.name in path_set
            is_letter = v.name[0] in green_labels

            if on_path:
                # 🔵 Big blue dot on the route
                color, size, weight = "blue", 260, "bold"
            elif is_letter:
                # 🟢 Green for A–H (not on route)
                color, size, weight = "green", 200, "bold"
            else:
                # 🔴 Red for others
                color, size, weight = "red", 110, "normal"

            ax.scatter(v.x, v.y, s=size, color=color,
                       edgecolors="black", linewidths=1.2, zorder=3)

            ax.annotate(v.name, (v.x, v.y),
                        xytext=(0, 10), textcoords="offset points",
                        fontsize=6.5, ha="center", fontweight=weight,
                        color="black", zorder=4,
                        bbox=dict(boxstyle="round,pad=0.15",
                                  fc="white", ec="none", alpha=0.75))

        # 4. Number the steps on the path (1, 2, 3...)
        if path:
            for i, name in enumerate(path):
                v = self.vertices[name]
                ax.annotate(
                    str(i + 1),
                    xy=(v.x, v.y),
                    xytext=(14, -14),           # bottom-right of the dot
                    textcoords="offset points",
                    fontsize=9, fontweight="bold",
                    color="white", zorder=6,
                    bbox=dict(boxstyle="circle,pad=0.35",
                              fc="blue", ec="white", linewidth=1.2)
                )

        # 5. Blue ring around start and end
        if path:
            for endpoint in (path[0], path[-1]):
                v = self.vertices[endpoint]
                ax.scatter(v.x, v.y, s=550, facecolors="none",
                           edgecolors="blue", linewidths=3.5, zorder=5)

        # 6. Auto-scale axes
        xs = [v.x for v in self.vertices.values()]
        ys = [v.y for v in self.vertices.values()]
        mx = (max(xs) - min(xs)) * 0.15 or 0.001
        my = (max(ys) - min(ys)) * 0.15 or 0.001
        ax.set_xlim(min(xs) - mx, max(xs) + mx)
        ax.set_ylim(min(ys) - my, max(ys) + my)

        # 7. Title and labels
        title = "Myeongdong, Seoul — Street Graph"
        if path:
            title += (f"\nRoute: {path[0]}  →  {path[-1]}  "
                      f"({len(path)} stops)")
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("Longitude (X)")
        ax.set_ylabel("Latitude (Y)")
        ax.grid(True, linestyle="--", alpha=0.4)
        if path:
            ax.legend(loc="upper right")

        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        print(f"Map saved as {filename} ✅")

        if matplotlib.get_backend().lower() != "agg":
            plt.show()


# ==========================================
# 4. BUILD THE GRAPH — 32 vertices, 42 edges
# ==========================================
g = Graph()

places = [
    ("A. Myeongdong Cathedral",             126.9835, 37.5635),
    ("B. Myeongdong-gil West",              126.9840, 37.5630),
    ("C. Myeongdong-gil East",              126.9855, 37.5630),
    ("D. Toegye-ro 20-gil",                 126.9835, 37.5620),
    ("E. Central Myeongdong",               126.9848, 37.5620),
    ("F. Myeongdong 8-gil",                 126.9860, 37.5615),
    ("G. Myeongdong 7-gil & Toegye-ro",     126.9845, 37.5605),
    ("H. Myeongdong Station",               126.9860, 37.5605),
    ("Myeongdong Shopping St",              126.9845, 37.5638),
    ("Shinsegae Dept Store",                126.9865, 37.5645),
    ("Lotte Dept Store",                    126.9855, 37.5615),
    ("Nike Store",                          126.9850, 37.5610),
    ("Hotel Skypark Central",               126.9820, 37.5615),
    ("Subway Exit 6",                       126.9843, 37.5603),
    ("Subway Exit 7",                       126.9863, 37.5603),
    ("Subway Exit 8",                       126.9865, 37.5602),
    ("Myeongdong 7-gil North",              126.9848, 37.5625),
    ("Myeongdong 8-gil North",              126.9858, 37.5628),
    ("Myeongdong 8-gil South",              126.9858, 37.5610),
    ("Toegye-ro West",                      126.9820, 37.5608),
    ("Toegye-ro East",                      126.9870, 37.5600),
    ("Myeongdong-gil West End",             126.9830, 37.5632),
    ("Myeongdong-gil East End",             126.9868, 37.5628),
    ("Toegye-ro 20-gil North",              126.9832, 37.5625),
    ("Toegye-ro 20-gil South",              126.9830, 37.5610),
    ("Myeongdong 7-gil South",              126.9845, 37.5610),
    ("Myeongdong 8-gil Middle",             126.9858, 37.5620),
    ("Shopping Alley 1",                    126.9842, 37.5628),
    ("Shopping Alley 2",                    126.9852, 37.5622),
    ("Food Street",                         126.9847, 37.5632),
    ("Nike Store Entrance",                 126.9852, 37.5612),
    ("Lotte Store Entrance",                126.9857, 37.5616),
]

for name, x, y in places:
    g.add_vertex(Vertex(name, x, y))

streets = [
    ("A. Myeongdong Cathedral",         "B. Myeongdong-gil West"),
    ("B. Myeongdong-gil West",          "C. Myeongdong-gil East"),
    ("A. Myeongdong Cathedral",         "D. Toegye-ro 20-gil"),
    ("B. Myeongdong-gil West",          "E. Central Myeongdong"),
    ("C. Myeongdong-gil East",          "F. Myeongdong 8-gil"),
    ("D. Toegye-ro 20-gil",             "E. Central Myeongdong"),
    ("E. Central Myeongdong",           "F. Myeongdong 8-gil"),
    ("E. Central Myeongdong",           "G. Myeongdong 7-gil & Toegye-ro"),
    ("F. Myeongdong 8-gil",             "H. Myeongdong Station"),
    ("G. Myeongdong 7-gil & Toegye-ro", "H. Myeongdong Station"),
    ("Myeongdong Shopping St",  "A. Myeongdong Cathedral"),
    ("Myeongdong Shopping St",  "B. Myeongdong-gil West"),
    ("Shinsegae Dept Store",    "C. Myeongdong-gil East"),
    ("Shinsegae Dept Store",    "Myeongdong-gil East End"),
    ("Lotte Dept Store",        "E. Central Myeongdong"),
    ("Lotte Dept Store",        "F. Myeongdong 8-gil"),
    ("Nike Store",              "G. Myeongdong 7-gil & Toegye-ro"),
    ("Nike Store",              "H. Myeongdong Station"),
    ("Hotel Skypark Central",   "D. Toegye-ro 20-gil"),
    ("Hotel Skypark Central",   "Toegye-ro West"),
    ("Subway Exit 6",           "G. Myeongdong 7-gil & Toegye-ro"),
    ("Subway Exit 7",           "H. Myeongdong Station"),
    ("Subway Exit 8",           "H. Myeongdong Station"),
    ("Myeongdong 7-gil North",  "B. Myeongdong-gil West"),
    ("Myeongdong 7-gil North",  "E. Central Myeongdong"),
    ("Myeongdong 8-gil North",  "C. Myeongdong-gil East"),
    ("Myeongdong 8-gil North",  "F. Myeongdong 8-gil"),
    ("Myeongdong 8-gil South",  "F. Myeongdong 8-gil"),
    ("Myeongdong 8-gil South",  "H. Myeongdong Station"),
    ("Toegye-ro West",          "D. Toegye-ro 20-gil"),
    ("Toegye-ro West",          "G. Myeongdong 7-gil & Toegye-ro"),
    ("Toegye-ro East",          "H. Myeongdong Station"),
    ("Myeongdong-gil West End", "B. Myeongdong-gil West"),
    ("Myeongdong-gil East End", "C. Myeongdong-gil East"),
    ("Toegye-ro 20-gil North",  "D. Toegye-ro 20-gil"),
    ("Toegye-ro 20-gil South",  "Toegye-ro West"),
    ("Myeongdong 7-gil South",  "G. Myeongdong 7-gil & Toegye-ro"),
    ("Myeongdong 7-gil South",  "E. Central Myeongdong"),
    ("Myeongdong 8-gil Middle", "F. Myeongdong 8-gil"),
    ("Myeongdong 8-gil Middle", "Myeongdong 8-gil North"),
    ("Shopping Alley 1",        "Myeongdong Shopping St"),
    ("Shopping Alley 1",        "B. Myeongdong-gil West"),
    ("Shopping Alley 2",        "E. Central Myeongdong"),
    ("Shopping Alley 2",        "Myeongdong Shopping St"),
    ("Food Street",             "Myeongdong Shopping St"),
    ("Food Street",             "B. Myeongdong-gil West"),
    ("Nike Store Entrance",     "Nike Store"),
    ("Nike Store Entrance",     "H. Myeongdong Station"),
    ("Lotte Store Entrance",    "Lotte Dept Store"),
    ("Lotte Store Entrance",    "F. Myeongdong 8-gil"),
]

for a, b in streets:
    g.add_edge(a, b)


# ==========================================
# 5. SHOW ROUTE + DRAW
# ==========================================
def show_route(start, end):
    try:
        path, dist, s_full, e_full = g.shortest_path(start, end)
    except ValueError as e:
        print(e)
        return None

    print(f"\n=== SHORTEST PATH: {len(path)} stops ===")
    print(f"From: {s_full}")
    print(f"To:   {e_full}\n")
    for i, node in enumerate(path, 1):
        print(f"  {i}. {node}")
    print(f"\nTotal distance: {dist:.1f} meters")
    return path


# ==========================================
# 6. INTERACTIVE INPUT LOOP
# ==========================================
print("\n============================================")
print("   Myeongdong Shortest Path Finder")
print("============================================")
print("Available places:")
for name in g.vertices:
    print("  •", name)

while True:
    print("\n--------------------------------------------")
    start = input("Start (or 'q' to quit): ").strip()
    if start.lower() == "q":
        break
    end = input("End: ").strip()

    path = show_route(start, end)
    if path:
        safe = lambda s: "".join(c if c.isalnum() else "_" for c in s)[:20]
        fname = f"route_{safe(start)}_to_{safe(end)}.png"
        g.draw_map(path=path, filename=fname)

print("\nGoodbye! 👋")