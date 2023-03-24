#include <array>
#include <cassert>
#include <cstring>
#include <iostream>
#include <vector>
#include <x86intrin.h>

using namespace std;

using i8 = signed char;
using ll = long long;
using ull = unsigned long long;

template <typename T, int max_size> struct Stack {
    // T は memcpy でコピーできてデストラクタを呼ぶ必要が無いことを仮定
    using value_type = T;
    int siz;
    array<T, max_size> arr;
    Stack() = default;
    Stack(const Stack& rhs) {
        siz = rhs.siz;
        memcpy(&*arr.begin(), &*rhs.arr.begin(), rhs.siz * sizeof(T));
    }
    Stack& operator=(const Stack& rhs) {
        siz = rhs.siz;
        memcpy(&*arr.begin(), &*rhs.arr.begin(), rhs.siz * sizeof(T));
    }
    void push_back(const T& value) {
        assert(siz < max_size);
        arr[siz++] = value;
    }
    size_t size() const { return siz; }
    void clear() { siz = 0; }
};

struct Input {
    int D;
    int n_pixels;
    array<array<array<short, 14>, 14>, 2> fronts;
    array<array<array<short, 14>, 14>, 2> rights;
    void Read() {
        cin >> D;
        string s;
        n_pixels = 0;
        for (auto i = 0; i < 2; i++) {
            for (auto y = 0; y < D; y++) {
                cin >> s;
                for (auto x = 0; x < D; x++)
                    fronts[i][y][x] = s[x] == '1' ? n_pixels++ : -1;
            }
            for (auto y = 0; y < D; y++) {
                cin >> s;
                for (auto x = 0; x < D; x++)
                    rights[i][y][x] = s[x] == '1' ? n_pixels++ : -1;
            }
        }
    }
};

static Input input;

struct alignas(4) Vec3 {
    i8 x, y, z, w;
    Vec3() = default;
    Vec3(i8 x, i8 y, i8 z, i8 w = -1) : x(x), y(y), z(z), w(w) {}
    Vec3 operator+(const Vec3& rhs) const {
        return Vec3(x + rhs.x, y + rhs.y, z + rhs.z, w);
    }
    const auto& operator[](const int idx) const {
        assert(idx >= 0);
        assert(idx < 3);
        return idx == 0 ? x : idx == 1 ? y : z;
    }
    bool IsIn() const {
        const auto& D = input.D;
        return 0 <= x && x < D && 0 <= y && y < D && 0 <= z && z < D;
    }
    void Print(ostream& os) const {
        os << (int)x << "," << (int)y << "," << (int)z;
    }
};
template <typename T> struct Cube : array<array<array<T, 14>, 14>, 14> {
    void Visualize(ostream& os = cout) const {
        const auto& D = input.D;
        for (auto x = 0; x < D; x++) {
            for (auto y = 0; y < D; y++) {
                for (auto z = 0; z < D; z++) {
                    os << (*this)[x][y][z] << " ";
                }
            }
        }
        os << endl;
    }
};
namespace info {

static array<int, 2> n_nodes;
static array<Cube<short>, 2> coord_to_node_id;

struct Node {
    // vector<int> edge_ids;
    array<short, 2> pixel_ids;
    Vec3 coord;
};

struct Pixel {
    Stack<short, 14> node_ids;
};

struct alignas(64) Silhouette {
    array<ull, 8> data;

    void Set(short i) { data[i / 64] |= 1ull << i % 64; }
    void Reset(short i) { data[i / 64] ^= data[i / 64] & 1ull << i % 64; }
    void clear() { data = {}; }
    __m256i* AsM256i() { return (__m256i*)&data[0]; }
    const __m256i* AsM256i() const { return (__m256i*)&data[0]; }
    Silhouette& operator|=(const Silhouette& rhs) {
        AsM256i()[0] = _mm256_or_si256(AsM256i()[0], rhs.AsM256i()[0]);
        AsM256i()[1] = _mm256_or_si256(AsM256i()[1], rhs.AsM256i()[1]);
        return *this;
    }
};

static vector<Pixel> pixels;
static vector<Node> nodes;
static Silhouette full_silhouette;

struct Edge {
    struct Neighbour {
        int edge_id;
        i8 direction;
    };
    array<short, 2> node_ids;
    Stack<Neighbour, 6> neighbours;
    int edge_group_id;
    Silhouette silhouette;
};

static vector<Edge> edges;

struct EdgeGroup {
    vector<int> edge_ids;
    Silhouette silhouette; // TODO
    void Visualize(ostream& os = cout) const {
        auto out = array<Cube<int>, 2>();
        for (const auto edge_id : edge_ids) {
            const auto& e = edges[edge_id];
            for (auto i = 0; i < 2; i++) {
                const auto node_id = e.node_ids[i];
                const auto v = nodes[node_id].coord;
                out[i][v.x][v.y][v.z] = 1;
            }
        }
        os << 1 << endl;
        out[0].Visualize(os);
        out[1].Visualize(os);
    }
    auto size() const { return edge_ids.size(); }
};

static vector<EdgeGroup> edge_groups;
static array<array<Stack<int, 50>, 14 * 14 * 14 / 2>, 2>
    candidate_edge_ids_for_each_node; // TODO: 50

static void Init() {
    for (auto&& a : candidate_edge_ids_for_each_node)
        for (auto&& b : a)
            b.clear();

    const auto& D = input.D;
    n_nodes = {};

    const auto check_ok = [](const int i, const Vec3& v) {
        return input.fronts[i][v.z][v.x] != -1 &&
               input.rights[i][v.z][v.y] != -1;
    };
    const auto ok0 = [&check_ok](const Vec3& v) { return check_ok(0, v); };

    // pixels
    pixels.clear();
    pixels.resize(input.n_pixels);
    full_silhouette.clear();
    for (auto i = 0; i < input.n_pixels; i++)
        full_silhouette.Set(i);

    // nodes
    nodes.clear();
    for (auto i = 0; i < 2; i++) {
        for (auto x = 0; x < D; x++) {
            for (auto y = 0; y < D; y++) {
                for (auto z = 0; z < D; z++) {
                    if (input.fronts[i][z][x] != -1 &&
                        input.rights[i][z][y] != -1) {
                        n_nodes[i]++;
                        coord_to_node_id[i][x][y][z] = nodes.size();
                        const auto pixel_ids = array<short, 2>{
                            input.fronts[i][z][x], input.rights[i][z][y]};
                        nodes.push_back({
                            pixel_ids,
                            Vec3(x, y, z, i),
                        });
                        for (const auto pixel_id : pixel_ids) {
                            pixels[pixel_id].node_ids.push_back(nodes.size() -
                                                                1);
                        }
                    } else {
                        coord_to_node_id[i][x][y][z] = -1;
                    }
                }
            }
        }
    }

    // 辺を構築
    edges.clear();
    auto tmp_edges = vector<pair<short, short>>();
    auto tmp_edge_groups = vector<pair<int, int>>();
    edge_groups.clear();
    for (auto p : {
             array<int, 6>{0, 1, 2, 0, 0, 0},
             {0, 1, 2, 0, 1, 1},
             {0, 1, 2, 1, 0, 1},
             {0, 1, 2, 1, 1, 0},
             {1, 2, 0, 0, 0, 0},
             {1, 2, 0, 0, 1, 1},
             {1, 2, 0, 1, 0, 1},
             {1, 2, 0, 1, 1, 0},
             {2, 0, 1, 0, 0, 0},
             {2, 0, 1, 0, 1, 1},
             {2, 0, 1, 1, 0, 1},
             {2, 0, 1, 1, 1, 0},
             {0, 2, 1, 0, 0, 1},
             {0, 2, 1, 0, 1, 0},
             {0, 2, 1, 1, 0, 0},
             {0, 2, 1, 1, 1, 1},
             {1, 0, 2, 0, 0, 1},
             {1, 0, 2, 0, 1, 0},
             {1, 0, 2, 1, 0, 0},
             {1, 0, 2, 1, 1, 1},
             {2, 1, 0, 0, 0, 1},
             {2, 1, 0, 0, 1, 0},
             {2, 1, 0, 1, 0, 0},
             {2, 1, 0, 1, 1, 1},
         }) {
        auto coord_to_rotated_node_id = Cube<short>();
        for (auto x = 0; x < D; x++) {
            for (auto y = 0; y < D; y++) {
                for (auto z = 0; z < D; z++) {
                    const auto xyz = Vec3(x, y, z);
                    const auto tx = p[3] ? D - 1 - xyz[p[0]] : xyz[p[0]];
                    const auto ty = p[4] ? D - 1 - xyz[p[1]] : xyz[p[1]];
                    const auto tz = p[5] ? D - 1 - xyz[p[2]] : xyz[p[2]];
                    coord_to_rotated_node_id[x][y][z] =
                        coord_to_node_id[1][tx][ty][tz];
                }
            }
        }

        auto b = Vec3{};

        auto para = Vec3{};
        auto visited = Cube<bool>();
        for (para.x = 0; para.x < D; para.x++) {
            for (para.y = 0; para.y < D; para.y++) {
                for (para.z = 0; para.z < D; para.z++) {
                    auto siz =
                        Vec3(D - abs(para.x), D - abs(para.y), D - abs(para.z));
                    for (b.x = 0; b.x < siz.x; b.x++) {
                        for (b.y = 0; b.y < siz.y; b.y++) {
                            fill(visited[b.x][b.y].begin(),
                                 visited[b.x][b.y].begin() + siz.z, false);
                            for (b.z = 0; b.z < siz.z; b.z++) {
                                if (!ok0(Vec3(b.x + max(+para.x, 0),
                                              b.y + max(+para.y, 0),
                                              b.z + max(+para.z, 0))) ||
                                    coord_to_rotated_node_id
                                            [b.x + max(-para.x, 0)]
                                            [b.y + max(-para.y, 0)]
                                            [b.z + max(-para.z, 0)] == -1)
                                    visited[b.x][b.y][b.z] = true;
                            }
                        }
                    }

                    // node
                    for (b.x = 0; b.x < siz.x; b.x++) {
                        for (b.y = 0; b.y < siz.y; b.y++) {
                            for (b.z = 0; b.z < siz.z; b.z++) {
                                static auto q = array<Vec3, 14 * 14 * 14 / 2>();
                                if (visited[b.x][b.y][b.z])
                                    continue;
                                visited[b.x][b.y][b.z] = true;
                                q[0] = b;
                                auto left = 0;
                                auto right = 1;
                                while (left != right) {
                                    auto v = q[left++];
                                    if (v.x >= 1) {
                                        v.x--;
                                        if (!visited[v.x][v.y][v.z])
                                            visited[v.x][v.y][v.z] = true,
                                            q[right++] = v;
                                        v.x++;
                                    }
                                    if (v.x < siz.x - 1) {
                                        v.x++;
                                        if (!visited[v.x][v.y][v.z])
                                            visited[v.x][v.y][v.z] = true,
                                            q[right++] = v;
                                        v.x--;
                                    }
                                    if (v.y >= 1) {
                                        v.y--;
                                        if (!visited[v.x][v.y][v.z])
                                            visited[v.x][v.y][v.z] = true,
                                            q[right++] = v;
                                        v.y++;
                                    }
                                    if (v.y < siz.y - 1) {
                                        v.y++;
                                        if (!visited[v.x][v.y][v.z])
                                            visited[v.x][v.y][v.z] = true,
                                            q[right++] = v;
                                        v.y--;
                                    }
                                    if (v.z >= 1) {
                                        v.z--;
                                        if (!visited[v.x][v.y][v.z])
                                            visited[v.x][v.y][v.z] = true,
                                            q[right++] = v;
                                        v.z++;
                                    }
                                    if (v.z < siz.z - 1) {
                                        v.z++;
                                        if (!visited[v.x][v.y][v.z])
                                            visited[v.x][v.y][v.z] = true,
                                            q[right++] = v;
                                        v.z--;
                                    }
                                }
                                if (right < 5)
                                    continue;
                                tmp_edge_groups.emplace_back(
                                    tmp_edges.size(), tmp_edges.size() + right);
                                for (auto i = 0; i < right; i++) {
                                    const auto& v = q[i];
                                    tmp_edges.emplace_back(
                                        coord_to_node_id[0]
                                                        [v.x + max(+para.x, 0)]
                                                        [v.y + max(+para.y, 0)]
                                                        [v.z + max(+para.z, 0)],
                                        coord_to_rotated_node_id
                                            [v.x + max(-para.x, 0)]
                                            [v.y + max(-para.y, 0)]
                                            [v.z + max(-para.z, 0)]);
                                }
                            }
                        }
                    }
                }
            }
        }
        cerr << "tmp_edge_groups.size()=" << tmp_edge_groups.size() << endl;
        cerr << "tmp_edges.size()=" << tmp_edges.size() << endl;
    }

    sort(tmp_edge_groups.begin(), tmp_edge_groups.end(),
         [](const pair<int, int> a, const pair<int, int> b) {
             return a.second - a.first > b.second - b.first;
         });

    // 各頂点に対して、それを含むgroupの中で、サイズが大きい上位何個かを取り出す
    const auto n_candidate_edges_for_node = 50;
    auto n_candidate_groups = array<array<int, 14 * 14 * 14 / 2>, 2>();
    auto all_candidate_tmp_groups = vector<int>();

    for (auto idx_groups = 0; idx_groups < (int)tmp_edge_groups.size();
         idx_groups++) {
        const auto [l, r] = tmp_edge_groups[idx_groups];
        auto use = false;
        for (auto i = l; i < r; i++) {
            if (n_candidate_groups[0][tmp_edges[i].first] <
                n_candidate_edges_for_node)
                n_candidate_groups[0][tmp_edges[i].first]++, use = true;
            if (n_candidate_groups[1][tmp_edges[i].second] <
                n_candidate_edges_for_node)
                n_candidate_groups[1][tmp_edges[i].second]++, use = true;
        }
        if (use)
            all_candidate_tmp_groups.push_back(idx_groups);
    }
    for (auto group_id = 0; group_id < (int)all_candidate_tmp_groups.size();
         group_id++) {
        const auto& tmp_group = all_candidate_tmp_groups[group_id];
        const auto [l, r] = tmp_edge_groups[tmp_group];
        edge_groups.push_back({});
        auto& edge_group = edge_groups.back();
        static auto visited = Cube<int>();
        for (auto&& a : visited)
            for (auto&& b : a)
                fill(b.begin(), b.end(), -1);
        for (auto i = l; i < r; i++) {
            const auto edge_id = (int)edges.size();
            const auto e = tmp_edges[i];
            if (candidate_edge_ids_for_each_node[0][e.first].size() <
                n_candidate_edges_for_node)
                candidate_edge_ids_for_each_node[0][e.first].push_back(edge_id);
            if (candidate_edge_ids_for_each_node[1][e.second].size() <
                n_candidate_edges_for_node)
                candidate_edge_ids_for_each_node[1][e.second].push_back(
                    edge_id);
            auto v = nodes[e.first].coord;
            visited[v.x][v.y][v.z] = edge_id;
            auto silhouette = Silhouette{};
            for (const auto node_id : {e.first, e.second})
                for (const auto pixel_id : nodes[node_id].pixel_ids)
                    silhouette.Set(pixel_id);
            edges.push_back({
                {e.first, e.second},
                {}, // neighboring_edge_ids
                group_id,
                silhouette,
            });
            edge_group.edge_ids.push_back(edge_id);
            edge_group.silhouette |= silhouette;
            for (const auto& [direction, dxyz] : {
                     pair<i8, Vec3>{0, Vec3(0, 0, 1)},
                     {1, Vec3(0, 0, -1)},
                     {2, Vec3(0, 1, 0)},
                     {3, Vec3(0, -1, 0)},
                     {4, Vec3(1, 0, 0)},
                     {5, Vec3(-1, 0, 0)},
                 }) {
                const auto u = v + dxyz;
                if (u.IsIn()) {
                    const auto neighbour_edge_id = visited[u.x][u.y][u.z];
                    if (neighbour_edge_id != -1) {
                        edges[edge_id].neighbours.push_back(
                            {neighbour_edge_id, direction});
                        edges[neighbour_edge_id].neighbours.push_back(
                            {edge_id, (i8)(direction ^ 1)});
                    }
                }
            }
        }
    }

    cerr << "edge_groups.size()=" << edge_groups.size() << endl;
    cerr << "edge_groups[0].size()=" << edge_groups[0].size() << endl;
    for (const auto edge_id : edge_groups[0].edge_ids) {
        const auto& e = edges[edge_id];
        for (auto i = 0; i < 2; i++) {
            const auto node_id = e.node_ids[i];
            const auto v = nodes[node_id].coord;
            v.Print(cerr);
            cerr << " ";
        }
        cerr << endl;
    }
    edge_groups[0].Visualize();
    edge_groups[1].Visualize();
}

} // namespace info

static void Init() {
    input.Read();
    info::Init();
}

struct State {
    struct Core {
        int edge_id;
        array<bool, 6> dierction_priority;
    };
    Stack<Core, 200> cores;
};

static void Solve() {
    // TODO

    // TODO:
    // そのcore集合で全部のシルエットを埋められるかを、bitboardで先に確認する

    // 遷移
    // core をランダム?に変更
    // 埋めきれなかった場合、core を埋めきれなかった場所に追加
    // 埋めきれた場合、小さいブロックの core を取り除く
    // 古いやつとマージ
}

int main() {
    Init();
    Solve();
}

// 集合被覆問題
// * 集合全部使えるとは限らない