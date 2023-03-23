#include <array>
#include <cassert>
#include <iostream>
#include <vector>

using namespace std;

using i8 = signed char;

struct alignas(4) Vec3 {
    i8 x, y, z;
    Vec3() = default;
    Vec3(i8 x, i8 y, i8 z) : x(x), y(y), z(z) {}
    Vec3 operator+(Vec3& rhs) const {
        return Vec3(x + rhs.x, y + rhs.y, z + rhs.z);
    }
    const auto& operator[](const int idx) const {
        assert(idx >= 0);
        assert(idx < 3);
        return idx == 0 ? x : idx == 1 ? y : z;
    }
    void Print(ostream& os) const {
        os << (int)x << "," << (int)y << "," << (int)z;
    }
};

struct Input {
    int D;
    array<array<array<bool, 14>, 14>, 2> fronts;
    array<array<array<bool, 14>, 14>, 2> rights;
    void Read() {
        cin >> D;
        string s;
        for (auto i = 0; i < 2; i++) {
            for (auto y = 0; y < D; y++) {
                cin >> s;
                for (auto x = 0; x < D; x++)
                    fronts[i][y][x] = s[x] == '1';
            }
            for (auto y = 0; y < D; y++) {
                cin >> s;
                for (auto x = 0; x < D; x++)
                    rights[i][y][x] = s[x] == '1';
            }
        }
    }
};

static Input input;

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
static array<vector<Vec3>, 2> node_id_to_coord;

struct Node {
    vector<int> edge_ids;
};

struct Edge {
    array<short, 2> node_ids;
    array<int, 6> neighboring_edge_ids;
    int edge_group_id;
};

static vector<Edge> edges;
struct EdgeGroup {
    vector<int> edge_ids;
    void Visualize(ostream& os = cout) const {
        auto out = array<Cube<int>, 2>();
        for (const auto edge_id : edge_ids) {
            const auto& e = edges[edge_id];
            for (auto i = 0; i < 2; i++) {
                const auto node_id = e.node_ids[i];
                const auto v = node_id_to_coord[i][node_id];
                out[i][v.x][v.y][v.z] = 1;
            }
        }
        os << 1 << endl;
        out[0].Visualize(os);
        out[1].Visualize(os);
    }
};

static vector<EdgeGroup> edge_groups;
static array<array<vector<int>, 14 * 14 * 14 / 2>, 2>
    candidate_edge_ids_for_each_node; // TODO: スタックにする

static void Init() {
    for (auto a : candidate_edge_ids_for_each_node)
        for (auto b : a)
            fill(b.begin(), b.end(), -1);

    const auto& D = input.D;
    n_nodes = {};

    const auto check_ok = [](const int i, const Vec3& v) {
        return input.fronts[i][v.z][v.x] && input.rights[i][v.z][v.y];
    };
    const auto ok0 = [&check_ok](const Vec3& v) { return check_ok(0, v); };

    for (auto i = 0; i < 2; i++) {
        node_id_to_coord[i].clear();
        for (auto x = (i8)0; x < D; x++) {
            for (auto y = (i8)0; y < D; y++) {
                for (auto z = (i8)0; z < D; z++) {
                    if (check_ok(i, {x, y, z})) {
                        coord_to_node_id[i][x][y][z] = n_nodes[i]++;
                        node_id_to_coord[i].push_back({x, y, z});
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
        // cerr << "lr=" << l << " " << r << endl;
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
        for (auto i = l; i < r; i++) {
            const auto edge_id = (int)edges.size();
            edge_groups.back().edge_ids.push_back(edge_id);
            const auto e = tmp_edges[i];
            if (candidate_edge_ids_for_each_node[0][e.first].size() <
                n_candidate_edges_for_node)
                candidate_edge_ids_for_each_node[0][e.first].push_back(edge_id);
            if (candidate_edge_ids_for_each_node[1][e.second].size() <
                n_candidate_edges_for_node)
                candidate_edge_ids_for_each_node[1][e.second].push_back(
                    edge_id);
            edges.push_back({{e.first, e.second},
                             {}, // TODO: neighboring_edge_ids
                             group_id});
        }
    }

    cerr << "edge_groups.size()=" << edge_groups.size() << endl;
    for (const auto edge_id : edge_groups[0].edge_ids) {
        const auto& e = edges[edge_id];
        for (auto i = 0; i < 2; i++) {
            const auto node_id = e.node_ids[i];
            const auto v = node_id_to_coord[i][node_id];
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

int main() {
    Init();
    //
}