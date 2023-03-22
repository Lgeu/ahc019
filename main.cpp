#include <array>
#include <iostream>
#include <vector>

using namespace std;

// 回転 24 通りあるのか……
// 平行 2d^3 通り

using i8 = signed char;

template <typename T> using Cube = array<array<array<T, 14>, 14>, 14>;

struct alignas(4) Vec3 {
    i8 x, y, z;
    Vec3() = default;
    Vec3(i8 x, i8 y, i8 z) : x(x), y(y), z(z) {}
    Vec3 operator+(Vec3& rhs) const {
        return Vec3(x + rhs.x, y + rhs.y, z + rhs.z);
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

namespace info {

static array<int, 2> n_nodes;
static array<Cube<int>, 2> coord_to_node_id;
static array<vector<Vec3>, 2> node_id_to_coord;

struct Node {
    vector<int> edge_ids;
};

struct Edge {
    array<int, 2> node_ids;
    array<int, 6> neighboring_edge_ids;
    int edge_group_id;
};

struct EdgeGroup {
    vector<int> edge_ids;
};

static vector<Edge> edges;
static vector<EdgeGroup> edge_groups;

static void Init() {
    const auto& D = input.D;
    n_nodes = {};

    const auto check_ok = [](const int i, const Vec3& v) {
        return input.fronts[i][v.z][v.x] && input.rights[i][v.z][v.y];
    };
    const auto ok0 = [&check_ok](const Vec3& v) { return check_ok(0, v); };
    const auto ok1 = [&check_ok](const Vec3& v) { return check_ok(1, v); };

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
    auto tmp_edges = edges;
    auto tmp_edge_groups = vector<pair<int, int>>();
    edge_groups.clear();
    {
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
                                    !ok1(Vec3(b.x + max(-para.x, 0),
                                              b.y + max(-para.y, 0),
                                              b.z + max(-para.z, 0))))
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
                                tmp_edge_groups.emplace_back(tmp_edges.size(),
                                                             tmp_edges.size() +
                                                                 right - left);
                                for (auto i = 0; i < right; i++) {
                                    const auto& v = q[i];
                                    tmp_edges.push_back(
                                        {{
                                             coord_to_node_id
                                                 [0][v.x + max(+para.x, 0)]
                                                 [v.y + max(+para.y, 0)]
                                                 [v.z + max(+para.z, 0)],
                                             coord_to_node_id
                                                 [1][v.x + max(-para.x, 0)]
                                                 [v.y + max(-para.y, 0)]
                                                 [v.z + max(-para.z, 0)],
                                         },
                                         {}, // neightbour は後で
                                         -1} // group id も後で
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
        // TODO: 回転
        cerr << "tmp_edge_groups.size()=" << tmp_edge_groups.size() << endl;
        cerr << "tmp_edges.size()=" << tmp_edges.size() << endl;
    }

    // 各頂点に対して、それを含むgroupの中で、サイズが大きい上位何個かを取り出す
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