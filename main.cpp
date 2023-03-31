#include <algorithm>
#include <array>
#include <cassert>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <tuple>
#include <vector>
#include <x86intrin.h>

using namespace std;

using i8 = signed char;
using u8 = unsigned char;
using ll = long long;
using ull = unsigned long long;

// auto rng = mt19937(42);

struct Random {
    using result_type = unsigned;
    ull state;
    Random(unsigned seed) : state(seed) {
        assert(seed != 0);
        for (auto i = 0; i < 5; i++)
            (*this)();
    }
    unsigned operator()() {
        state ^= state << 7;
        state ^= state >> 9;
        return (unsigned)state;
    }
    int RandInt(int mini, int maxi) {
        auto x = (ull)(*this)();
        x *= maxi - mini + 1;
        x >>= 32;
        return mini + (int)x;
    }
    static constexpr unsigned max() { return 0xffffffffu; }
    static constexpr unsigned min() { return 1u; }
};

auto rng = Random(42);

static inline int CountRightZero(const ull x) {
#ifdef _MSC_VER
    unsigned long r;
    _BitScanForward64(&r, x);
    return (int)r;
#else
    return __builtin_ctzll(x);
#endif
}

static inline int PopCount(const ull x) {
#ifdef _MSC_VER
    return (int)__popcnt64(x);
#else
    return __builtin_popcountll(x);
#endif
}

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
        return *this;
    }
    void push_back(const T& value) {
        assert(siz < max_size);
        arr[siz++] = value;
    }
    void pop_back() {
        assert(siz > 0);
        siz--;
    }
    size_t size() const { return siz; }
    void clear() { siz = 0; }
    const T& operator[](const int idx) const { return arr[idx]; }
    T& operator[](const int idx) { return arr[idx]; }
    const T* begin() const { return &*arr.begin(); }
    const T* end() const { return begin() + siz; }
};

template <typename T, int buffer_size, int initial_offset = buffer_size / 2>
struct Queue {
    using value_type = T;
    int left, right;
    array<T, buffer_size> arr;
    Queue() : left(initial_offset), right(initial_offset), arr() {}
    Queue(const Queue& rhs) {
        left = rhs.left;
        right = rhs.right;
        memcpy(&*arr.begin() + rhs.left, &*rhs.arr.begin() + rhs.left,
               (rhs.right - rhs.left) * sizeof(T));
    }
    void push_back(const T& value) {
        assert(right < buffer_size);
        arr[right++] = value;
    }
    void push_front(const T& value) {
        assert(left > 0);
        arr[--left] = value;
    }
    T& front() { return arr[left]; }
    void pop_front() {
        assert(left != right);
        left++;
    }
    size_t size() const { return right - left; }
    bool empty() const { return right == left; }
    void clear() {
        left = initial_offset;
        right = initial_offset;
    }
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
    vector<int> edge_ids;
};

struct alignas(64) Silhouette {
    array<ull, 8> data;

    void Set(short i) { data[i / 64] |= 1ull << i % 64; }
    void Reset(short i) { data[i / 64] ^= data[i / 64] & 1ull << i % 64; }
    void clear() { data = {}; }
    __m256i* AsM256i() { return (__m256i*)&data[0]; }
    const __m256i* AsM256i() const { return (__m256i*)&data[0]; }
    const u8* AsU8() const { return (u8*)&data[0]; }
    inline Silhouette& operator|=(const Silhouette& rhs) {
        AsM256i()[0] = _mm256_or_si256(AsM256i()[0], rhs.AsM256i()[0]);
        AsM256i()[1] = _mm256_or_si256(AsM256i()[1], rhs.AsM256i()[1]);
        return *this;
    }
    inline Silhouette& operator^=(const Silhouette& rhs) {
        AsM256i()[0] = _mm256_xor_si256(AsM256i()[0], rhs.AsM256i()[0]);
        AsM256i()[1] = _mm256_xor_si256(AsM256i()[1], rhs.AsM256i()[1]);
        return *this;
    }
    inline Silhouette& operator&=(const Silhouette& rhs) {
        AsM256i()[0] = _mm256_and_si256(AsM256i()[0], rhs.AsM256i()[0]);
        AsM256i()[1] = _mm256_and_si256(AsM256i()[1], rhs.AsM256i()[1]);
        return *this;
    }
    inline Silhouette& AndNotAssign(const Silhouette& rhs) {
        AsM256i()[0] = _mm256_andnot_si256(rhs.AsM256i()[0], AsM256i()[0]);
        AsM256i()[1] = _mm256_andnot_si256(rhs.AsM256i()[1], AsM256i()[1]);
        return *this;
    }
    inline bool empty() const {
        auto tmp = _mm256_or_si256(AsM256i()[0], AsM256i()[1]);
        return _mm256_testz_si256(tmp, tmp);
    }
    inline int CountRightZero() const {
        // empty で無いことを仮定
        const __m256i x0 = _mm256_cmpeq_epi8(
            AsM256i()[0], _mm256_setzero_si256()); // 8bit x 32
        const __m256i x1 =
            _mm256_cmpeq_epi8(AsM256i()[1], _mm256_setzero_si256());
        const auto nonzero_bytes = ~((ull)(unsigned)_mm256_movemask_epi8(x0) |
                                     (ull)_mm256_movemask_epi8(x1) << 32);
        if (nonzero_bytes == 0) {
            cout << "_mm256_movemask_epi8(x1)=" << _mm256_movemask_epi8(x1)
                 << endl;
            cout << "(ull)_mm256_movemask_epi8(x1)="
                 << (ull)_mm256_movemask_epi8(x1) << endl;
        }
        assert(nonzero_bytes != 0);
        const auto nonzero_byte = ::CountRightZero(nonzero_bytes);
        return nonzero_byte * 8 + ::CountRightZero(AsU8()[nonzero_byte]);
    }
    inline int PopCount() const {
        // 必要なら後で高速化
        auto res = 0;
        for (auto i = 0; i < 7; i++) // 7 まででいい
            res += ::PopCount(data[i]);
        return res;
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
    Silhouette silhouette;
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
static constexpr auto n_candidate_edges_for_node = 50; // パラメータ
static array<array<Stack<int, n_candidate_edges_for_node>, 14 * 14 * 14 / 2>, 2>
    candidate_edge_ids_for_each_node;

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
        for (para.x = -D + 1; para.x < D; para.x++) {
            for (para.y = -D + 1; para.y < D; para.y++) {
                for (para.z = -D + 1; para.z < D; para.z++) {
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
                for (const auto pixel_id : nodes[node_id].pixel_ids) {
                    silhouette.Set(pixel_id);
                    pixels[pixel_id].edge_ids.push_back(edge_id);
                }
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
    // for (const auto edge_id : edge_groups[0].edge_ids) {
    //     const auto& e = edges[edge_id];
    //     for (auto i = 0; i < 2; i++) {
    //         const auto node_id = e.node_ids[i];
    //         const auto v = nodes[node_id].coord;
    //         v.Print(cerr);
    //         cerr << " ";
    //     }
    //     cerr << endl;
    // }
    // edge_groups[0].Visualize();
    // edge_groups[1].Visualize();
}

} // namespace info

static void Init() {
    input.Read();
    info::Init();
}

struct Solution {
    bool success;
    double score;
    array<short, 5488> blocks;
};

struct State {
    struct Core {
        int edge_id;
        array<bool, 6> dierction_priority;
    };
    Stack<Core, 200> cores;

    auto SCPCovered() const {
        auto silhouette = info::Silhouette();
        for (const auto& core : cores)
            silhouette |=
                info::edge_groups[info::edges[core.edge_id].edge_group_id]
                    .silhouette;
        return silhouette;
    }

    auto BFS() const {
        // 最後まで行っても埋めきれないか、スコアが超過したかで終了
        struct BFSResult {
            enum struct Status {
                kSuccess,
                kFailureSilhouette,
                kFailureScoreExcess,
            };
            info::Silhouette visited_silhouette;
            array<short, 14 * 14 * 14 * 2> visited_nodes;
            array<short, 200> block_sizes;
            double score;
            Status status;
        };
        auto res = BFSResult{};
        fill(res.visited_nodes.begin(),
             res.visited_nodes.begin() + info::nodes.size(), -1);
        struct QueueElement {
            int edge_id;
        };
        static auto qs = vector<Queue<QueueElement, 14 * 14 * 14 * 2>>();
        if (qs.size() < cores.size())
            qs.resize(cores.size());
        for (auto core_id = 0; core_id < (int)cores.size(); core_id++) {
            const auto edge_id = cores[core_id].edge_id;
            const auto& edge = info::edges[edge_id];
            qs[core_id].clear();
            qs[core_id].push_back({edge_id});
            res.visited_silhouette |= edge.silhouette;
        }
        auto next_core_id = vector<short>(cores.size());
        iota(next_core_id.begin(), next_core_id.end() - 1, 1);
        next_core_id.back() = 0;
        auto last_core_id = (short)-1;
        for (auto core_id = 0;; core_id = next_core_id[core_id]) {
            const auto& core = cores[core_id];
            auto& q = qs[core_id];
            assert(!q.empty());
            int edge_id;
            while (!q.empty()) {
                edge_id = q.front().edge_id;
                q.pop_front();
                const auto& edge = info::edges[edge_id];
                if (res.visited_nodes[edge.node_ids[0]] == -1 &&
                    res.visited_nodes[edge.node_ids[1]] == -1)
                    goto ok;
            }
            next_core_id[last_core_id] = next_core_id[core_id];
            if (next_core_id[core_id] == core_id)
                break;
            continue;
        ok:;
            {
                res.block_sizes[core_id]++;
                const auto& edge = info::edges[edge_id];
                res.visited_nodes[edge.node_ids[0]] = core_id;
                res.visited_nodes[edge.node_ids[1]] = core_id;
                res.visited_silhouette |= edge.silhouette;
                for (const auto& neighbour : edge.neighbours) {
                    if (core.dierction_priority[neighbour.direction]) {
                        q.push_back({neighbour.edge_id});
                    } else {
                        q.push_front({neighbour.edge_id});
                    }
                }
                if (q.empty()) { // ほとんど起こらないはず
                    next_core_id[last_core_id] = next_core_id[core_id];
                    if (next_core_id[core_id] == core_id)
                        break;
                    continue;
                }
                last_core_id = core_id;
            }
        }
        for (auto i = 0; i < (int)cores.size(); i++)
            res.score += 1.0 / (double)res.block_sizes[i];
        return res;
    }

    void Update() {
        // TODO
    }

    Solution Greedy() {
        auto scp_not_covered = info::full_silhouette;
        scp_not_covered ^= SCPCovered();

        // SCP を貪欲に解く
        while (!scp_not_covered.empty()) {
            auto best_edge_group_id = -100;
            auto best_value = 0;
            for (auto edge_group_id = 0;
                 edge_group_id < (int)info::edge_groups.size();
                 edge_group_id++) {
                auto edge_group_silhouette =
                    info::edge_groups[edge_group_id].silhouette;
                edge_group_silhouette &= scp_not_covered;
                const auto pc = edge_group_silhouette.PopCount();
                if (best_value < pc) {
                    best_value = pc;
                    best_edge_group_id = edge_group_id;
                }
            }
            const auto& best_edge_group = info::edge_groups[best_edge_group_id];
            scp_not_covered.AndNotAssign(best_edge_group.silhouette);

            while (true) {
                const auto r = uniform_int_distribution<>(
                    0, best_edge_group.size() - 1)(rng);
                const auto edge_id = best_edge_group.edge_ids[r];
                if (CheckAddable(edge_id)) {
                    cores.push_back({
                        edge_id,
                        {true, true, true, true, true, true},
                    });
                    break;
                }
            }
        }

        // 01BFS で広げてく
        while (true) {
            // BFS
            const auto bfs_result = BFS();
            // 全部のシルエット条件を満たしたか確認
            auto non_visited_silhouette = bfs_result.visited_silhouette;
            non_visited_silhouette ^= info::full_silhouette;
            if (non_visited_silhouette.empty())
                return {true, bfs_result.score, bfs_result.visited_nodes};
            // 満たしていない pixel のところに edge を追加
            const auto pixel_id = non_visited_silhouette.CountRightZero();
            const auto& pixel = info::pixels[pixel_id];
            for (const auto new_edge_id : pixel.edge_ids) {
                if (CheckAddable(new_edge_id)) {
                    cores.push_back({
                        new_edge_id,
                        //{false, false, false, false, false, false},
                        {true, true, true, true, true, true},
                    });
                    goto ok2;
                }
            }
            assert(false);
        ok2:;
        }
    }

    inline Solution Random(const int max_n_cores) {
        auto scp_not_covered = info::full_silhouette;
        scp_not_covered ^= SCPCovered();

        // SCP を解く
        auto trial = 0;
        while (!scp_not_covered.empty()) {
            if (max_n_cores <= (int)cores.size())
                return {false, 1e9, {}};
            if (trial++ >= 50)
                return {false, 1e9, {}};
            const auto edge_id = rng.RandInt(0, info::edges.size() - 1);
            const auto& edge = info::edges[edge_id];
            const auto edge_group_id = edge.edge_group_id;
            const auto& edge_group = info::edge_groups[edge_group_id];
            auto edge_group_silhouette = edge_group.silhouette;
            edge_group_silhouette &= scp_not_covered;
            if (edge_group_silhouette.empty())
                continue;
            if (!CheckAddable(edge_id))
                continue;
            cores.push_back({
                edge_id,
                // {false, false, false, false, false, false},
                {true, true, true, true, true, true},
            });
            trial = 0;
            scp_not_covered.AndNotAssign(edge_group.silhouette);
        }
        // 01BFS で広げてく
        while (true) {
            // BFS
            const auto bfs_result = BFS();
            // 全部のシルエット条件を満たしたか確認
            auto non_visited_silhouette = bfs_result.visited_silhouette;
            non_visited_silhouette ^= info::full_silhouette;
            if (non_visited_silhouette.empty())
                return {true, bfs_result.score, bfs_result.visited_nodes};
            // 失敗
            if (max_n_cores <= (int)cores.size())
                return {false, 1e9, {}};
            // 満たしていない pixel のところに edge を追加
            // この順番もランダムにした方が良いのか？
            const auto pixel_id = non_visited_silhouette.CountRightZero();
            const auto& pixel = info::pixels[pixel_id];
            // 探索順を決める、ここもうちょっと高速化はできそう
            static array<short, 20000> order;
            iota(order.begin(), order.begin() + pixel.edge_ids.size(), 0);
            for (auto i = (int)pixel.edge_ids.size() - 1; i >= 0; i--) {
                const auto idx_order = rng.RandInt(0, i);
                const auto new_edge_id = pixel.edge_ids[order[idx_order]];
                order[idx_order] = order[i];
                if (CheckAddable(new_edge_id)) {
                    cores.push_back({
                        new_edge_id,
                        // {false, false, false, false, false, false},
                        {true, true, true, true, true, true},
                    });
                    goto ok2;
                }
            }
            return {false, 1e9, {}};
            assert(false);
        ok2:;
        }
    }

    bool CheckAddable(const int edge_id) const {
        for (const auto& core : cores)
            for (const auto core_node_id : info::edges[core.edge_id].node_ids)
                for (const auto new_node_id : info::edges[edge_id].node_ids)
                    if (core_node_id == new_node_id)
                        return false;
        return true;
    }
};

static void Visualize(const array<short, 5488>& blocks) {
    auto puzzle = array<Cube<int>, 2>();
    auto n_blocks = 0;
    for (auto node_id = 0; node_id < (int)info::nodes.size(); node_id++) {
        const auto& p = info::nodes[node_id].coord;
        puzzle[p.w][p.x][p.y][p.z] = blocks[node_id] + 1;
        n_blocks = max(n_blocks, blocks[node_id] + 1);
    }

    cout << n_blocks << endl;
    puzzle[0].Visualize();
    puzzle[1].Visualize();
}

[[maybe_unused]] static void SolveGreedy() {
    auto state = State();
    auto blocks = state.Greedy().blocks;
    Visualize(blocks);
}

[[maybe_unused]] static void SolveRandom() {
    auto state = State();
    auto [success, _, blocks] = state.Random(200);
    assert(success);
    Visualize(blocks);
}

[[maybe_unused]] static void SolveRandomLoop() {
    array<short, 5488> best_blocks;
    auto min_score = 1e9;
    auto min_n_cores = 200;
    for (auto i = 0; i < 1e4; i++) {
        auto state = State();
        auto [success, score, blocks] = state.Random(min_n_cores);
        if (!success)
            continue;
        if ((int)state.cores.size() < min_n_cores) {
            min_n_cores = (int)state.cores.size();
        }
        if (score < min_score) {
            min_score = score;
            best_blocks = blocks;
        }
    }
    cerr << "min_score=" << min_score << endl;
    cerr << "min_n_cores=" << min_n_cores << endl;
    Visualize(best_blocks);
}

[[maybe_unused]] static void Solve() {
    // TODO
    // プールを用意する
    // * 完全ランダムに作成してスコア低いやつと置き換え
    // * どれか 1 つを、半分くらい core (特に小さいブロック？) を
    //   消してその後ランダムに継ぎ足し、スコア改善で置き換え
    // * どれか 1 つの state の 1/3 くらいの core に
    //   他の state 内の core を継ぎ足し

    struct Element {
        State state;
        Solution solution;
    };

    const auto population_size = 1;
    auto population = vector<Element>(population_size);
    for (auto&& [state, solution] : population) {
        while (!solution.success) {
            state.cores.clear();
            solution = state.Random(200);
        }
    }
    for (auto i = 0; i < 5e6 * 4; i++) {
        const auto r = rng.RandInt(0, population_size - 1);
        auto state = population[r].state;
        for (auto i = (int)state.cores.size() - 1; i >= 0; i--) {
            if (rng.RandInt(0, 99) < 50) {
                state.cores[i] = state.cores[state.cores.size() - 1];
                state.cores.pop_back();
            }
        }

        auto solution = state.Random(population[r].state.cores.size());
        if (!solution.success)
            continue;
        if (solution.score < population[r].solution.score) {
            population[r].state = state;
            population[r].solution = solution;
            cerr << "Improved!  i=" << i;
            cerr << "  score=" << solution.score;
            cerr << "  cores.size()=" << state.cores.size() << endl;
        }
    }

    auto min_score = 1e9;
    auto min_n_cores = 99999;
    auto argmin_score = -100;
    for (auto i = 0; i < population_size; i++) {
        if (min_score > population[i].solution.score) {
            min_score = population[i].solution.score;
            argmin_score = i;
        }
        if (min_n_cores > (int)population[i].state.cores.size()) {
            min_n_cores = (int)population[i].state.cores.size();
        }
    }

    cerr << "min_score=" << min_score << endl;
    cerr << "min_n_cores=" << min_n_cores << endl;
    Visualize(population[argmin_score].solution.blocks);
}

int main() {
    Init();
    // SolveGreedy();
    // SolveRandom();
    // SolveRandomLoop();
    Solve();
}

// core はブロックの中央であった方が良い
// core 同士の座標は離す必要がある

// TODO: 大きいのは探索範囲を絞った方が良い