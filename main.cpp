#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <tuple>
#include <type_traits>
#include <vector>

#ifdef _MSC_VER
#include <intrin.h>
#endif
#ifdef __GNUC__
#include <x86intrin.h>
#endif

#ifdef __clang__
#pragma clang attribute push(__attribute__((target("arch=skylake"))),          \
                             apply_to = function)
#elif defined(__GNUC__)
#pragma GCC target(                                                            \
    "sse,sse2,sse3,ssse3,sse4,popcnt,abm,mmx,avx,avx2,tune=native")
#pragma GCC optimize("O3")
#endif

static auto n_candidate_edges_for_node = 150;
static auto n_new_core_candidates = 2;
static auto remove_2_ratio = 0.5;
static auto distance_exponent = 2.0;
static auto n_small_core_candidates = 2;
static auto start_temperature = 0.5;
static auto end_temperature = 0.5;
static auto annealing_param_a = 0.0;
static auto annealing_param_b = 1.0;
static auto fewer_candidates = 0;

using namespace std;

using i8 = signed char;
using u8 = unsigned char;
using ll = long long;
using ull = unsigned long long;

static inline double Time() {
    return (double)(chrono::duration_cast<chrono::nanoseconds>(
                        chrono::steady_clock::now().time_since_epoch())
                        .count()) *
           1e-9;
}

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
    double Rand() {
        const auto x = (*this)();
        return (double)x * (1.0 / (double)(1ull << 32));
    }
    static constexpr unsigned max() { return 0xffffffffu; }
    static constexpr unsigned min() { return 1u; }
};

static auto rng = Random(42);

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
    auto L1Distance(const Vec3& rhs) const {
        return abs(x - rhs.x) + abs(y - rhs.y) + abs(z - rhs.z);
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
static constexpr auto kMaxNCandidateEdgesForNode = 200;
static array<array<Stack<int, kMaxNCandidateEdgesForNode>, 14 * 14 * 14>, 2>
    candidate_edge_ids_for_each_node;

static auto mean_degree = 0.0;

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

    static const auto kDxyzs = array<Vec3, 6>{
        Vec3(0, 0, 1),  Vec3(0, 0, -1), Vec3(0, 1, 0),
        Vec3(0, -1, 0), Vec3(1, 0, 0),  Vec3(-1, 0, 0),
    };
    mean_degree = 0.0;
    for (auto i = 0; i < 2; i++) {
        auto v = Vec3();
        for (v.x = 0; v.x < D; v.x++)
            for (v.y = 0; v.y < D; v.y++)
                for (v.z = 0; v.z < D; v.z++)
                    if (coord_to_node_id[i][v.x][v.y][v.z] != -1)
                        for (const auto& dxyz : kDxyzs) {
                            const auto u = v + dxyz;
                            if (u.IsIn())
                                mean_degree +=
                                    coord_to_node_id[i][u.x][u.y][u.z] != -1;
                        }
    }
    mean_degree /= (double)nodes.size();
    cerr << "n_nodes=" << nodes.size() << endl;
    cerr << "mean_degree=" << mean_degree << endl;

#ifdef DUMP_INFO_ONLY
    exit(0);
#endif

    if (nodes.size() < 173) {
        if (mean_degree < 3.1592966666666666) {
            // a
            remove_2_ratio = 0.9218864823439933;
        } else if (mean_degree <= 3.5939) {
            // b
            remove_2_ratio = 0.8398777100272572;
        } else {
            // c
            remove_2_ratio = 0.056701157366232935;
        }
        n_new_core_candidates = 2;
        n_small_core_candidates = 2;
        start_temperature = 3.0;
        end_temperature = 3.0;
    } else if (nodes.size() <= 514) {
        if (mean_degree < 3.8762033333333337) {
            // d
            annealing_param_a = -7.181959083657732;
            annealing_param_b = 0.006935683724648756;
            end_temperature = 0.7805181254898048;
            remove_2_ratio = 0.4720546444669666;
            start_temperature = 2.2426956006292613;
        } else if (mean_degree <= 4.399126666666667) {
            // e
            annealing_param_a = -4.271183370533636;
            annealing_param_b = 1.537547480976249;
            end_temperature = 1.0563753419550566;
            remove_2_ratio = 0.9204116097148919;
            start_temperature = 2.5613498004124264;
        } else {
            // f
            annealing_param_a = 2.431448945527027;
            annealing_param_b = 0.02508515581387205;
            end_temperature = 2.6682468037906912;
            remove_2_ratio = 0.9912314538913594;
            start_temperature = 0.6473180196690247;
        }
        n_new_core_candidates = 3;
        n_small_core_candidates = 3;
    } else {
        if (mean_degree < 4.414796666666667) {
            // g
            annealing_param_a = -9.87155675256468;
            annealing_param_b = 2.1181607666471;
            end_temperature = 0.48333592195366826;
            remove_2_ratio = 0.9733052823543014;
            start_temperature = 4.979551144728833;
        } else if (mean_degree <= 4.83589) {
            // h
            annealing_param_a = -3.489694694019225;
            annealing_param_b = 2.7825041793733574;
            end_temperature = 1.2997699433076362;
            remove_2_ratio = 0.5004234409045778;
            start_temperature = 2.731243073332406;
        } else {
            // i
            annealing_param_a = -3.3550288393623484;
            annealing_param_b = 2.162164664274324;
            end_temperature = 1.1435152636621457;
            remove_2_ratio = 0.6341158338628746;
            start_temperature = 3.3780372808987322;
        }
        fewer_candidates = 1;
        n_new_core_candidates = 3;
        n_small_core_candidates = 3;
    }

    // 辺を構築
    edges.clear();
    auto tmp_edges = vector<pair<short, short>>();
    auto tmp_edge_groups = vector<pair<int, int>>();
    edge_groups.clear();
    for (const auto& p : {
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
        for (para.x = fewer_candidates >= 1 ? 0 : -D + 1; para.x < D;
             para.x++) {
            for (para.y = fewer_candidates >= 2 ? 0 : -D + 1; para.y < D;
                 para.y++) {
                for (para.z = fewer_candidates >= 3 ? 0 : -D + 1; para.z < D;
                     para.z++) {
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
                                static auto q = array<Vec3, 14 * 14 * 14>();
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
    }

    sort(tmp_edge_groups.begin(), tmp_edge_groups.end(),
         [](const pair<int, int> a, const pair<int, int> b) {
             return a.second - a.first > b.second - b.first;
         });

    // 各頂点に対して、それを含むgroupの中で、サイズが大きい上位何個かを取り出す
    auto n_candidate_groups = array<array<int, 14 * 14 * 14>, 2>();
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
            if ((int)candidate_edge_ids_for_each_node[0][e.first].size() <
                n_candidate_edges_for_node)
                candidate_edge_ids_for_each_node[0][e.first].push_back(edge_id);
            if ((int)candidate_edge_ids_for_each_node[1][e.second].size() <
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
            for (const auto dxyz : kDxyzs) {
                const auto u = v + dxyz;
                if (u.IsIn()) {
                    const auto neighbour_edge_id = visited[u.x][u.y][u.z];
                    if (neighbour_edge_id != -1) {
                        edges[edge_id].neighbours.push_back(
                            {neighbour_edge_id});
                        edges[neighbour_edge_id].neighbours.push_back(
                            {edge_id});
                    }
                }
            }
        }
    }

    cerr << "edge_groups.size()=" << edge_groups.size() << endl;
}

} // namespace info

static void Init() {
    input.Read();
    info::Init();
}

struct Solution {
    bool success;
    double score;
    array<u8, 5488> blocks;
    array<short, 200> block_sizes;
};

struct State {
    struct Core {
        int edge_id;
    };
    Stack<Core, 200> cores;

    inline auto SCPCovered() const {
        auto silhouette = info::Silhouette();
        for (const auto& core : cores)
            silhouette |=
                info::edge_groups[info::edges[core.edge_id].edge_group_id]
                    .silhouette;
        return silhouette;
    }

    inline auto BFS() const {
        struct BFSResult {
            enum struct Status {
                kSuccess,
                kFailureSilhouette,
                kFailureScoreExcess,
            };
            info::Silhouette visited_silhouette;
            array<u8, 14 * 14 * 14 * 2> visited_nodes;
            array<short, 200> block_sizes;
            double score;
            Status status;
        };
        auto res = BFSResult{};
        fill(res.visited_nodes.begin(),
             res.visited_nodes.begin() + info::nodes.size(), (u8)-1);
        struct QueueElement {
            int edge_id;
        };
        static auto qs = vector<Queue<QueueElement, 14 * 14 * 14, 0>>();
        if (qs.size() < cores.size())
            qs.resize(cores.size());
        for (auto core_id = 0; core_id < (int)cores.size(); core_id++) {
            const auto edge_id = cores[core_id].edge_id;
            const auto& edge = info::edges[edge_id];
            qs[core_id].clear();
            qs[core_id].push_back({edge_id});
            res.visited_silhouette |= edge.silhouette;
        }
        auto next_core_id = vector<u8>(cores.size());
        iota(next_core_id.begin(), next_core_id.end() - 1, 1);
        next_core_id.back() = 0;
        auto last_core_id = (u8)-1;
        for (auto core_id = (u8)0;; core_id = next_core_id[core_id]) {
            auto& q = qs[core_id];
            assert(!q.empty());
            int edge_id;
            do {
                edge_id = q.front().edge_id;
                q.pop_front();
                const auto& edge = info::edges[edge_id];
                if (res.visited_nodes[edge.node_ids[0]] == (u8)-1 &&
                    res.visited_nodes[edge.node_ids[1]] == (u8)-1)
                    goto ok;
            } while (!q.empty());
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
                static_assert(sizeof(decltype(edge.neighbours)::value_type) ==
                              4);
                static_assert(
                    sizeof(remove_reference_t<decltype(q)>::value_type) == 4);
                memcpy(&q.arr[q.right], edge.neighbours.begin(),
                       edge.neighbours.size() * 4);
                q.right += edge.neighbours.size();
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

    inline Solution Random(const int max_n_cores) {
        auto scp_not_covered = info::full_silhouette;
        scp_not_covered ^= SCPCovered();

        // SCP を解く
        auto trial = 0;
        while (!scp_not_covered.empty()) {
            if (max_n_cores <= (int)cores.size())
                return {false, 1e9, {}, {}};
            if (trial++ >= 50)
                return {false, 1e9, {}, {}};
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
            cores.push_back({edge_id});
            trial = 0;
            scp_not_covered.AndNotAssign(edge_group.silhouette);
        }
        // BFS で広げてく
        while (true) {
            // BFS
            const auto bfs_result = BFS();
            // 全部のシルエット条件を満たしたか確認
            auto non_visited_silhouette = bfs_result.visited_silhouette;
            non_visited_silhouette ^= info::full_silhouette;
            if (non_visited_silhouette.empty())
                return {true, bfs_result.score, bfs_result.visited_nodes,
                        bfs_result.block_sizes};
            // 失敗
            if (max_n_cores <= (int)cores.size())
                return {false, 1e9, {}, {}};
            // 満たしていない pixel のところに edge を追加
            // この順番もランダムにした方が良いのか？
            const auto pixel_id = non_visited_silhouette.CountRightZero();
            const auto& pixel = info::pixels[pixel_id];
            // 探索順を決める、ここもうちょっと高速化はできそう
            static array<short, 50000> order;
            iota(order.begin(), order.begin() + pixel.edge_ids.size(), 0);
            auto new_edge_id = -100;
            auto n_found_candidates = 0;
            auto min_sum_inv_distance = 1e9;
            for (auto i = (int)pixel.edge_ids.size() - 1; i >= 0; i--) {
                const auto idx_order = rng.RandInt(0, i);
                const auto new_edge_id_candidate =
                    pixel.edge_ids[order[idx_order]];
                order[idx_order] = order[i];
                if (CheckAddable(new_edge_id_candidate)) {
                    n_found_candidates++;
                    auto sum_inv_distance = 0.0;
                    for (const auto& core : cores)
                        for (auto idx_node_ids = 0; idx_node_ids < 2;
                             idx_node_ids++) {
                            auto distance =
                                info::nodes[info::edges[core.edge_id]
                                                .node_ids[idx_node_ids]]
                                    .coord.L1Distance(
                                        info::nodes
                                            [info::edges[new_edge_id_candidate]
                                                 .node_ids[idx_node_ids]]
                                                .coord);
                            sum_inv_distance += 1.0 / pow((double)(distance),
                                                          distance_exponent);
                        }

                    if (sum_inv_distance < min_sum_inv_distance) {
                        new_edge_id = new_edge_id_candidate;
                        min_sum_inv_distance = sum_inv_distance;
                    }
                    if (n_found_candidates == n_new_core_candidates)
                        goto ok2;
                }
            }
            if (n_found_candidates == 0)
                return {false, 1e9, {}, {}};
        ok2:;
            cores.push_back({new_edge_id});
        }
    }

    bool CheckAddable(const int edge_id) const {
        const auto& node_ids = info::edges[edge_id].node_ids;
        for (const auto& core : cores) {
            if (info::edges[core.edge_id].node_ids[0] == node_ids[0])
                return false;
            if (info::edges[core.edge_id].node_ids[1] == node_ids[1])
                return false;
        }
        return true;
    }
};

static void Visualize(const array<u8, 5488>& blocks) {
    auto puzzle = array<Cube<int>, 2>();
    auto n_blocks = 0;
    for (auto node_id = 0; node_id < (int)info::nodes.size(); node_id++) {
        const auto& p = info::nodes[node_id].coord;
        if (blocks[node_id] == (u8)-1)
            continue;
        puzzle[p.w][p.x][p.y][p.z] = blocks[node_id] + 1;
        n_blocks = max(n_blocks, blocks[node_id] + 1);
    }

    cout << n_blocks << endl;
    puzzle[0].Visualize();
    puzzle[1].Visualize();
}

auto t0 = Time();

static inline double Sigmoid(const double a, const double x) {
    return 1.0 / (1.0 + exp(-a * x));
}

// f: [0, 1] -> [0, 1]
static inline double MonotonicallyIncreasingFunction(const double a,
                                                     const double b,
                                                     const double x) {
    if (a == 0.0)
        return x;
    const double x_left = a > 0 ? -b - 0.5 : b - 0.5;
    const double x_right = x_left + 1.0;
    const double left = Sigmoid(a, x_left);
    const double right = Sigmoid(a, x_right);
    const double y = Sigmoid(a, x + x_left);
    return (y - left) / (right - left);
}

// f: [0, 1] -> [start, end]
static inline double MonotonicFunction(const double start, const double end,
                                       const double a, const double b,
                                       const double x) {
    return MonotonicallyIncreasingFunction(a, b, x) * (end - start) + start;
}

static inline double ComputeTemperature(const double progress) {
    return MonotonicFunction(start_temperature, end_temperature,
                             annealing_param_a, annealing_param_b, progress);
    return (1.0 - progress) * start_temperature + progress * end_temperature;
}

[[maybe_unused]] static void Solve() {
    struct Element {
        State state;
        Solution solution;
    };

    static constexpr auto kTimeLimit = 5.8;

    auto current = Element{};
    while (!current.solution.success) {
        current.state.cores.clear();
        current.solution = current.state.Random(200);
        if (Time() - t0 >= kTimeLimit)
            exit(1);
    }

    auto best_solution = current.solution;
    auto t = Time() - t0;
    for (auto trial = 0; trial < 1e9; trial++) {
        if (trial % 16 == 0 && (t = Time() - t0) >= kTimeLimit) {
            cerr << "executed_trials=" << trial << endl;
            break;
        }
        auto state = current.state;
        auto block_sizes = current.solution.block_sizes;
        auto n_removed_cores = 0;
        const auto n_remove = rng.Rand() < remove_2_ratio ? 2 : 3;
        for (auto removing_trial = 0; removing_trial < n_remove;
             removing_trial++) {
            if (state.cores.size() == 0)
                break;
            auto idx_core = -100;
            auto min_block_size = 99999;
            for (auto i = 0; i < n_small_core_candidates; i++) {
                auto idx_core_i = rng.RandInt(0, (int)state.cores.size() - 1);
                if (block_sizes[idx_core_i] < min_block_size) {
                    min_block_size = block_sizes[idx_core_i];
                    idx_core = idx_core_i;
                }
            }
            state.cores[idx_core] = state.cores[state.cores.size() - 1];
            block_sizes[idx_core] = block_sizes[state.cores.size() - 1];
            state.cores.pop_back();
            n_removed_cores++;
        }

        auto solution = state.Random(current.state.cores.size());
        if (!solution.success)
            continue;
        const auto gain = log2(current.solution.score) - log2(solution.score);
        const auto progress = t / kTimeLimit;
        const auto temperature = ComputeTemperature(progress);
        const auto acceptance_proba = exp(gain / temperature);
        if (rng.Rand() < acceptance_proba) {
            current.state = state;
            current.solution = solution;
        }
        if (solution.score < best_solution.score) {
            best_solution = solution;
        }
    }

    cerr << "min_score=" << best_solution.score << endl;
    Visualize(best_solution.blocks);
}

int main() {
    Init();
    Solve();
}

#ifdef __clang__
#pragma clang attribute pop
#endif
