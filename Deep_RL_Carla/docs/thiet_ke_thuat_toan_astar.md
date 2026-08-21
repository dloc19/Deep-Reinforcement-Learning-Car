# Thiết kế Thuật toán A* cho Điều hướng Toàn cục trong CARLA

> Tài liệu thiết kế (không phải hướng dẫn vận hành — xem `manual_dieu_huong_astar.md` cho quy
> trình cài đặt/chạy). Mục tiêu: trình bày **cơ sở lý luận** đằng sau cách biểu diễn bản đồ
> thành đồ thị tìm kiếm, thuật toán A* dùng để lập kế hoạch toàn cục, và cách kết quả A* được
> chuyển thành tín hiệu điều khiển — để đưa trực tiếp vào chương "Phương pháp" / "Thiết kế hệ
> thống" của báo cáo. Mã nguồn tương ứng: `router_plan/graph_builder.py`, `router_plan/astar.py`,
> `router_plan/route_tracker.py`, `router_plan/controller.py`, hợp đồng dữ liệu gốc tại
> `data_collection/ASTAR_SCHEMA.md`.

---

## Mục lục

1. [Bài toán tìm đường trong bản đồ CARLA](#1-bài-toán-tìm-đường-trong-bản-đồ-carla)
2. [Thiết kế đồ thị tìm kiếm](#2-thiết-kế-đồ-thị-tìm-kiếm)
3. [Thiết kế thuật toán A*](#3-thiết-kế-thuật-toán-a)
4. [Tính đúng đắn: heuristic admissible và consistent](#4-tính-đúng-đắn-heuristic-admissible-và-consistent)
5. [Thiết kế route tracker](#5-thiết-kế-route-tracker)
6. [Thiết kế controller demo (pure-pursuit + PID)](#6-thiết-kế-controller-demo-pure-pursuit--pid)
7. [Phương án thay thế đã cân nhắc](#7-phương-án-thay-thế-đã-cân-nhắc)
8. [Đánh giá chất lượng mã nguồn](#8-đánh-giá-chất-lượng-mã-nguồn)
9. [Hạn chế và hướng phát triển](#9-hạn-chế-và-hướng-phát-triển)
10. [Tài liệu tham khảo](#10-tài-liệu-tham-khảo)

---

## 1. Bài toán tìm đường trong bản đồ CARLA

`router_plan/` trả lời câu hỏi **"đi đường nào để tới đích?"** (lập kế hoạch toàn cục), tách
biệt hoàn toàn khỏi câu hỏi **"lái thế nào để không lệch làn?"** (điều khiển cục bộ — việc của
`behavior_cloning/`/`drl_training/`). Đây là nguyên tắc kiến trúc lấy từ
`data_collection/ASTAR_SCHEMA.md` ("Pipeline separates global planning from local vehicle
control"), được giữ nguyên vẹn khi triển khai `router_plan/`.

Bài toán được hình thức hoá thành tìm đường ngắn nhất trên một đồ thị có hướng, có trọng số:

| Thành phần | Định nghĩa | Ghi chú thiết kế |
|---|---|---|
| **Đỉnh (node)** | Mỗi waypoint CARLA trên lane `Driving`, lấy theo lưới đều `resolution_m` (mặc định 2 m) | Rời rạc hoá liên tục — CARLA không cho một đồ thị hữu hạn sẵn có, phải tự sinh từ `world_map.generate_waypoints()` |
| **Cạnh (edge)** | 4 loại: `LANE_FOLLOW`, `JUNCTION_BRANCH`, `LANE_CHANGE_LEFT`, `LANE_CHANGE_RIGHT` | Đúng 4 loại chuyển động hợp lệ một xe có thể thực hiện giữa hai waypoint liền kề, không hơn không kém |
| **Chi phí cạnh** | Khoảng cách Euclid, nhân hệ số phạt cho 2 loại đổi làn | Ưu tiên đi thẳng trong lane hơn đổi làn không cần thiết, mà vẫn cho phép đổi làn khi *cần* để tới đích |
| **Điểm bắt đầu/đích** | Vị trí xe hiện tại / điểm người dùng chọn, chiếu (`project_to_road`) vào node gần nhất | Người dùng chọn toạ độ world tuỳ ý, không nhất thiết trùng một node có sẵn trong đồ thị |
| **Đầu ra** | Danh sách `node_id` có thứ tự từ start đến goal | Đầu vào cho `route_tracker` — không phải quỹ đạo liên tục, cần tracker nội suy thành điểm mục tiêu tức thời |

Ba câu hỏi thiết kế cốt lõi mà các mục sau trả lời: **(a)** biểu diễn đồ thị thế nào cho vừa
đủ chi tiết vừa xây nhanh, **(b)** thuật toán tìm đường nào cho lời giải tối ưu mà không cần
duyệt toàn bộ đồ thị, **(c)** chuyển một danh sách node rời rạc thành tín hiệu lái mượt ra sao.

---

## 2. Thiết kế đồ thị tìm kiếm

Cài đặt tại `graph_builder.RouteGraph.build()`.

### 2.1 Sinh đỉnh

`world_map.generate_waypoints(resolution_m)` trả về waypoint cách đều nhau `resolution_m` mét
dọc mọi lane của map — chỉ giữ lại lane `Driving` (loại bỏ `Sidewalk`/`Shoulder`/`Parking`...
không xe nào được phép đi vào). `node_id` dùng chung hàm `waypoint_id()` với
`data_collection/carla_collector/map_export.py`, đảm bảo hai công cụ (thu thập dữ liệu và định
tuyến) "nhìn" cùng một khái niệm node cho cùng một map.

**Đánh đổi `resolution_m`**: giá trị nhỏ hơn → route bám sát tâm lane hơn (đặc biệt ở khúc
cua) nhưng đồ thị nhiều đỉnh hơn, build chậm hơn (mặc định 2.0 m là điểm cân bằng đã chọn, xem
mục 8 — chưa đo thực nghiệm định lượng, chỉ là lựa chọn hợp lý theo kích thước lane tiêu chuẩn
~3.5 m).

### 2.2 Sinh cạnh — phân loại 4 nhóm chuyển động

Với mỗi node, dò các hướng đi hợp lệ bằng chính CARLA waypoint API (không tự suy luận hình học
từ toạ độ thô — tận dụng OpenDRIVE topology mà CARLA đã parse sẵn):

```python
successors = wp.next(resolution_m)
is_branch = wp.is_junction or len(successors) > 1
```

- **`LANE_FOLLOW`**: `len(successors) == 1` và không trong junction — đi thẳng dọc lane, không
  có lựa chọn nào khác.
- **`JUNCTION_BRANCH`**: `wp.is_junction` là `True`, hoặc `wp.next()` trả về nhiều hơn một
  waypoint (rẽ nhánh cả trong và ngoài junction hình học — một số map CARLA có điểm rẽ nhánh
  nằm ngoài vùng `is_junction` chính thức). Dùng **hoặc** của hai điều kiện thay vì chỉ một, để
  không bỏ sót nhánh nào.
- **`LANE_CHANGE_LEFT`/`RIGHT`**: chỉ tạo cạnh khi **cả hai** điều kiện đúng:
  1. `wp.lane_change` (thuộc tính OpenDRIVE) cho phép đổi làn theo hướng đó (`"left"`,
     `"right"`, hoặc `"both"`).
  2. `wp.lane_id * neighbor.lane_id > 0` — lane cạnh **cùng dấu** `lane_id`, tức cùng chiều đi.
     Không kiểm tra điều kiện này sẽ tạo cạnh sai sang lane ngược chiều bất cứ khi nào
     `get_left_lane()`/`get_right_lane()` trả về một lane hợp lệ về mặt hình học nhưng đối
     nghịch về chiều lưu thông — một lỗi nghiêm trọng nếu bỏ sót (route sẽ đưa xe sang làn
     ngược chiều).

### 2.3 Hàm chi phí — vì sao đổi làn bị phạt nhưng rẽ ở giao lộ thì không

```
cost(LANE_FOLLOW)       = distance_m
cost(JUNCTION_BRANCH)   = distance_m
cost(LANE_CHANGE_*)     = distance_m × lane_change_cost   (mặc định × 3.0)
```

Rẽ tại giao lộ (`JUNCTION_BRANCH`) **không** bị phạt thêm — quãng đường thực tế đi qua khúc cua
đã dài hơn đi thẳng một cách tự nhiên (theo hình học road network), nên không cần hệ số nhân
nhân tạo để A* "không thích" rẽ; A* tự động chọn rẽ khi đó là đường ngắn nhất tới đích và
không có lựa chọn nào khác.

Đổi làn (`LANE_CHANGE_*`) thì ngược lại: về mặt hình học hai node ở hai lane liền kề, cùng `s`
gần bằng nhau, có khoảng cách Euclid **rất ngắn** (chỉ bằng bề rộng lane, ~3.5 m) — nếu không
phạt, A* sẽ có xu hướng đổi làn liên tục bất cứ khi nào lợi được vài centimet trên một route
dài, tạo route lắt léo phi thực tế. Hệ số `lane_change_cost=3.0` làm chi phí một lần đổi làn
tương đương khoảng 10.5 m đi thẳng — đủ lớn để A* chỉ đổi làn khi thực sự cần (ví dụ lane hiện
tại không dẫn tới đích), không lớn tới mức chặn hoàn toàn các trường hợp đổi làn hợp lý.

> Hệ số này chỉ ảnh hưởng đến **thứ tự ưu tiên** A* chọn, không ảnh hưởng đến **tính khả thi**
> (nếu tồn tại đường đi, A* vẫn tìm ra dù `lane_change_cost` là bao nhiêu, vì mọi cost đều
> dương hữu hạn) — ghi rõ trong `docs/manual_dieu_huong_astar.md` mục xử lý sự cố.

### 2.4 Chiếu (snap) một vị trí bất kỳ vào node gần nhất

`_nearest_node_id()` cần cho 3 việc: chiếu vị trí xe hiện tại, chiếu điểm đích người dùng chọn,
và nối một waypoint kết quả từ `.next()`/`.get_left_lane()`/`.get_right_lane()` (không nhất
thiết trùng đúng lưới `resolution_m`) về node gần nhất đã có trong đồ thị.

Cách làm: mỗi lane `(road_id, section_id, lane_id)` giữ một danh sách `(s, node_id)` đã **sắp
xếp theo `s`** (toạ độ dọc lane trong OpenDRIVE) trong `_lane_index`, sau đó dùng
`bisect.bisect_left` để tìm vị trí chèn rồi so sánh 2-3 ứng viên lân cận — O(log n) thay vì
duyệt tuyến tính toàn bộ node của lane. Quan trọng hơn tốc độ: **giới hạn tìm kiếm trong đúng
lane `(road_id, section_id, lane_id)`** loại trừ khả năng snap nhầm sang node của một lane khác
chỉ vì nó tình cờ gần hơn về khoảng cách Euclid tuyệt đối (ví dụ hai lane song song ở hai chiều
ngược nhau, cách nhau vài mét) — một lỗi mà so sánh khoảng cách Euclid thuần trên toàn đồ thị sẽ
mắc phải.

### 2.5 Độ phức tạp xây dựng

Với `N` node: sinh node `O(N)`, build `_lane_index` + sort mỗi lane `O(N log N)` tổng, và với
mỗi node dò tối đa 4 cạnh (1-vài successor + trái + phải), mỗi lần snap `O(log n_lane)` →
tổng thời gian build xấp xỉ `O(N log N)`. Trên một Town CARLA đầy đủ (~8000–9000 node ở
`resolution_m=2.0`), việc này mất vài giây — chấp nhận được vì build chỉ chạy một lần mỗi phiên
(xem mục 9 về việc build lại mỗi lần chạy `drive_to_goal.py`).

---

## 3. Thiết kế thuật toán A*

Cài đặt tại `astar.find_path()`, đúng công thức trong `ASTAR_SCHEMA.md`:

```
g(next) = g(current) + edge.cost_m
h(node) = EuclideanDistance(node, goal)
f(node) = g(node) + h(node)
```

### 3.1 Vì sao A* (không phải Dijkstra hay BFS)

| Thuật toán | Đảm bảo tối ưu | Tốc độ thực tế trên đồ thị này | Vì sao (không) chọn |
|---|---|---|---|
| BFS/DFS thuần | Không (không tính đến `cost_m`, đặc biệt phạt đổi làn) | Nhanh | Loại ngay — route sẽ đổi làn tự do vì BFS chỉ đếm số cạnh, không phân biệt cạnh rẻ/đắt |
| Dijkstra (`h(node)=0`) | Có | Chậm hơn A* — duyệt đều mọi hướng quanh `start`, không có "định hướng" về phía goal | Đúng nhưng lãng phí: trên một map rộng, route thường chỉ cần đi theo một hành lang hẹp về phía đích, Dijkstra vẫn mở rộng nhiều node ra mọi hướng không liên quan |
| Greedy best-first (`f=h`, bỏ `g`) | Không | Nhanh nhất nhưng có thể chọn nhầm nhánh cụt gần đích rồi phải quay lại | Loại — không có gì đảm bảo route rẻ nhất, có thể cho ra đường đi dài hơn thực tế cần thiết dù "trông" như đang tiến gần đích |
| **A* (`f=g+h`, `h` admissible+consistent)** | **Có** (chứng minh ở mục 4) | Cân bằng — dùng `h` để "định hướng" tìm kiếm về phía goal, giữ nguyên tính tối ưu của Dijkstra | **Chọn** — tối ưu như Dijkstra, nhanh hơn nhờ heuristic, và đủ đơn giản để cài đặt đúng bằng stdlib thuần |

### 3.2 Cài đặt

- **Cấu trúc dữ liệu**: min-heap (`heapq`) theo `f_score`, `g_score` là `dict[node_id] -> float`,
  `came_from` là `dict[node_id] -> node_id` để truy vết đường đi khi tìm thấy goal.
- **Xử lý stale heap entry**: `heapq` không hỗ trợ giảm khoá (decrease-key) trực tiếp — khi tìm
  thấy đường rẻ hơn tới một node đã có trong heap, cài đặt **đẩy thêm một bản ghi mới** thay vì
  sửa bản ghi cũ, và dùng tập `open_set` để bỏ qua bản ghi cũ (`if current not in open_set:
  continue`) khi nó bị pop ra sau. Đây là kỹ thuật chuẩn cho A*/Dijkstra dùng binary heap
  không hỗ trợ decrease-key (tương đương cách CPython `heapq` docs khuyến nghị) — đơn giản hơn
  cài một heap có decrease-key, đổi lại heap có thể chứa nhiều bản ghi trùng `node_id`
  (không ảnh hưởng tính đúng đắn, chỉ tốn thêm bộ nhớ/thời gian hằng số).
- **`closed` set**: khi một node đã pop ra khỏi heap và không phải bản ghi stale, nó được thêm
  vào `closed` và **không bao giờ xử lý lại** — hợp lệ *vì* heuristic consistent (chứng minh ở
  mục 4); nếu heuristic không consistent, cách làm này có thể bỏ lỡ một đường rẻ hơn phát hiện
  sau khi node đã đóng.
- **Trả về `None` thay vì raise**: khi `open_heap` cạn mà chưa tới `goal_id`, hàm trả `None` —
  để tầng gọi (`GlobalRoutePlanner.plan()`) quyết định cách báo lỗi (`RouteNotFoundError` với
  thông điệp tiếng Việt gợi ý khắc phục), giữ `astar.py` là một hàm tìm kiếm thuần tuý, không
  gắn với ngữ cảnh CARLA hay cách xử lý lỗi của ứng dụng gọi nó.
- **Độ phức tạp**: `O((V + E) log V)` với binary heap tiêu chuẩn — `V`, `E` là số đỉnh/cạnh của
  đồ thị đã build ở mục 2 (không phải toàn bộ map, vì chỉ các node/cạnh trong bán kính A* thực
  sự mở rộng mới được đẩy vào heap).
- **Không phụ thuộc CARLA**: `astar.py` chỉ import `heapq`/`math`, thao tác trên `graph.nodes`/
  `graph.edges` (dữ liệu Python thuần đã trích xuất) — tách biệt hoàn toàn khỏi việc đồ thị đó
  đến từ CARLA hay từ đâu khác, thuận lợi cho việc viết unit test không cần server CARLA chạy
  (xem mục 9, hiện **chưa có** test nào — hạn chế cần bổ sung).

---

## 4. Tính đúng đắn: heuristic admissible và consistent

A* chỉ đảm bảo tìm ra đường đi **chi phí nhỏ nhất** nếu heuristic `h` thoả hai tính chất sau.
Đây là phần chứng minh áp dụng cho lựa chọn cụ thể `h(node) = EuclideanDistance(node, goal)` và
hàm chi phí `cost_m = distance_m × multiplier` (`multiplier ≥ 1`, xem mục 2.3) của module này.

### 4.1 Admissible (không bao giờ đánh giá quá cao chi phí thật còn lại)

Cần chứng minh: với mọi node `n`, `h(n) ≤ cost thật rẻ nhất từ n tới goal`.

Với bất kỳ đường đi nào từ `n` tới `goal` gồm các cạnh `e_1, …, e_k`:

```
cost(đường đi) = Σ distance_m(e_i) × multiplier(e_i) ≥ Σ distance_m(e_i)   (vì multiplier ≥ 1)
               ≥ EuclideanDistance(n, goal)                                (bất đẳng thức tam giác,
                                                                              tổng các đoạn thẳng nối
                                                                              n→…→goal luôn ≥ đường
                                                                              thẳng nối trực tiếp)
               = h(n)
```

Vậy `h(n) ≤ cost(đường đi)` cho **mọi** đường đi, tức cũng đúng cho đường rẻ nhất → `h` admissible.

### 4.2 Consistent (thoả bất đẳng thức tam giác giữa heuristic và chi phí cạnh)

Cần chứng minh: với mọi cạnh `(u → v)`, `h(u) ≤ cost_m(u, v) + h(v)`.

`h` là khoảng cách Euclid tới một điểm cố định (`goal`), nên tự nó thoả bất đẳng thức tam giác:

```
h(u) = EuclideanDistance(u, goal) ≤ EuclideanDistance(u, v) + EuclideanDistance(v, goal)
     = distance_m(u, v) + h(v)
     ≤ cost_m(u, v) + h(v)                         (vì cost_m(u,v) = distance_m(u,v) × multiplier ≥ distance_m(u,v))
```

Vậy `h(u) ≤ cost_m(u, v) + h(v)` cho mọi cạnh → `h` consistent.

### 4.3 Hệ quả cho cài đặt `astar.py`

Consistent kéo theo admissible (không cần chứng minh 4.1 riêng để có 4.2, nhưng cả hai được nêu
tách bạch ở đây vì mỗi tính chất bảo vệ một phần khác nhau của cài đặt), và quan trọng hơn:
**consistent là điều kiện đủ để việc không bao giờ xử lý lại một node đã đóng (`closed` set,
mục 3.2) không làm mất tính tối ưu** — khi A* pop một node ra khỏi heap lần đầu (không phải bản
ghi stale), `g_score` của nó tại thời điểm đó **đã là giá trị tối ưu**, không có đường nào phát
hiện sau này có thể rẻ hơn. Đây chính là lý do cài đặt trong `astar.py` an toàn khi thêm node
vào `closed` ngay khi pop, không cần cơ chế "reopen node đã đóng" mà một số cài đặt A* cho
heuristic không đảm bảo consistent phải có.

---

## 5. Thiết kế route tracker

Cài đặt tại `route_tracker.RouteTracker`. A* trả về một **danh sách node rời rạc**, không phải
tín hiệu điều khiển — `RouteTracker` là lớp chuyển đổi, chạy mỗi tick, đúng vai trò được mô tả
trong `ASTAR_SCHEMA.md` ("Planner output contract").

### 5.1 Theo dõi mục tiêu hiện tại (`target_index`)

`target_index` chỉ **tăng dần, không bao giờ giảm** — mỗi tick, nếu xe đã vào bán kính
`tolerance_m` của node đang nhắm, tăng `target_index` lên node kế tiếp (vòng `while`, có thể
nhảy qua nhiều node liền nếu chúng gần nhau hơn `tolerance_m`, ví dụ ở tốc độ cao/tick hiếm).
Tính đơn điệu này quan trọng: nếu cho phép `target_index` lùi lại khi xe tạm thời trôi ra xa
node hiện tại (nhiễu bám lane), controller (mục 6) sẽ nhận tín hiệu mục tiêu dao động qua lại,
dẫn tới lái giật cục — đây là lý do route tracker **không** dùng "node gần xe nhất" làm mục
tiêu (cách trực quan nhưng sai), mà dùng "node tiếp theo chưa đạt tới" theo đúng thứ tự route.

### 5.2 `route_progress_m` / `route_remaining_m` — quãng đường dọc route, không phải khoảng cách thẳng

Tiền xử lý một lần khi tạo tracker: `_cum_dist[i]` = tổng độ dài các đoạn từ node 0 đến node
`i` dọc route đã tìm (không phải khoảng cách Euclid tới đích). `route_remaining_m =
total_m − progress_m` — đúng yêu cầu trong `ASTAR_SCHEMA.md` ("Remaining planned path length,
**not** Euclidean distance"), vì khoảng cách thẳng tới đích luôn đánh giá thấp quãng đường
thật khi route có khúc cua, gây hiểu lầm khi hiển thị tiến độ cho người dùng.

### 5.3 Suy ra `route_command` — quy ước dấu góc rẽ

```python
if edge_type == "JUNCTION_BRANCH":
    if yaw_delta_deg > TURN_YAW_THRESHOLD_DEG:   # +20°
        return "RIGHT"
    if yaw_delta_deg < -TURN_YAW_THRESHOLD_DEG:  # -20°
        return "LEFT"
    return "STRAIGHT"
```

Quy ước dấu (**yaw tăng = rẽ phải**) không phải lựa chọn tuỳ ý — nó được suy trực tiếp từ quy
ước "phải dương" đã dùng xuyên suốt repo trong `carla_collector/geometry.py::world_to_ego`
(`lane_offset_m > 0` nghĩa là lệch về bên phải). Chứng minh ngắn gọn: tại `yaw=0`, vector
`forward = (cos 0, sin 0) = (1, 0)` và vector `right = (-sin 0, cos 0) = (0, 1)` (theo định
nghĩa `world_to_ego`). Khi yaw tăng một lượng nhỏ `ε`, `forward` dịch về phía
`(cos ε, sin ε) ≈ (1, ε)` — tức dịch theo hướng `+Y`, **cùng hướng** với vector `right` tại
`yaw=0`. Vậy yaw tăng ⇔ mũi xe xoay về phía "phải" ⇔ `RIGHT`. Giữ nhất quán quy ước này giữa
`route_tracker.py` và `geometry.py` loại bỏ khả năng hai module "hiểu" trái/phải theo hai chiều
ngược nhau — một lớp lỗi khó phát hiện bằng mắt (mọi số liệu vẫn "chạy", chỉ label bị đảo).

Ngưỡng `TURN_YAW_THRESHOLD_DEG = 20°` là một hằng số tách biệt, dễ tinh chỉnh (đã ghi trong
`manual_dieu_huong_astar.md` mục xử lý sự cố) — không hard-code trực tiếp trong biểu thức so
sánh, vì đây là điểm nhiều khả năng cần điều chỉnh theo thực nghiệm (bán kính bo cua khác nhau
giữa các Town có thể cho `yaw_delta_deg` khác nhau ở cùng một "cảm giác" rẽ trái/phải).

Riêng `LANE_CHANGE_LEFT`/`RIGHT` ánh xạ thẳng sang `CHANGELANELEFT`/`CHANGELANERIGHT` — không
cần suy luận từ `yaw_delta_deg` vì loại cạnh (`edge_type`, lấy từ `astar.path_edge_types()`) đã
xác định rõ ràng, không mập mờ như trường hợp junction.

---

## 6. Thiết kế controller demo (pure-pursuit + PID)

Cài đặt tại `controller.RoutePurePursuitController`. **Vai trò**: kiểm chứng graph/A*/route
tracker chạy đúng bằng cách lái hết một route thật trong CARLA, **không phải** kết quả nghiên
cứu chính của đồ án (đó là `behavior_cloning/` + `drl_training/`, xem
`docs/manual_dieu_huong_astar.md` mục 1 cho lý do đầy đủ). Controller này cố tình **tự viết,
không dùng mạng nơ-ron, không dùng `agents.navigation` có sẵn của CARLA** (thư mục
`PythonAPI/carla/agents/`, tách biệt khỏi gói `carla` cài qua `.egg`/`.whl` dùng trong toàn
repo — không đảm bảo nằm trên `sys.path`).

### 6.1 Lái (lateral) — pure-pursuit

- **Điểm nhắm động (lookahead point)**: đi bộ dọc route từ `target_index` của `RouteTracker`
  cho tới khi tích luỹ đủ `lookahead_m = max(min_lookahead_m, lookahead_gain × speed_mps)` —
  lookahead tỉ lệ thuận với tốc độ, để điểm nhắm không "giật" ở tốc độ thấp (node ngay sau quá
  gần) và không quá gần ở tốc độ cao (thiếu thời gian phản ứng).
- **Công thức độ cong pure-pursuit chuẩn**: `kappa = 2y / L²`, với `y` là toạ độ ngang (hệ xe,
  quy ước phải dương — khớp `world_to_ego`) của điểm nhắm, `L` là khoảng cách tới điểm đó.
- **Suy góc lái qua mô hình xe đạp (bicycle model)**: `steer_rad = atan(kappa × wheelbase_m)`,
  `wheelbase_m=2.85` (giá trị mặc định gần đúng cho xe hạng sedan như blueprint mặc định
  `vehicle.lincoln.mkz2017`), rồi chuẩn hoá về `[-1, 1]` theo `max_steer_deg=70°`.
- **Controller đọc `target_index` trực tiếp từ `RouteTracker`** (không tự duy trì một chỉ số
  route riêng) — đảm bảo lái và theo dõi tiến độ luôn thống nhất về "node tiếp theo" là node
  nào, loại bỏ khả năng hai module lệch pha với nhau.

### 6.2 Ga/phanh (longitudinal) — PID

PID kinh điển (`kp=0.35, ki=0.05, kd=0.05`) bám theo `target_speed_kmh` (mặc định 30 km/h, tự
động giảm nếu vượt giới hạn tốc độ CARLA tại vị trí hiện tại: `min(target_speed_kmh,
speed_limit_kmh)`). Đầu ra PID dương → `throttle`, âm → `brake` (tách qua `max(0, x)`/
`max(0, -x)`), không cho throttle và brake cùng khác 0 trong cùng một tick.

### 6.3 Ranh giới rõ ràng với phần nghiên cứu chính

Bảng đối chiếu để tránh nhầm lẫn khi viết báo cáo:

| | `controller.py` (module này) | `behavior_cloning/` + `drl_training/` |
|---|---|---|
| Đầu vào | Vị trí/vận tốc xe (state đầy đủ từ CARLA) + toạ độ node route | Ảnh segmentation + vector trạng thái đã chuẩn hoá (không có toạ độ tuyệt đối) |
| Cách quyết định | Công thức hình học tường minh (pure-pursuit + PID), không có tham số học được | Mạng nơ-ron đã học từ dữ liệu (IL) rồi fine-tune (DRL) |
| Vai trò trong đồ án | Công cụ kiểm chứng/demo A*, baseline "điều hướng cổ điển" để so sánh khi báo cáo | Đối tượng nghiên cứu chính |
| Có "biết" route A* không | Có (đây chính là mục đích tồn tại) | **Chưa** — quan sát hiện tại chỉ có tín hiệu bám làn, chưa có `route_target_local_x/y`/`route_command` (xem `router_plan/README.md` bước 4, việc chưa làm) |

---

## 7. Phương án thay thế đã cân nhắc

| Quyết định đã chọn | Phương án khác đã cân nhắc | Vì sao không chọn |
|---|---|---|
| Build đồ thị **live** từ `world.get_map()` mỗi lần chạy | Export sẵn `map_nodes.csv`/`map_edges.csv` (đã có sẵn công cụ ở `data_collection/carla_collector/map_export.py`) rồi đọc lại | Cần một bước export riêng cho từng map, dễ lệch nếu map/version CARLA thay đổi mà quên export lại; build live chậm hơn vài giây nhưng luôn đúng — đánh đổi chấp nhận được vì `router_plan/` chỉ chạy tương tác, không chạy hàng loạt |
| A* với `h` = khoảng cách Euclid (admissible + consistent, mục 4) | Dijkstra thuần (`h=0`) | Dijkstra vẫn đúng nhưng chậm hơn trên đồ thị rộng — không có lý do đánh đổi tốc độ lấy gì cả khi A* cho cùng độ chính xác |
| | Heuristic có trọng số (`w·h`, `w>1`, "weighted A*") để tìm nhanh hơn, chấp nhận lời giải gần tối ưu | Không cần — quy mô đồ thị (~10⁴ node/Town) đã đủ nhỏ để A* thuần chạy dưới 1 giây cho một truy vấn `find_path()`; đánh đổi độ chính xác lấy tốc độ không có lợi ích rõ ràng ở quy mô này |
| Tự viết `controller.py` (pure-pursuit + PID) | Dùng `agents.navigation.BasicAgent`/`LocalPlanner` có sẵn trong CARLA PythonAPI | Sống ở thư mục `PythonAPI/carla/agents/`, tách biệt khỏi gói `carla` cài qua `.egg`/`.whl` dùng chung toàn repo — không đảm bảo có trên `sys.path` của môi trường đã cài đặt theo `docs/manual_thu_thap_du_lieu.md`; viết từ đầu loại bỏ hẳn phụ thuộc này |
| Phạt đổi làn bằng hệ số nhân chi phí cố định (`lane_change_cost=3.0`) | Cấm hoàn toàn đổi làn trừ khi bắt buộc (loại bỏ cạnh `LANE_CHANGE_*` khỏi đồ thị, chỉ thêm lại khi A* thất bại) | Phức tạp hoá luồng gọi (cần thử lại A* hai lần), trong khi một hệ số phạt đơn giản đã đủ tạo đúng hành vi ưu tiên mong muốn mà không cần logic hai giai đoạn |
| `route_tracker`/`goal_selection` tái dùng `world_to_ego`/`normalize_angle`/convention `--goal-spawn-index` từ `data_collection/` | Viết lại độc lập trong `router_plan/` | Hai cài đặt độc lập của cùng một phép biến đổi toạ độ có nguy cơ lệch nhau theo thời gian khi một bên được sửa mà bên kia không — tái dùng trực tiếp loại bỏ rủi ro này (đánh đổi: `router_plan/` phụ thuộc `data_collection/` nằm cùng cấp thư mục, ghi rõ trong lỗi `ImportError` nếu thiếu) |

---

## 8. Đánh giá chất lượng mã nguồn

*(Tổng hợp từ rà soát mã nguồn `router_plan/graph_builder.py`, `astar.py`, `route_tracker.py`,
`controller.py`, `Global_Route_Planner.py`, `goal_selection.py`.)*

### 8.1 Điểm mạnh

- **`astar.py` không phụ thuộc CARLA** — chỉ dùng `heapq`/`math` trên dữ liệu Python thuần, nên
  logic tìm đường có thể unit-test độc lập với một đồ thị giả lập nhỏ, không cần server CARLA
  chạy (dù hiện **chưa có** test nào viết ra, xem mục 9).
- **Xử lý đúng vấn đề kinh điển của A*/Dijkstra dùng binary heap không decrease-key** (stale
  heap entries, mục 3.2) — một lỗi rất dễ mắc (heap chứa bản ghi cũ vẫn được xử lý, ghi đè
  `came_from` sai) nếu không có kiểm tra `open_set`.
- **Snap vị trí giới hạn đúng trong lane `(road_id, section_id, lane_id)`** (mục 2.4) — tránh
  lớp lỗi "snap nhầm sang lane khác gần hơn về khoảng cách tuyệt đối" mà một cách làm ngây thơ
  (so khoảng cách Euclid trên toàn đồ thị) sẽ mắc phải, đặc biệt nguy hiểm ở lane ngược chiều
  song song sát nhau.
- **Kiểm tra `lane_id` cùng dấu trước khi tạo cạnh đổi làn** (mục 2.2) — loại trừ khả năng route
  đưa xe sang lane ngược chiều, một lỗi an toàn nghiêm trọng nếu bỏ sót.
- **Tái sử dụng triệt để hạ tầng đã có** (`waypoint_id`, `world_to_ego`, `normalize_angle`,
  `location_distance` từ `data_collection/carla_collector/geometry.py`; convention CLI chọn
  đích từ `carla_collector/config.py`) — không có cài đặt lại trùng lặp nào của cùng một phép
  toán hình học, giảm hẳn nguy cơ hai nơi "hiểu" cùng một khái niệm theo hai cách khác nhau.
- **Ranh giới module rõ, mỗi lớp một trách nhiệm** (đồ thị / tìm đường / theo dõi tiến độ /
  điều khiển), giao tiếp qua dữ liệu đơn giản (`node_id` list, dict các trường `route_*`) —
  không có phụ thuộc vòng, dễ thay thế từng lớp độc lập (ví dụ thay `controller.py` bằng policy
  IL/DRL ở bước tích hợp tương lai mà không đụng tới 3 lớp còn lại).
- **Fail rõ ràng thay vì âm thầm sai**: `find_path()` trả `None` khi không có đường (không phải
  trả một route rỗng/một phần), `GlobalRoutePlanner.plan()` nâng cấp thành
  `RouteNotFoundError` có thông điệp tiếng Việt gợi ý khắc phục; `RouteGraph.__init__` validate
  `resolution_m > 0`/`lane_change_cost >= 1` ngay khi khởi tạo thay vì lỗi khó hiểu về sau.

### 8.2 Điểm cần lưu ý / rủi ro tiềm ẩn

| Vị trí | Quan sát | Mức độ | Đề xuất |
|---|---|---|---|
| `route_tracker.py::TURN_YAW_THRESHOLD_DEG` | Ngưỡng cố định 20°, không co giãn theo bán kính cua thực tế của từng junction | Thấp — đã ghi nhận là điểm cần tinh chỉnh trong `manual_dieu_huong_astar.md` | Theo dõi nhãn `route_command` qua `--output` CSV trên vài Town khác nhau trước khi chốt số cho báo cáo |
| Toàn bộ `router_plan/` | Chưa có unit test nào cho `astar.py`/`graph_builder.py`, dù `astar.py` không phụ thuộc CARLA nên chi phí viết test rất thấp | Trung bình | Thêm test tối thiểu: đồ thị nhỏ dựng tay (5–10 node) kiểm tra `find_path()` chọn đúng đường rẻ nhất khi có nhiều lựa chọn, và trả `None` đúng khi đồ thị không liên thông |
| `graph_builder.py::build()` | Build lại toàn bộ đồ thị mỗi lần gọi `drive_to_goal.py`, không cache giữa các lần chạy trên cùng map | Thấp (đã ghi nhận trong README/manual là đánh đổi có chủ đích) | Nếu cần chạy nhiều route liên tiếp, dùng `GlobalRoutePlanner` trực tiếp trong một script/notebook thay vì gọi lại CLI nhiều lần (đã có hướng dẫn ở `manual_dieu_huong_astar.md` mục 9) |
| `controller.py` | Hằng số PID (`kp/ki/kd`) và pure-pursuit (`lookahead_gain`, `min_lookahead_m`) là giá trị khởi tạo hợp lý theo lý luận thiết kế, chưa tinh chỉnh thực nghiệm trên nhiều loại xe/tốc độ | Thấp (controller không phải kết quả nghiên cứu chính, xem mục 6.3) | Không cần đầu tư nhiều — chỉ cần đủ ổn định để kiểm chứng route, đã có hướng dẫn tinh chỉnh nhanh trong `manual_dieu_huong_astar.md` mục xử lý sự cố |

Không phát hiện lỗi logic (correctness bug) trong công thức A*, cơ chế snap node, hay suy luận
`route_command` khi rà soát thủ công so với `ASTAR_SCHEMA.md` và các chứng minh hình học ở mục
4–5 của tài liệu này — các điểm ở bảng trên là hạn chế thiết kế/kiểm thử cần bổ sung, không phải
lỗi lập trình.

---

## 9. Hạn chế và hướng phát triển

- **Chưa có unit test** cho `astar.py`/`graph_builder.py`/`route_tracker.py` — ưu tiên bổ sung
  đầu tiên vì `astar.py` đặc biệt rẻ để test (không cần CARLA, xem mục 8.2).
- **Route chưa được đưa vào observation contract của IL/DRL** — `drl_training/` hiện chỉ biết
  bám làn, chưa biết rẽ theo lộ trình A*. Kế hoạch tích hợp đã chốt nhưng chưa triển khai, xem
  `router_plan/README.md` mục "Các bước triển khai", bước 4.
- **1 CARLA instance / 1 xe** — không hỗ trợ nhiều xe chạy nhiều route song song trong cùng một
  script; mở rộng tự nhiên là spawn nhiều `vehicle`/`RouteTracker` trong một vòng lặp tick
  chung, tương tự cách `carla_lane_keep_env.py` cấu trúc state cục bộ không dùng biến toàn cục.
- **Đồ thị build lại mỗi lần chạy**, không cache giữa các phiên — chấp nhận được ở quy mô hiện
  tại (vài giây/Town), nhưng nếu mở rộng sang batch nhiều route/nhiều map, nên thêm một lớp
  cache theo tên map (ví dụ pickle `RouteGraph` đã build, invalidate khi map thay đổi).
- **Ngưỡng `TURN_YAW_THRESHOLD_DEG` chưa kiểm chứng trên nhiều Town** — cần dữ liệu thực nghiệm
  (CSV `--output` từ vài lần chạy trên các Town khác nhau) trước khi coi là số liệu cuối cùng
  cho báo cáo.

---

## 10. Tài liệu tham khảo

- Hart, P. E., Nilsson, N. J., Raphael, B. (1968). *A Formal Basis for the Heuristic
  Determination of Minimum Cost Paths.* IEEE Transactions on Systems Science and Cybernetics —
  nguồn gốc thuật toán A* và định nghĩa admissible/consistent heuristic dùng trong chứng minh ở
  mục 4.
- Dijkstra, E. W. (1959). *A Note on Two Problems in Connexion with Graphs.* Numerische
  Mathematik — trường hợp đặc biệt `h=0` đối chiếu ở mục 3.1.
- Coulter, R. C. (1992). *Implementation of the Pure Pursuit Path Tracking Algorithm.* Carnegie
  Mellon University, Technical Report CMU-RI-TR-92-01 — công thức độ cong `kappa = 2y/L²` dùng
  trong `controller.py` (mục 6.1).
- `data_collection/ASTAR_SCHEMA.md` — hợp đồng dữ liệu gốc của đồ án: định nghĩa node/edge, công
  thức A*, và các trường `route_*` mà `route_tracker.py` phải sinh ra.
- `router_plan/README.md` — quyết định thiết kế đã chốt và roadmap tích hợp vào IL/DRL.

---

*Tài liệu thiết kế, biên soạn dựa trên rà soát mã nguồn `router_plan/` (`graph_builder.py`,
`astar.py`, `route_tracker.py`, `controller.py`, `Global_Route_Planner.py`, `goal_selection.py`)
và `data_collection/ASTAR_SCHEMA.md` — 2026-08-14.*
