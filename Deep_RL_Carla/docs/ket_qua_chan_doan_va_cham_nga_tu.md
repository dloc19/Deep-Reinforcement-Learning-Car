# Chẩn đoán va chạm còn lại của PPO — toàn bộ nằm ở nút giao

> **Vị trí đề xuất trong báo cáo**: chương Kết quả, sau phần đánh giá PPO; hoặc dùng làm phần
> định lượng cho §14 "Rủi ro thiết kế, hạn chế và hướng phát triển" của tài liệu thiết kế.
> Nguồn số liệu: `drl_training/runs/ppo_v4/probe_town04_va_cham.csv`.

## 1. Vấn đề

Sau khi hội tụ, `ppo_v4` đạt 15,8 % va chạm trung bình trên bốn bản đồ, nhưng phân bố rất
không đều:

| Bản đồ | Tỉ lệ va chạm | Thời gian trong nút giao | Lệch làn, đường thẳng (m) |
|---|---|---|---|
| Town01 | **0,0 %** | 9,0 % | 0,129 |
| Town03 | 16,7 % | 29,8 % | 0,375 |
| Town04 | **33,3 %** | 25,7 % | **0,099** |
| Town05 | 13,3 % | 26,1 % | 0,148 |

Town04 đồng thời là bản đồ có tỉ lệ va chạm **cao nhất** và độ lệch làn **thấp nhất**. Hai
điều này không thể cùng đúng nếu nguyên nhân va chạm là bám làn kém — đó là dấu hiệu cho thấy
va chạm đến từ một cơ chế khác.

## 2. Phương pháp đo

Chạy 30 episode trên Town04 với checkpoint `ppo_v4`, chế độ tất định, ghi lại trạng thái đầy
đủ ở **từng bước** và trích 15 bước cuối trước mỗi va chạm: toạ độ xe, tốc độ, giới hạn tốc
độ của đoạn đường, cờ `is_junction`, độ lệch làn, sai số hướng, và lệnh lái/ga mà chính sách
phát ra. Thu được 10 va chạm trên 30 episode (33,3 %, khớp kết quả đánh giá).

## 3. Kết quả

Trạng thái tại đúng bước va chạm, trung bình trên 10 vụ:

| Đại lượng | Giá trị |
|---|---|
| Tốc độ | 2,70 m/s (10 km/h) |
| Giới hạn tốc độ đoạn đường | 42 km/h |
| Tỉ lệ tốc độ so với giới hạn | 0,25 |
| **Đang ở trong nút giao** | **10/10 vụ (100 %)** |
| Trên đoạn cao tốc (≥ 60 km/h) | 2/10 vụ (20 %) |
| Sai số hướng | 0,925 rad (53°) |

Diễn biến 14 bước (2,8 giây) trước va chạm, cũng trung bình trên 10 vụ:

| | 14 bước trước | Lúc va chạm |
|---|---|---|
| Độ lệch làn | 0,069 m | 1,062 m (gấp **15 lần**) |
| Lệnh lái | 0,048 | 0,394 (gấp **8 lần**) |

Mô thức lặp lại **giống hệt nhau ở cả mười vụ**: xe bám làn tốt (lệch ~0,07 m), đi vào nút
giao, rồi trong vòng 6–15 bước quyết định (1,2–3 giây) độ lệch bung lên khoảng 1 m và va
chạm, trong khi chính sách phát ra lệnh lái biên độ lớn.

Giả thuyết ban đầu — chính sách nằm ngoài phân phối huấn luyện ở tốc độ cao trên đoạn cao tốc
90 km/h của Town04 — **bị bác bỏ**: va chạm xảy ra ở 10 km/h, tức một phần tư giới hạn tốc độ,
và chỉ 20 % số vụ nằm trên đoạn cao tốc.

## 4. Nguyên nhân

Nút giao **không có vạch kẻ làn**. Chính sách ra quyết định từ ảnh phân vùng ngữ nghĩa, và
trong nút giao ảnh đó không chứa thông tin xác định hướng đi. Quan trọng hơn, không gian quan
sát **không chứa** `route_command` hay `route_target_*`: các trường này để trống theo đúng
thiết kế, vì bộ lập tuyến A* (`router_plan/Global_Route_Planner.py`) chưa được cài đặt (§1.3).

Nghĩa là ở nút giao, chính sách **không có bất kỳ tín hiệu đầu vào nào cho biết phải rẽ hướng
nào**, nên nó chọn một hướng tuỳ theo đặc trưng ảnh và thường chọn sai. Đây không phải hạn chế
của thuật toán PPO, của siêu tham số, hay của ngân sách huấn luyện — mà là **thông tin thiếu ở
đầu vào**.

Nhận định này được củng cố bởi ba quan sát độc lập:

1. Huấn luyện thêm không giúp: `ppo_v4` bão hoà từ khoảng bước 150 000 (tỉ lệ va chạm 17,9 %
   rồi 20,7 % ở hai phần năm cuối), và `ppo_v5` chạy tiếp thêm 60 000 bước cho kết quả đánh
   giá không tốt hơn (18,3 % so với 15,8 %).
2. Town01, bản đồ có tỉ lệ thời gian trong nút giao thấp nhất (9,0 %), đạt **0,0 %** va chạm.
3. Trên đoạn đường thẳng, `ppo_v4` bám làn tốt hơn cả chính sách IL khởi tạo (0,126 m so với
   0,142 m, đo trên cùng ba bản đồ).

## 5. Hệ quả đối với phạm vi đồ án

Tên đề tài và §1.3 giới hạn phạm vi ở **bám làn**, không bao gồm điều hướng theo tuyến. Kết
quả trên cho thấy hai năng lực này tách bạch rõ ràng trong số liệu: chính sách bám làn tốt hơn
chuyên gia IL trên đoạn đường thẳng, và toàn bộ thất bại còn lại nằm ở phần **ngoài phạm vi**.

Do đó báo cáo nên trình bày:

- **Chỉ số chính**: độ lệch làn trên đoạn đường thẳng (loại nút giao), là thước đo trực tiếp
  của năng lực nằm trong phạm vi.
- **Tỉ lệ va chạm**: báo cáo kèm phân tích ở mục này, nêu rõ 100 % va chạm xảy ra trong nút
  giao — nơi bài toán chưa được cấp thông tin để giải.
- **Giải pháp đã có và đã kiểm chứng**: ghép bộ lập tuyến A* với chính sách học được, để A*
  lo nút giao và chính sách lo bám làn. Xem §6 — chế độ này đưa tỉ lệ hoàn thành tuyến trên
  Town03 từ 10 % lên 100 %.
- **Hướng phát triển sâu hơn**: bổ sung `route_target_local_x/y` và one-hot `route_command`
  vào `ObservationContract` và `CarlaLaneKeepEnv._build_state`, rồi **huấn luyện lại IL** với
  các cột này trước khi khởi tạo nóng cho DRL — giữ nguyên tắc IL và DRL luôn dùng chung một
  hợp đồng quan sát. Khi đó chính sách tự đi hết tuyến mà không cần bàn giao. Đây là một vòng
  pipeline đầy đủ (thu thập dữ liệu → IL → DRL), không phải một lần huấn luyện lại PPO.

## 6. Giải pháp đã kiểm chứng: ghép A* với chính sách học được

Trong khi chờ mở rộng hợp đồng quan sát, có một giải pháp kiến trúc dùng được ngay: **tách
hoạch định topology khỏi điều khiển ngang**. Bộ lập tuyến A* biết phải rẽ hướng nào; chính
sách biết bám làn. Chế độ `ROUTE_DRL_AUTOPILOT`
(`CarlaDashBoard/bridge_server/bridge/modes/route_learned_autopilot.py`) chia việc theo đúng
thế mạnh từng bên:

| Tình huống | Bộ điều khiển |
|---|---|
| Đường thường | Chính sách PPO đã học |
| Ngã tư rẽ, đổi làn (`LEFT`/`RIGHT`/`CHANGELANE*`) | Pure-pursuit bám tuyến A* |

Lệnh `STRAIGHT` **không** nằm trong danh sách bàn giao — đi thẳng qua ngã tư chính là việc
chính sách làm tốt, giao cho pure-pursuit chỉ làm mất độ mượt.

### 6.1 Thiết kế thí nghiệm

Ba chế độ chạy trên **đúng cùng một danh sách 10 tuyến A\*** trên Town03 (cùng điểm xuất
phát, cùng đích, cùng thứ tự), dài trung bình 150 m. Town03 được chọn vì có tỉ lệ waypoint
nằm trong nút giao cao nhất trong năm bản đồ (44,2 %). So sánh ghép cặp mạnh hơn đối chiếu
với một lô đánh giá độc lập: nếu mỗi lần chạy một tuyến ngẫu nhiên khác nhau thì chênh lệch
đo được sẽ lẫn với chênh lệch tuyến dễ/khó.

### 6.2 Kết quả

| Chế độ | n | Va chạm | Đến đích | Tiến độ tuyến | Lệch làn (m) | Planner cầm lái |
|---|---|---|---|---|---|---|
| `policy` (PPO thuần) | 10 | 10,0 % | **10,0 %** | 42,7 % | 0,498 | 0,0 % |
| `astar` (pure-pursuit thuần) | 10 | 0,0 % | 100,0 % | 98,5 % | 0,041 | 99,7 % |
| **`hybrid` (Route + DRL)** | 10 | **0,0 %** | **100,0 %** | **98,5 %** | 0,122 | 44,4 % |

Kiểm định McNemar ghép cặp giữa `policy` và `hybrid`:

| | Chỉ policy thành công | Chỉ hybrid thành công | p |
|---|---|---|---|
| Đến đích | 0 | 9 | **0,0039** |
| Va chạm | 1 | 0 | 1,0000 |

### 6.3 Nhận xét

**Việc ghép A\* có hiệu quả quyết định.** PPO thuần đi hết được 1/10 tuyến, trung bình chỉ đạt
42,7 % quãng đường; chế độ ghép đi hết 10/10. Khác biệt có ý nghĩa thống kê (p = 0,0039) dù
cỡ mẫu chỉ 10, vì kết quả gần như phân ly hoàn toàn.

**Chính sách học được vẫn đảm nhiệm phần lớn quãng đường.** Bộ lập tuyến chỉ cầm lái 44,4 %
số bước; hơn một nửa còn lại do chính sách tự bám làn, ở mức lệch 0,122 m — so với 0,498 m
khi chính sách phải tự xoay xở cả nút giao. Chế độ ghép vì vậy không phải là "tắt DRL đi cho
bộ điều khiển hình học lái".

**Ba lưu ý khi diễn giải:**

1. Độ lệch làn 0,041 m của `astar` **không** chứng minh pure-pursuit bám làn tốt hơn: nó bám
   thẳng waypoint tâm làn, tức được chấm điểm trên chính đường nó đi theo — một phép đo tự
   quy chiếu. Cặp so sánh có nghĩa là hybrid 0,122 m với policy 0,498 m.
2. Tỉ lệ va chạm chưa kết luận được (p = 1,0) vì chính sách thất bại chủ yếu bằng **lệch làn**
   chứ không phải va chạm: 9/10 lần đi lạc khỏi đường rồi bị dừng, chỉ 1 lần va chạm.
3. Tuyến xuất phát từ spawn 42 cho thấy chính sách **thỉnh thoảng** qua được nút giao — nó đi
   hết tuyến dù chế độ ghép trên cùng tuyến đó phải bàn giao 30,5 % số bước. Phát biểu đúng
   là "chính sách không đáng tin cậy ở nút giao", không phải "luôn thất bại ở nút giao".

Số liệu từng tuyến: `drl_training/runs/best/route_town03.csv`. Sinh lại bằng:

```powershell
python evaluate_route.py --algorithm ppo --config ppo_config_v4.json ^
    --resume runs/best/ppo_latest.pt --town Town03 --trials 10 ^
    --modes policy,astar,hybrid --out runs/best/route_town03.csv
```
