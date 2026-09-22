# Tỉ lệ hoàn thành quãng đường (đi từ A đến B)

> Sinh tự động bởi `drl_training/make_route_goal_report.py` từ các file
> `drl_training/runs/best/route_*.csv` do `evaluate_route.py` ghi ra — **không có số
> liệu nào nhập tay**. Chạy lại script sau mỗi lô đánh giá mới.

## 1. Chỉ số này khác gì bảng ở `ket_qua_ti_le_hoan_thanh.md`

| | `ket_qua_ti_le_hoan_thanh.md` | Bảng này |
|---|---|---|
| Nhiệm vụ | Bám làn tự do, **không có đích** | Đi hết tuyến A* từ A đến B |
| "Hoàn thành" | Sống sót hết 500 bước, không va chạm | Xe **thực sự tới đích** (`route_completed`) |
| Script | `evaluate.py` | `evaluate_route.py` |
| Nguồn | `runs/*/eval_*.csv` | `runs/best/route_*.csv` |

Chỉ bảng này mới trả lời được câu "xe đi hết quãng đường bao nhiêu phần trăm số lần".

## 2. Ba bộ điều khiển chạy trên **cùng một bộ tuyến** (thiết kế ghép cặp)

| Chế độ | Ai cầm lái | Đo cái gì |
|---|---|---|
| `policy` | PPO lái toàn bộ, kể cả trong ngã tư | Giới hạn của **hợp đồng quan sát**: observation không có trường nào mang ý định rẽ, nên trong ngã tư phân nhánh policy không thể biết rẽ trái hay phải |
| `astar` | Pure-pursuit lái toàn bộ | Trần hình học thuần tuý, không học gì |
| `hybrid` | PPO lái đường thường, pure-pursuit lái ngã tư rẽ và đổi làn | **Đúng cấu hình đem triển khai** trên dashboard |

Tuyến dài 159 m trung bình (lọc 80–220 m), giới hạn 90 s mô phỏng mỗi tuyến, xe `vehicle.lincoln.mkz2017`, chạy tất định, không có xe nền.
Checkpoint: `runs/best/ppo_latest.pt` (PPO-v4, update 195).

## 3. Tỉ lệ hoàn thành quãng đường theo bản đồ

| Kịch bản | Bản đồ | Chế độ | n | Tới đích | Tỉ lệ hoàn thành | KTC 95 % | Tiến độ TB | Va chạm | Lệch làn (m) | A* cầm lái |
|---|---|---|---|---|---|---|---|---|---|---|
| KB-01 | Town01 | PPO thuan (DRL lai toan bo) | 20 | 18 | **90.0 %** | 69.9 – 97.2 % | 75.2 % | 0.0 % | 0.136 | 0.0 % |
| KB-01 | Town01 | A* + pure-pursuit (khong hoc) | 20 | 20 | **100.0 %** | 83.9 – 100.0 % | 98.5 % | 0.0 % | 0.066 | 99.7 % |
| KB-01 | Town01 | Route+DRL (cau hinh trien khai) | 20 | 20 | **100.0 %** | 83.9 – 100.0 % | 98.5 % | 0.0 % | 0.147 | 19.2 % |
| KB-02 | Town02 | PPO thuan (DRL lai toan bo) | 20 | 19 | **95.0 %** | 76.4 – 99.1 % | 60.9 % | 5.0 % | 0.169 | 0.0 % |
| KB-02 | Town02 | A* + pure-pursuit (khong hoc) | 20 | 20 | **100.0 %** | 83.9 – 100.0 % | 98.8 % | 0.0 % | 0.090 | 99.8 % |
| KB-02 | Town02 | Route+DRL (cau hinh trien khai) | 20 | 20 | **100.0 %** | 83.9 – 100.0 % | 98.8 % | 0.0 % | 0.159 | 27.7 % |
| KB-03 | Town03 | PPO thuan (DRL lai toan bo) | 20 | 3 | **15.0 %** | 5.2 – 36.0 % | 43.8 % | 25.0 % | 0.357 | 0.0 % |
| KB-03 | Town03 | A* + pure-pursuit (khong hoc) | 20 | 20 | **100.0 %** | 83.9 – 100.0 % | 98.7 % | 0.0 % | 0.063 | 99.7 % |
| KB-03 | Town03 | Route+DRL (cau hinh trien khai) | 20 | 18 | **90.0 %** | 69.9 – 97.2 % | 96.1 % | 10.0 % | 0.173 | 52.2 % |
| KB-04 | Town04 | PPO thuan (DRL lai toan bo) | 20 | 9 | **45.0 %** | 25.8 – 65.8 % | 46.8 % | 55.0 % | 0.121 | 0.0 % |
| KB-04 | Town04 | A* + pure-pursuit (khong hoc) | 20 | 20 | **100.0 %** | 83.9 – 100.0 % | 98.5 % | 0.0 % | 0.046 | 99.7 % |
| KB-04 | Town04 | Route+DRL (cau hinh trien khai) | 20 | 20 | **100.0 %** | 83.9 – 100.0 % | 98.5 % | 0.0 % | 0.094 | 43.9 % |
| KB-05 | Town05 | PPO thuan (DRL lai toan bo) | 20 | 16 | **80.0 %** | 58.4 – 91.9 % | 36.9 % | 20.0 % | 0.136 | 0.0 % |
| KB-05 | Town05 | A* + pure-pursuit (khong hoc) | 20 | 20 | **100.0 %** | 83.9 – 100.0 % | 98.5 % | 0.0 % | 0.071 | 99.7 % |
| KB-05 | Town05 | Route+DRL (cau hinh trien khai) | 20 | 19 | **95.0 %** | 76.4 – 99.1 % | 98.3 % | 0.0 % | 0.131 | 46.1 % |

### Gộp toàn bộ bản đồ (gộp tuyến, không lấy trung bình của trung bình)

| Chế độ | Tổng tuyến | Tới đích | Tỉ lệ hoàn thành | KTC 95 % | Va chạm | Tiến độ TB |
|---|---|---|---|---|---|---|
| PPO thuan (DRL lai toan bo) | 100 | 65 | **65.0 %** | 55.3 – 73.6 % | 21.0 % | 52.7 % |
| A* + pure-pursuit (khong hoc) | 100 | 100 | **100.0 %** | 96.3 – 100.0 % | 0.0 % | 98.6 % |
| Route+DRL (cau hinh trien khai) | 100 | 97 | **97.0 %** | 91.5 – 99.0 % | 2.0 % | 98.1 % |

### Kiểm định McNemar ghép cặp — `policy` đối chứng `hybrid`

Dùng McNemar chứ không dùng z-test hai tỉ lệ: hai nhánh chạy **đúng cùng những tuyến** nên không độc lập; chỉ các cặp bất đồng mới mang thông tin.

| Phạm vi | Chỉ `policy` tới đích | Chỉ `hybrid` tới đích | p (hai phía) | Kết luận |
|---|---|---|---|---|
| Town01 | 0 | 2 | 0.5000 | chưa đủ để kết luận |
| Town02 | 0 | 1 | 1.0000 | chưa đủ để kết luận |
| Town03 | 0 | 15 | 0.0001 | khác biệt có ý nghĩa |
| Town04 | 0 | 11 | 0.0010 | khác biệt có ý nghĩa |
| Town05 | 1 | 3 | 0.6250 | chưa đủ để kết luận |
| **Gộp** | 1 | 30 | 0.000000 | khác biệt có ý nghĩa |

## 4. Hình minh hoạ

| Hình | File | Nội dung |
|---|---|---|
| Hình D | `report_figures/15_ti_le_hoan_thanh_quang_duong.png` | Tỉ lệ tới đích theo bản đồ × bộ điều khiển, kèm KTC Wilson |
| Hình E | `report_figures/16_tien_do_quang_duong.png` | Phân bố tiến độ từng tuyến — cho thấy hỏng ở đâu, không chỉ hỏng hay không |
| Hình F | `report_figures/17_route_drl_theo_ban_do.png` | **Chỉ chế độ triển khai Route+DRL** trên Town01–Town05, kèm cột gộp |

## 5. Giới hạn

1. **Một seed chọn tuyến duy nhất (20260906).** Bộ tuyến cố định để ba chế độ ghép
   cặp được, nhưng chưa lặp lại với bộ tuyến khác.
2. **Không có xe nền.** Va chạm đo được là va chạm với hạ tầng tĩnh.
3. **Tuyến 80–220 m.** Tuyến ngắn hơn không đi qua ngã tư nào nên không đo được gì;
   tuyến dài hơn làm mỗi lượt tốn vài phút.
4. **Thời tiết mặc định của bản đồ (ClearNoon).** Chưa quét thời tiết ở bài này.
