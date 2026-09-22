# Kịch bản kiểm thử và tỉ lệ hoàn thành tuyến đường

> Sinh tự động bởi `drl_training/make_route_report.py` từ các file `eval_*.csv` có sẵn
> trong `runs/` — **không có số liệu nào nhập tay**. Chạy lại script sau mỗi lần đánh
> giá mới để bảng và hình luôn khớp nhau.

## 1. Định nghĩa chỉ số

Nhiệm vụ đánh giá là **bám làn liên tục trong một khoảng thời gian cố định**, không phải
chạy từ điểm A tới điểm B. Vì vậy "hoàn thành tuyến đường" được định nghĩa như sau:

| Chỉ số | Định nghĩa vận hành | Cột nguồn trong CSV |
|---|---|---|
| **Hoàn thành tuyến** | Xe đi hết 500 bước điều khiển (≈ 100 s mô phỏng, `action_repeat` = 4 @ 20 FPS) mà **không va chạm** và **không bị dừng sớm do rời làn** | `terminate_reason == time_limit` |
| **Tiến độ tuyến** | Số bước đi được / 500 bước, tính cho *mọi* episode kể cả episode thất bại | `length` |
| **Thất bại do va chạm** | Cảm biến va chạm kích hoạt, episode kết thúc ngay | `terminate_reason == collision` |
| **Thất bại do rời làn** | Lệch khỏi tâm làn quá ngưỡng liên tiếp `off_lane_patience_steps` = 10 bước | `terminate_reason == off_lane` |
| **Độ lệch làn** | Trung bình trị tuyệt đối độ lệch tâm làn, **chỉ tính ngoài nút giao** | `mean_abs_lane_offset_road` |
| **KTC 95 %** | Khoảng tin cậy Wilson cho tỉ lệ nhị phân (dùng Wilson vì n = 30 và tỉ lệ hay chạm 0 % hoặc 100 %) | tính từ số episode |

## 2. Bảng kịch bản kiểm thử

Mỗi kịch bản là một bản đồ CARLA. Trong mỗi kịch bản, **điểm xuất phát được bốc ngẫu
nhiên từ toàn bộ danh sách spawn point của bản đồ** (`_rng.shuffle`, seed = 42) — mỗi
episode một điểm khác nhau, nên 30 episode phủ 30 vị trí xuất phát khác nhau.

| Mã | Bản đồ | Vai trò | Đặc trưng địa hình | Độ khó | Số episode/mô hình | Điểm xuất phát | Thời tiết | Chính sách |
|---|---|---|---|---|---|---|---|---|
| KB-01 | Town01 | Huấn luyện | Thị trấn nhỏ, đường hai làn, ngã ba chữ T, không có đường cao tốc | Thấp | 10 / 30 | Ngẫu nhiên, mỗi episode một điểm | ClearNoon (mac dinh) | Tất định |
| KB-02 | Town02 | Huấn luyện | Thị trấn nhỏ và gọn, đường hai làn, ngã ba chữ T và ngã tư vuông góc | Thấp | 30 | Ngẫu nhiên, mỗi episode một điểm | ClearNoon (mac dinh) | Tất định |
| KB-03 | Town03 | Huấn luyện | Đô thị lớn: vòng xuyến, hầm chui, ngã tư nhiều làn, đường dốc | Cao | 30 | Ngẫu nhiên, mỗi episode một điểm | ClearNoon (mac dinh) | Tất định |
| KB-04 | Town04 | Huấn luyện | Vòng lặp cao tốc quanh núi và thị trấn, nhiều làn, nhánh nhập/tách | Trung bình | 10 / 30 | Ngẫu nhiên, mỗi episode một điểm | ClearNoon (mac dinh) | Tất định |
| KB-05 | Town05 | Giữ lại (held-out) | Lưới ô vuông, đường bốn làn hai chiều, cầu vượt | Trung bình | 10 / 30 | Ngẫu nhiên, mỗi episode một điểm | ClearNoon (mac dinh) | Tất định |

**Điều kiện chung cho mọi ô:** xe `vehicle.lincoln.mkz2017`; quan sát là ảnh phân vùng
ngữ nghĩa 240×192 cộng vector trạng thái; `action_repeat` = 4; giới hạn 500 bước/episode;
chạy ở chế độ tất định (`--deterministic`: lấy kỳ vọng của phân phối hành động, không
lấy mẫu ngẫu nhiên); không có phương tiện hay người đi bộ nền.

## 3. Tỉ lệ hoàn thành tuyến đường

| Kịch bản | Bản đồ | Thời tiết | Mô hình | n | Hoàn thành | Tỉ lệ hoàn thành | KTC 95 % | Tiến độ TB | Va chạm | Rời làn | Lệch làn (m) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| KB-01 | Town01 | ClearNoon (mac dinh) | PPO@6k (moc som) | 10 | 3 | **30.0 %** | 10.8 – 60.3 % | 57.6 % | 70.0 % | 0.0 % | 0.138 |
| KB-01 | Town01 | ClearNoon (mac dinh) | PPO-v4 (trien khai) | 30 | 30 | **100.0 %** | 88.6 – 100.0 % | 100.0 % | 0.0 % | 0.0 % | 0.129 |
| KB-01 | Town01 | ClearNoon (mac dinh) | PPO-v5 (doi chung) | 30 | 30 | **100.0 %** | 88.6 – 100.0 % | 100.0 % | 0.0 % | 0.0 % | 0.115 |
| KB-01 | Town01 | ClearNoon (mac dinh) | SAC-b (nhanh so sanh) | 30 | 5 | **16.7 %** | 7.3 – 33.6 % | 49.4 % | 83.3 % | 0.0 % | 0.145 |
| KB-02 | Town02 | ClearNoon (mac dinh) | PPO-v4 (trien khai) | 30 | 28 | **93.3 %** | 78.7 – 98.2 % | 94.4 % | 6.7 % | 0.0 % | 0.191 |
| KB-02 | Town02 | ClearNoon (mac dinh) | SAC-b (nhanh so sanh) | 30 | 0 | **0.0 %** | 0.0 – 11.4 % | 21.7 % | 100.0 % | 0.0 % | 0.231 |
| KB-03 | Town03 | ClearNoon (mac dinh) | PPO-v4 (trien khai) | 30 | 4 | **13.3 %** | 5.3 – 29.7 % | 46.7 % | 16.7 % | 70.0 % | 0.375 |
| KB-03 | Town03 | ClearNoon (mac dinh) | PPO-v5 (doi chung) | 30 | 7 | **23.3 %** | 11.8 – 40.9 % | 52.2 % | 13.3 % | 63.3 % | 0.413 |
| KB-03 | Town03 | ClearNoon (mac dinh) | SAC-b (nhanh so sanh) | 30 | 11 | **36.7 %** | 21.9 – 54.5 % | 62.1 % | 43.3 % | 20.0 % | 0.187 |
| KB-04 | Town04 | ClearNoon (mac dinh) | PPO@6k (moc som) | 10 | 7 | **70.0 %** | 39.7 – 89.2 % | 89.4 % | 20.0 % | 10.0 % | 0.098 |
| KB-04 | Town04 | ClearNoon (mac dinh) | PPO-v4 (trien khai) | 30 | 20 | **66.7 %** | 48.8 – 80.8 % | 76.8 % | 33.3 % | 0.0 % | 0.099 |
| KB-04 | Town04 | ClearNoon (mac dinh) | PPO-v5 (doi chung) | 30 | 19 | **63.3 %** | 45.5 – 78.1 % | 75.0 % | 36.7 % | 0.0 % | 0.102 |
| KB-04 | Town04 | ClearNoon (mac dinh) | SAC-b (nhanh so sanh) | 30 | 19 | **63.3 %** | 45.5 – 78.1 % | 82.1 % | 36.7 % | 0.0 % | 0.081 |
| KB-05 | Town05 | ClearNoon (mac dinh) | PPO@6k (moc som) | 10 | 3 | **30.0 %** | 10.8 – 60.3 % | 75.1 % | 40.0 % | 30.0 % | 0.189 |
| KB-05 | Town05 | ClearNoon (mac dinh) | PPO-v4 (trien khai) | 30 | 25 | **83.3 %** | 66.4 – 92.7 % | 89.1 % | 13.3 % | 3.3 % | 0.148 |
| KB-05 | Town05 | ClearNoon (mac dinh) | PPO-v5 (doi chung) | 30 | 22 | **73.3 %** | 55.6 – 85.8 % | 84.3 % | 23.3 % | 3.3 % | 0.163 |
| KB-05 | Town05 | ClearNoon (mac dinh) | SAC-b (nhanh so sanh) | 30 | 10 | **33.3 %** | 19.2 – 51.2 % | 66.2 % | 43.3 % | 23.3 % | 0.191 |

### Gộp bốn kịch bản (gộp episode, không lấy trung bình của trung bình)

| Mô hình | Thời tiết đã gộp | Tổng episode | Hoàn thành | Tỉ lệ hoàn thành | KTC 95 % | Va chạm | Rời làn |
|---|---|---|---|---|---|---|---|
| PPO@6k (moc som) | ClearNoon (mac dinh) | 30 | 13 | **43.3 %** | 27.4 – 60.8 % | 43.3 % | 13.3 % |
| PPO-v4 (trien khai) | ClearNoon (mac dinh) | 150 | 107 | **71.3 %** | 63.6 – 78.0 % | 14.0 % | 14.7 % |
| PPO-v5 (doi chung) | ClearNoon (mac dinh) | 120 | 78 | **65.0 %** | 56.1 – 72.9 % | 18.3 % | 16.7 % |
| SAC-b (nhanh so sanh) | ClearNoon (mac dinh) | 150 | 45 | **30.0 %** | 23.2 – 37.8 % | 61.3 % | 8.7 % |

*(Mốc sớm PPO@6k chỉ chạy trên ba bản đồ nên dòng gộp của nó không so trực tiếp được
với ba dòng còn lại; đưa vào để thấy mức xuất phát trước khi chuẩn hoá hàm thưởng.)*

## 4. Hình minh hoạ

| Hình | File | Nội dung |
|---|---|---|
| Hình A | `report_figures/10_ti_le_hoan_thanh.png` | Tỉ lệ hoàn thành theo kịch bản × mô hình, kèm KTC Wilson |
| Hình B | `report_figures/11_ly_do_ket_thuc.png` | Phân rã kết cục episode của mô hình triển khai PPO-v4 |
| Hình C | `report_figures/12_tien_do_tuyen.png` | Phân bố tiến độ tuyến của từng episode |

## 5. Giới hạn của bộ số liệu

Cần ghi rõ trong báo cáo, nếu không bảng sẽ bị hiểu sai:

1. **Chưa quét thời tiết.** Toàn bộ episode ở trên chạy với thời tiết mặc định của bản
   đồ (ClearNoon), nên cột thời tiết hiện chỉ có một giá trị. Đầu vào của mô hình là
   ảnh phân vùng ngữ nghĩa — trên nguyên tắc bất biến với thời tiết — nhưng đó là
   *giả định*, và `drl_training_sac/check_weather_invariance.py` được viết ra chính là
   để kiểm chứng nó. Muốn bảng có nhiều thời tiết thì chạy `evaluate.py` với
   `--weather <preset>`; script này tự nhận các lô mới và thêm dòng vào bảng.
2. **Nhiệm vụ là bám làn theo thời lượng, không phải đi theo lộ trình A→B.** "Hoàn thành
   tuyến" ở đây nghĩa là sống sót hết khung thời gian, không phải tới đích do A* hoạch định.
3. **Không có phương tiện nền.** Tỉ lệ va chạm đo được là va chạm với hạ tầng tĩnh (lề,
   dải phân cách, cột), không phải va chạm giữa các xe.
4. **Một seed duy nhất (42).** Chưa lặp lại với nhiều seed nên chưa tách được phương sai
   do khởi tạo khỏi phương sai do bản đồ.
