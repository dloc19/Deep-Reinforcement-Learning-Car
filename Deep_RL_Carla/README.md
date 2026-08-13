# Deep_RL_Carla

Nghiên cứu ứng dụng học tăng cường sâu (DRL) cho bài toán điều khiển bám làn xe tự hành mô
phỏng trên CARLA 0.9.10. Pipeline gồm 3 bước, mỗi bước có thư mục riêng và chạy ở môi trường
riêng (2 bước đầu trên Kaggle, bước 3 chạy local với CARLA server sống):

```
1) data_collection/    Thu thập dữ liệu passive-client từ automatic_control.py  → states.csv + seg_label/*.png
        │
        ▼
2) behavior_cloning/   [Kaggle] Segmentation (bước 2a) rồi Imitation Learning (bước 2b)
        │                       → best_carla_segmentation.pth, best_il_model.pth
        ▼
3) drl_training/        [Local, cần CARLA server] Warm-start actor từ best_il_model.pth,
                         fine-tune bằng PPO hoặc SAC → policy DRL cuối cùng
```

Tài liệu tham chiếu xuyên suốt cả 3 bước: `docs/csv_fields_by_task.md` (hợp đồng
observation/action/reward theo từng bài toán IL/DRL/A*) — **đọc trước khi sửa bất kỳ bước
nào**, vì IL và DRL bắt buộc dùng chung một observation contract để warm-start hoạt động đúng.

## Cấu trúc thư mục

| Thư mục | Vai trò | Chạy ở đâu |
|---|---|---|
| `data_collection/` | Passive client gắn vào xe `hero` (từ `automatic_control.py`), ghi `states.csv` + ảnh segmentation/RGB theo frame | Local (cùng máy CARLA server), Python 3.7 |
| `behavior_cloning/` | 2 notebook: segmentation (DeepLabV3+) rồi Imitation Learning (CNN+MLP → `[steer, longitudinal]`) | Kaggle (GPU) |
| `data_analysis/` | `split_csv.py` — tách `states.csv` thành `il_fields.csv`/`drl_fields.csv`/`astar_fields.csv` để soi riêng từng bài toán | Local, tuỳ chọn |
| `router_plan/` | A* global planner — **chưa triển khai** (`Global_Route_Planner.py` trống); xem `router_plan/README.md` cho kế hoạch chi tiết đã chốt | — |
| `drl_training/` | Fine-tune PPO hoặc SAC, warm-start actor từ checkpoint IL | Local, cần CARLA server sống, Python 3.7 |
| `docs/` | Manual thu thập dữ liệu, manual CARLA gốc, hợp đồng observation/action/reward theo từng bài toán | — |

## Quy ước đặt tên & một lưu ý khi mở lại phiên làm việc

Tất cả thư mục dùng `snake_case`, không dấu cách — tránh phải quote đường dẫn trong lệnh
shell/Python và khớp với cách `manual_thu_thap_du_lieu.md` vốn đã gọi tên thư mục
(`Data_Collection`). Nếu bạn thấy một thư mục `Behavior cloning` (có dấu cách, rỗng) còn sót
lại cạnh `behavior_cloning/` — đó là thư mục cũ trước khi đổi tên, bị Windows khoá (có
tiến trình nào đó — Explorer, IDE, Jupyter — đang mở nó) nên không xoá được tự động lúc dọn
dẹp. Đóng chương trình đang mở thư mục đó rồi xoá tay là an toàn (thư mục đã rỗng).

## Bắt đầu từ đâu

- Thu thập dữ liệu mới: `docs/manual_thu_thap_du_lieu.md`.
- Train segmentation/IL: mở 2 notebook trong `behavior_cloning/` trên Kaggle theo đúng thứ tự
  (segmentation trước, IL sau — IL cần checkpoint segmentation nếu bật
  `USE_PREDICTED_SEGMENTATION`). Tải `best_il_model.pth` về máy local sau khi train xong.
- Fine-tune DRL: `drl_training/README.md` — chọn PPO (mặc định, ít tốn RAM/VRAM hơn) hoặc SAC
  (mẫu hiệu quả hơn, cần replay buffer).
