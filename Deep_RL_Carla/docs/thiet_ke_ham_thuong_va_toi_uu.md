# Thiết kế Hàm phần thưởng và Chiến lược Tối ưu hoá cho Học tăng cường sâu trong Bài toán Bám làn (CARLA)

> Tài liệu thiết kế (không phải hướng dẫn vận hành — xem `manual_train_drl.md` cho quy trình
> chạy). Mục tiêu: trình bày **cơ sở lý luận** đằng sau hàm phần thưởng và chiến lược tối ưu
> hoá (PPO/SAC) dùng trong bước 3/3 của pipeline đồ án, để đưa trực tiếp vào chương
> "Phương pháp" / "Thiết kế hệ thống" của báo cáo. Mã nguồn tương ứng:
> `drl_training/envs/carla_lane_keep_env.py` (`_compute_reward`), `drl_training/policy/`,
> `drl_training/ppo/`, `drl_training/sac/`.

---

## Mục lục

1. [Mô hình hoá bài toán dưới dạng MDP](#1-mô-hình-hoá-bài-toán-dưới-dạng-mdp)
2. [Thiết kế hàm phần thưởng](#2-thiết-kế-hàm-phần-thưởng)
3. [Thiết kế chiến lược tối ưu hoá](#3-thiết-kế-chiến-lược-tối-ưu-hoá)
4. [Đánh giá chất lượng mã nguồn](#4-đánh-giá-chất-lượng-mã-nguồn)
5. [Hạn chế và hướng phát triển](#5-hạn-chế-và-hướng-phát-triển)
6. [Tài liệu tham khảo](#6-tài-liệu-tham-khảo)

---

## 1. Mô hình hoá bài toán dưới dạng MDP

Bài toán bám làn được hình thức hoá thành một Markov Decision Process (MDP) liên tục
`(S, A, P, r, γ)`:

| Thành phần | Định nghĩa | Ghi chú thiết kế |
|---|---|---|
| **State `s`** | Ảnh phân vùng ngữ nghĩa (`seg`, class-ID map, 4 lớp: `Background, Road, RoadLine, Sidewalk`) + vector vô hướng `[speed_mps, yaw_rate_rps, speed_limit_kmh, previous_steer, previous_longitudinal, onehot(traffic_light)]` | Chuẩn hoá z-score bằng đúng `norm_stats` đã dùng khi train IL (`policy/observation.py::ObservationContract`) — bắt buộc để warm-start không bị lệch phân phối đầu vào |
| **Action `a`** | `[steer, longitudinal] ∈ [-1,1]²`, `longitudinal≥0→throttle`, `<0→brake` | Không gian hành động liên tục, 2 chiều — khớp định dạng nhãn đã dùng ở bước Data Collection/IL |
| **Transition `P`** | Vật lý CARLA UE4, chế độ đồng bộ (`synchronous_mode=True`, `fixed_delta_seconds=1/fps`) | Đảm bảo 1 action ↔ 1 tick vật lý xác định, tái lập được — điều kiện tiên quyết để reward/log nhất quán giữa các lần chạy |
| **Reward `r`** | Hàm shaping nhiều thành phần, §2 | — |
| **`γ`** | 0.99 | Chiết khấu tiêu chuẩn cho control task horizon vài trăm–1000 bước |

Không gian quan sát **cố ý không chứa** `lane_offset_m`/`heading_error_rad` dù đây là hai đại
lượng trung tâm của reward — lý do trình bày ở §2.5. State và transition dùng chung nguyên
vẹn giữa hai thuật toán fine-tune (PPO, SAC) và giữa bước IL/DRL, để hai bước có thể so sánh
công bằng và IL checkpoint warm-start được trực tiếp (§3.1).

---

## 2. Thiết kế hàm phần thưởng

### 2.1 Nguyên tắc thiết kế

Ba nguyên tắc chi phối toàn bộ thiết kế reward:

1. **Reward dày (dense/shaped), không dùng reward thưa (sparse).** Một episode dài tới
   `max_episode_steps=1000` bước (~100s ở 10Hz). Nếu chỉ thưởng khi hoàn thành và phạt khi va
   chạm, tín hiệu học sẽ quá thưa để agent gán được công (credit assignment) cho hành động cụ
   thể nào đã dẫn tới kết quả tốt/xấu hàng trăm bước sau đó. Do đó reward được **shape** thành
   tổng nhiều thành phần liên tục, tính lại ở **mỗi bước**, phản ánh trực tiếp chất lượng lái ở
   bước đó.
2. **Tách biệt observation dùng để quyết định và đại lượng dùng để đánh giá.** `lane_offset_m`
   và `heading_error_rad` — hai đại lượng đo trực tiếp "đang lệch làn bao nhiêu" — bị loại khỏi
   vector quan sát đưa vào mạng, dù chúng là input tính reward. Nếu đưa cả vào observation,
   optimizer có thể học một chính sách suy biến kiểu gần-identity (vì bản thân input đã gần
   như "đáp án" của reward), tương tự vấn đề rò rỉ nhãn (label leakage) trong học có giám sát.
   Chính sách chỉ được quan sát ảnh + vận tốc + hành động trước đó, giống hệt những gì một mô
   hình lái thực tế (không có định vị lane-level chính xác) sẽ có.
3. **Tái sử dụng đúng các trường đã chuẩn hoá xuyên suốt pipeline.** Toàn bộ tên trường, đơn vị
   (m, m/s, rad, rad/s) khớp với `docs/csv_fields_by_task.md` (mục DRL) — cùng một định nghĩa
   `lane_offset_m`/`heading_error_rad` được dùng lại nguyên vẹn từ bước Data Collection, tránh
   3 bước của pipeline "hiểu" cùng một khái niệm theo 3 cách khác nhau.

### 2.2 Công thức đầy đủ

Cài đặt tại `CarlaLaneKeepEnv._compute_reward()`:

```
r_t =   w_speed        · clip(v_forward, 0, v_limit)
      − w_lane_offset   · |lane_offset_m|
      − w_heading       · |heading_error_rad|
      − w_steer_delta   · (Δsteer)²
      − w_long_delta    · (Δlongitudinal)²
      − w_yaw_rate      · yaw_rate²
      − 1[off_lane]     · off_lane_penalty
      − n_lane_invasion · lane_invasion_penalty
      − 1[collision]    · collision_penalty
```

với `Δsteer = steer_t − steer_{t−1}`, `Δlongitudinal = longitudinal_t − longitudinal_{t−1}`
(so với hành động **đã áp dụng** ở bước trước, không phải hành động mẫu thô), và
`v_limit = speed_limit_kmh / 3.6` (m/s, clip mẫu số dưới `1e-6` để tránh chia 0 khi API CARLA
trả `speed_limit_kmh=0`).

### 2.3 Vai trò và cơ sở lựa chọn từng thành phần

| Thành phần | Loại | Vai trò | Vì sao thiết kế như vậy |
|---|---|---|---|
| `w_speed·clip(v_forward,0,v_limit)` | Thưởng dương | Khuyến khích tiến về phía trước, đúng tốc độ giới hạn | `clip` ở cận trên = **không** thưởng thêm khi vượt tốc độ cho phép (tránh học phóng nhanh để tối đa hoá reward); `clip` ở cận dưới `0` = không phạt trực tiếp khi đứng yên/lùi, vì việc đứng yên đã gián tiếp bị các thành phần khác (đặc biệt off-lane khi xe trôi ra khỏi làn, hoặc đơn giản là mất reward dương) làm kém hấp dẫn hơn tiến đúng làn |
| `−w_lane_offset·\|lane_offset_m\|` | Phạt liên tục | Kéo xe về tâm làn | Hàm giá trị tuyệt đối (L1) thay vì bình phương (L2) — phạt tuyến tính theo độ lệch, không "tha" cho độ lệch nhỏ như L2 (đạo hàm L2 tại lệch nhỏ gần 0), giữ gradient học ổn định kể cả khi xe đã bám khá tốt |
| `−w_heading·\|heading_error_rad\|` | Phạt liên tục | Căn hướng xe song song với tâm làn | Bổ sung cho `lane_offset` — một xe có thể ở đúng tâm làn nhưng lệch hướng (sắp đi chéo ra khỏi làn ở bước sau); phạt hướng là tín hiệu "sớm" hơn phạt vị trí |
| `−w_steer_delta·(Δsteer)²` | Phạt liên tục | Phạt đánh lái đột ngột | Bình phương (không phải trị tuyệt đối) để phạt nặng các dao động lớn/giật cục nhưng gần như bỏ qua các điều chỉnh nhỏ mượt — mục tiêu là dập tắt hành vi "bang-bang" (đánh lái biên độ lớn liên tục) mà một chính sách RL mới fine-tune dễ rơi vào, đồng thời khuyến khích quỹ đạo mượt gần với phong cách lái trong dữ liệu IL |
| `−w_long_delta·(Δlongitudinal)²` | Phạt liên tục | Phạt tăng/giảm ga-phanh đột ngột | Cùng lý do với `steer_delta`, áp dụng cho trục dọc — tránh chuyển throttle↔brake giật cục |
| `−w_yaw_rate·yaw_rate²` | Phạt liên tục | Phạt tốc độ xoay thân xe lớn | Khác về **nguồn** so với `steer_delta`: `steer_delta` phạt sự thay đổi của *tín hiệu điều khiển*, còn `yaw_rate` phạt *kết quả vật lý* thực tế của xe (có thể lớn dù steer mượt, ví dụ do trượt bánh) — hai tín hiệu độc lập, bổ sung cho cùng mục tiêu "lái êm" |
| `−off_lane_penalty` (mỗi bước còn lệch làn) | Phạt sự kiện lặp | Tạo áp lực leo thang khi xe ra khỏi làn | Là phạt **hằng số cộng thêm**, độc lập với phạt liên tục theo `lane_offset_m` — một khi `\|lane_offset_m\| > half_lane_width`, mỗi bước còn ở trạng thái đó bị phạt thêm một lượng cố định, tạo động lực quay lại làn càng sớm càng tốt thay vì "chấp nhận" một mức lệch làn ổn định |
| `−lane_invasion_penalty·n` | Phạt sự kiện | Phạt vượt vạch kẻ đường | Lấy trực tiếp từ cảm biến `sensor.other.lane_invasion` của CARLA — một nguồn tín hiệu **độc lập** với phép tính hình học `lane_offset_m` (suy từ waypoint API), đóng vai trò lớp phòng vệ thứ hai: bắt được các trường hợp cắt vạch mà phép tính hình học có thể bỏ sót (ví dụ ở làn rẽ, giao lộ) |
| `−collision_penalty` | Phạt kết thúc | Phạt va chạm, kết thúc episode | Giá trị lớn nhất trong toàn bộ hàm reward (mặc định 50, gấp 10 lần `off_lane_penalty`) và luôn kèm `terminated=True` — mã hoá rõ **thứ tự ưu tiên an toàn > bám làn > tốc độ**, tránh chính sách học cách "chấp nhận" va chạm nhỏ để né một chuỗi phạt lệch làn dồn dập |

### 2.4 Thiết kế điều kiện kết thúc episode (Termination Design)

```
terminated = new_collision  OR  (off_lane liên tục ≥ off_lane_patience_steps)
truncated  = step_count ≥ max_episode_steps        (time-limit, không phải thất bại)
```

- **Cơ chế "patience" (`off_lane_patience_steps=20`, ≈2s @10Hz):** một bước đơn lẻ lệch làn
  (nhiễu cảm biến, cắt cua nhẹ hợp lý) **không** kết thúc episode ngay — bộ đếm
  `off_lane_streak` phải đạt ngưỡng liên tục mới `terminated=True`. Đây là điểm cân bằng giữa
  dung sai hợp lý và kỷ luật bám làn: đủ ngắn để không lãng phí thời gian huấn luyện trên một
  episode đã "hỏng", đủ dài để không trừng phạt các dao động nhỏ, tạm thời.
- **Time-limit KHÔNG được coi là thất bại.** `truncated=True` (hết giờ) là một sự kiện trung
  tính về mặt giá trị — khác về bản chất với `terminated=True` (thất bại thật: va chạm/lệch
  làn kéo dài). Nhầm lẫn hai loại này là lỗi kinh điển trong RL có time-limit (Pardo et al.,
  *"Time Limits in Reinforcement Learning"*, 2018): nếu coi episode bị cắt vì hết giờ như một
  trạng thái kết thúc thật (giá trị tương lai = 0), agent bị dạy sai rằng "tồn tại đến hết giờ"
  không có giá trị tiếp diễn, dẫn tới hành vi rủi ro gần cuối episode. Mỗi thuật toán xử lý
  theo cách phù hợp với cấu trúc dữ liệu của nó (chi tiết ở §3.2, §3.3):
  - **PPO**: bootstrap giá trị qua critic ngay tại thời điểm thu thập rollout — cộng thêm
    `γ·V(s_{t+1})` vào reward của bước bị cắt trước khi lưu vào buffer.
  - **SAC**: đơn giản hơn — **không lưu** transition bị time-limit-cắt vào replay buffer (vì
    buffer tối ưu bộ nhớ suy `next_obs` từ ô kế tiếp, mà sau `env.reset()` ô đó thuộc episode
    mới, không phải phần tiếp theo thật của transition này).

### 2.5 Rủi ro "reward hacking" đã lường trước

| Rủi ro | Biện pháp phòng tránh trong thiết kế |
|---|---|
| Chính sách "học tắt" từ `lane_offset_m`/`heading_error_rad` nếu chúng nằm trong observation | Loại hoàn toàn hai trường này khỏi vector quan sát (§1, §2.1) |
| Phóng tốc độ để tối đa `w_speed` term | `clip(v_forward, 0, v_limit)` — không có reward biên nào khi vượt giới hạn tốc độ |
| Dao động điều khiển tần số cao để "né" các phạt tức thời (ví dụ đánh lái rồi trả lại ngay để trung bình `lane_offset` thấp) | Phạt trực tiếp `(Δsteer)²`/`(Δlongitudinal)²`/`yaw_rate²` khiến chiến thuật dao động luôn có chi phí cao hơn lái mượt |
| "Chấp nhận" một mức lệch làn ổn định thay vì sửa | `off_lane_penalty` cộng dồn theo từng bước còn lệch (không phải một lần), cộng với việc chấm dứt episode sau `off_lane_patience_steps` — không có trạng thái "lệch làn ổn định, chi phí thấp" nào tồn tại lâu được |
| **Hạn chế còn tồn tại (chưa xử lý)**: xe **lùi hoặc đứng yên vô thời hạn** không bị phạt trực tiếp vì `clip(...,0,...)` chặn cận dưới ở 0 — reward speed-term bằng 0 chứ không âm | Cần theo dõi `episode_len`/`terminate_reason=time_limit` với `episode_reward` gần 0 trong log thực nghiệm; nếu chính sách hội tụ về hành vi "đứng yên an toàn", cân nhắc thêm phạt nhỏ cho vận tốc gần 0 kéo dài hoặc thưởng theo tiến độ quãng đường thay vì tốc độ tức thời |

### 2.6 Bảng trọng số mặc định

| Tham số | Giá trị mặc định | Đơn vị áp dụng | Ghi chú |
|---|---|---|---|
| `w_speed` | 1.0 | trên m/s | |
| `w_lane_offset` | 1.0 | trên mét lệch | Cùng độ lớn với `w_speed` — mặc định coi trọng bám làn ngang bằng tiến độ |
| `w_heading` | 0.5 | trên radian | Bằng nửa `w_lane_offset` vì `heading_error_rad` (thường < 0.3 rad khi lái ổn định) có biên độ số nhỏ hơn `lane_offset_m` cho cùng một "mức độ nghiêm trọng" cảm nhận |
| `w_steer_delta` | 1.0 | trên (đơn vị hành động)² | |
| `w_long_delta` | 0.5 | trên (đơn vị hành động)² | Thấp hơn `steer_delta` — dao động ga/phanh ít nguy hiểm hơn dao động lái |
| `w_yaw_rate` | 0.1 | trên (rad/s)² | Trọng số nhỏ nhất — vai trò regularizer phụ, không phải tín hiệu chính |
| `off_lane_penalty` | 5.0 | phạt cố định/bước | ≈ 5 lần `w_lane_offset` ở mức lệch 1m — đủ lớn để tạo áp lực leo thang rõ rệt |
| `lane_invasion_penalty` | 1.0 | phạt cố định/lần | |
| `collision_penalty` | 50.0 | phạt cố định, kèm terminate | Lớn nhất toàn hệ thống — 10× `off_lane_penalty` |

> Các giá trị này là **điểm khởi đầu hợp lý theo đúng thứ tự độ lớn** dựa trên tài liệu thiết
> kế của đồ án, **chưa phải số đã kiểm chứng thực nghiệm** trên môi trường CARLA thật (module
> chưa từng chạy với server sống tại thời điểm viết — xem `README.md`). Cần tinh chỉnh dựa trên
> log thực nghiệm trước khi đưa số liệu cuối cùng vào báo cáo.

### 2.7 Quy trình tinh chỉnh thực nghiệm (đề xuất cho chương Thực nghiệm)

Theo dõi `episode_log.csv`/`update_log.csv` sinh ra trong lúc train, và điều chỉnh theo bảng
triệu chứng → hướng xử lý sau:

| Triệu chứng quan sát được | Hướng điều chỉnh |
|---|---|
| `terminate_reason` gần như luôn là `collision`, chính sách "chấp nhận" va chạm để né phạt lệch làn dồn dập | Tăng `collision_penalty` và/hoặc giảm `off_lane_penalty` |
| Xe lái an toàn nhưng không chịu tăng tốc, `episode_reward` thấp dù không va chạm | Giảm `w_lane_offset` tương đối so với `w_speed`, hoặc tăng `w_speed` |
| Hành vi lái giật cục, `approx_kl`/`entropy` (PPO) dao động mạnh giữa các update | Tăng `w_steer_delta`/`w_long_delta` |
| Xe đứng yên/lùi kéo dài (§2.5) | Thêm phạt vận tốc-thấp-kéo-dài hoặc chuyển sang reward theo tiến độ quãng đường |

---

## 3. Thiết kế chiến lược tối ưu hoá

### 3.1 Kiến trúc mạng dùng chung và chiến lược warm-start từ IL

**Backbone dùng chung.** `PolicyBackbone` (CNN 5 lớp cho ảnh seg one-hot 4 lớp +
`AdaptiveAvgPool2d` + MLP 2 lớp cho vector vô hướng) và `build_trunk_head` (2 lớp Linear+ELU,
64→32) được **định nghĩa một lần**, tái sử dụng bởi cả 4 mạng (PPO actor/critic, SAC
actor/critic). Lý do thiết kế:

- **Nhất quán kiến trúc với IL** (`SteeringNet` trong `train_il_v9.ipynb`) đến từng tên
  layer (`conv`, `pool`, `cnn_fc`, `scalar_mlp`, `head.0`/`head.3`) — điều kiện bắt buộc để có
  thể copy tensor theo tên khoá state_dict từ checkpoint IL sang mạng DRL mà không cần huấn
  luyện lại từ đầu.
- `AdaptiveAvgPool2d((1,1))` làm backbone **bất biến với độ phân giải ảnh đầu vào** — cho phép
  môi trường DRL train (hoặc infer) ở độ phân giải khác độ phân giải IL đã train mà không lỗi
  shape khi nạp checkpoint, dù chất lượng feature có thể suy giảm nếu hạ quá thấp (xem
  `manual_train_drl.md` mục 12 về vạch kẻ làn biến mất ở độ phân giải quá thấp).

**Vì sao critic KHÔNG chia sẻ trunk với actor.** Thực hành phổ biến ở một số cài đặt PPO là
dùng chung một trunk cho cả actor và critic để tiết kiệm tính toán. Ở đây, actor và critic
dùng **hai bản sao độc lập** của backbone/trunk_head. Lý do: gradient từ value loss, nếu chảy
ngược qua một trunk dùng chung, sẽ làm trôi các đặc trưng đã được warm-start cẩn thận từ IL —
đúng điều mà warm-start đang cố gắng bảo vệ. Chi phí tính toán thêm là không đáng kể so với
một bước tick CARLA (điểm nghẽn tốc độ thật sự, §3.5).

**Warm-start từ checkpoint IL — động cơ.** Đây là mẫu hình BC-then-RL (Behaviour
Cloning warm-start rồi fine-tune bằng RL), phổ biến trong robot học/lái tự động vì hai lý do:

1. **An toàn/thời gian huấn luyện**: một chính sách RL khởi tạo ngẫu nhiên sẽ lái ngẫu nhiên
   trong hàng nghìn bước đầu — trong mô phỏng vật lý CARLA, điều này đồng nghĩa va chạm liên
   tục, lãng phí phần lớn ngân sách tương tác môi trường (vốn đắt — mỗi bước cần một tick vật
   lý + render) chỉ để học lại những gì IL đã học được từ dữ liệu chuyên gia.
2. **Sample efficiency**: fine-tune từ một chính sách đã hợp lý cho phép PPO/SAC tập trung
   ngân sách tương tác vào việc *cải thiện* hành vi (sửa lỗi hệ thống của IL, thích nghi với
   động lực học mà BC không nắm bắt được do exposure bias) thay vì học lại từ số 0.

**Cơ chế remap (`policy/il_compat.py`).** Ánh xạ theo **tên tensor** trong state_dict
(`conv.*`/`pool.*`/`cnn_fc.*`/`scalar_mlp.*` → `backbone`; `head.0`/`head.3` → `trunk_head`;
`head.6` → `mean_head`), dùng chung một bộ hàm remap cho cả PPO và SAC để tránh hai cài đặt
độc lập lệch nhau theo thời gian. Thiết kế **fail-fast**: bất kỳ khoá nào thiếu hoặc lệch shape
đều **raise lỗi ngay**, không âm thầm bỏ qua — vì bỏ qua âm thầm có nghĩa một phần mạng khởi
tạo ngẫu nhiên trong khi tưởng đã warm-start, một lỗi rất khó phát hiện bằng mắt thường
(training vẫn "chạy được", chỉ kém hiệu quả hơn, dễ bị nhầm là do siêu tham số).

**`log_std` khởi tạo nhỏ.** Cả hai actor (PPO: `log_std` cố định theo state;
SAC: `log_std_head` phụ thuộc state nhưng khởi tạo trọng số nhỏ, bias hằng) đều khởi tạo độ
lệch chuẩn hành động ở mức khiêm tốn (`log_std_init=-1.2`, σ≈0.3) thay vì mặc định lớn — giữ
rollout đầu tiên sau warm-start còn gần với hành vi IL, tránh nhiễu khám phá lớn phá vỡ ngay
lập tức chính sách vừa warm-start.

### 3.2 PPO — Thiết kế on-policy

Cài đặt theo đúng công thức PPO gốc (Schulman et al., 2017), tại `ppo/ppo_agent.py`:

- **Clipped surrogate objective**: `L = E[min(ratio·A, clip(ratio,1−ε,1+ε)·A)]` với
  `ratio = exp(log π_new(a|s) − log π_old(a|s))`, `ε=0.2` — giới hạn mức độ một update có thể
  thay đổi chính sách, quan trọng hơn bình thường ở đây vì cần **bảo vệ chính sách đã
  warm-start** khỏi một update quá hung hăng ngay từ vài bước fine-tune đầu tiên.
- **GAE(λ)** (`gae_lambda=0.95`) ước lượng advantage — cân bằng bias/variance tốt hơn Monte
  Carlo returns thuần hoặc TD(0) thuần cho horizon control task ~1000 bước.
- **Value function clipping** (`value_clip_range=0.2`), cùng công thức clip như surrogate
  objective, để ổn định việc học value.
- **Entropy bonus** mặc định **tắt** (`entropy_coef=0.0`) — vì chính sách đã warm-start từ IL
  vốn đã có phân phối hành động hợp lý (không cần ép entropy cao để khám phá không gian hành
  động từ đầu như training from scratch); có thể bật lại nếu quan sát chính sách hội tụ sớm
  vào một local behaviour kém tối ưu.
- **Gradient clipping** theo global norm (`max_grad_norm=0.5`) — chống exploding gradient,
  đặc biệt quan trọng vì loss tổng hợp cả policy loss và value loss (trọng số `value_coef=0.5`).
- **Early-stop theo approximate KL** (`target_kl=0.02`): nếu KL trung bình của một epoch vượt
  1.5× ngưỡng, dừng sớm các epoch còn lại của update đó — lớp bảo vệ **thứ hai**, độc lập với
  clip surrogate, chống việc một rollout "may mắn"/bất thường kéo chính sách đi quá xa trong
  một lần update.
- **Time-limit bootstrapping**: thực hiện **ngay tại thời điểm thu thập** rollout (trong
  `train_ppo.py`, trước khi ghi vào buffer) bằng cách cộng `γ·V(s_{t+1})` vào phần thưởng của
  bước bị time-limit cắt — cách tiếp cận chuẩn cho vấn đề nêu ở §2.4.
- **Lưu hành động thô, áp dụng hành động đã clip**: `log_prob` được tính trên mẫu Gaussian
  **chưa clip** (`raw_action`), trong khi môi trường nhận `env_action = clip(raw_action,-1,1)`.
  Đây là xấp xỉ chuẩn, được chấp nhận rộng rãi (ví dụ Stable-Baselines3) cho PPO liên tục có
  action bound — tính lại `log_prob` trên hành động đã clip ở thời điểm update sẽ làm sai lệch
  advantage gần biên hành động.

### 3.3 SAC — Thiết kế off-policy

Cài đặt theo SAC v2 (Haarnoja et al., 2018), tại `sac/sac_agent.py`, `sac/networks.py`:

- **Squashed Gaussian policy**: hành động = `tanh(N(μ(s), σ(s)))`, với `σ(s)` (log_std) là
  một **head phụ thuộc state riêng** — khác PPO (log_std cố định không phụ thuộc state), đúng
  thiết kế chuẩn của SAC. Vì hành động luôn đi qua `tanh`, log-prob cần hiệu chỉnh theo công
  thức đổi biến (`− log(1 − tanh(u)²)`, Haarnoja et al. phụ lục C).
- **Twin Q-network + lấy min** (Fujimoto et al., ý tưởng gốc từ TD3): hai mạng Q độc lập, dùng
  `min(Q1,Q2)` cho cả target lẫn policy gradient — chống thiên lệch overestimation cố hữu của
  Q-learning trong không gian hành động liên tục.
- **Automatic temperature tuning**: hệ số entropy `α` được học tự động qua gradient descent
  trên `log_alpha`, với entropy mục tiêu mặc định `= −action_dim = −2.0` — loại bỏ nhu cầu dò
  tay hệ số cân bằng exploitation/exploration, vốn là siêu tham số nhạy cảm nhất của SAC gốc
  (trước khi có auto-tuning).
- **Polyak-averaged target critic** (`τ=0.005`): target network cập nhật mượt theo
  `θ_target ← (1−τ)θ_target + τ·θ`, ổn định target cho Bellman backup thay vì copy cứng định kỳ.
- **`QNetwork` không warm-start**: mạng Q luôn khởi tạo ngẫu nhiên — nhất quán với thực hành
  BC+RL chuẩn (ví dụ AWAC), vì IL chưa từng học một hàm giá trị hành động-trạng thái (chỉ học
  ánh xạ trạng thái→hành động), không có trọng số nào để warm-start.
- **Replay buffer tối ưu bộ nhớ** (`sac/replay_buffer.py`): mỗi observation lưu **đúng một
  lần**; `next_obs` tại thời điểm sample được suy ra từ ô kế tiếp trong buffer vòng
  (circular), thay vì lưu trùng lặp `obs` và `next_obs` như cách triển khai SAC "sách giáo
  khoa". Ở độ phân giải observation 240×192 (khớp IL), cách này giảm ~2× dung lượng RAM cho phần ảnh
  segmentation — thành phần chiếm bộ nhớ áp đảo của buffer. Hệ quả kỹ thuật: **loại trừ** hoàn
  toàn transition bị time-limit-cắt khỏi buffer (§2.4) để tránh ghép sai `next_obs` với một
  episode mới không liên quan.
- **Warm-start-aware exploration**: SAC "sách giáo khoa" luôn dùng hành động **ngẫu nhiên
  thuần** trong `learning_starts` bước đầu để làm giàu buffer trước khi update. Ở đây, vì actor
  đã warm-start từ IL, hành vi ngẫu nhiên thuần sẽ (a) lãng phí chính lợi thế của warm-start,
  và (b) nhiều khả năng làm đầy buffer bằng các episode ngắn, chất lượng thấp (lái ngẫu nhiên
  va chạm nhanh trong CARLA) — không tốt cho critic học sớm. Vì vậy, khi `warm_start=True`,
  giai đoạn khởi động dùng **chính mẫu ngẫu nhiên Gaussian của actor đã warm-start**
  (`select_action(..., deterministic=False)`) thay vì `Uniform(-1,1)` — vẫn có khám phá (qua
  nhiễu Gaussian nội tại của actor) nhưng xuất phát từ một chính sách đã biết lái hợp lý. Chỉ
  quay lại random thuần khi `warm_start=False` (huấn luyện từ đầu), đúng SAC gốc.

### 3.4 So sánh PPO và SAC

| Tiêu chí | PPO (on-policy) | SAC (off-policy) |
|---|---|---|
| Bộ nhớ | Thấp — chỉ giữ 1 rollout (`n_steps=2048`) | Cao hơn nhiều — buffer sống suốt quá trình train (`buffer_capacity` mặc định 50 000) |
| Hiệu quả mẫu | Thấp hơn — mỗi transition dùng cho đúng 1 rollout rồi bỏ | Cao hơn — mỗi transition được tái sử dụng qua nhiều lần sample từ buffer |
| Độ ổn định / dễ chỉnh | Cao — 2 lớp bảo vệ (clip + KL early-stop) tự hạn chế mức thay đổi mỗi update | Nhạy hơn với `actor_lr`/`critic_lr`/`τ`, nhưng auto-tune `α` giảm bớt một chiều cần dò |
| Khi nào chọn | Máy đơn, CARLA là điểm nghẽn tốc độ, ưu tiên dễ debug/ổn định trước | Đủ RAM (máy thuê GPU), muốn tận dụng tối đa mỗi bước môi trường (vì bước CARLA đắt hơn nhiều một bước gradient) |

Cả hai cùng warm-start từ một checkpoint IL, cùng dùng chung env/reward/observation contract —
**thiết kế có chủ đích** để việc so sánh PPO/SAC trong báo cáo là một so sánh thực nghiệm công
bằng (khác biệt duy nhất là thuật toán fine-tune), phù hợp tinh thần "nghiên cứu ứng dụng học
tăng cường sâu".

### 3.5 Tối ưu hoá tài nguyên tính toán (kỹ thuật hệ thống)

Ngoài thiết kế thuật toán, một số quyết định kỹ thuật nhắm trực tiếp vào ràng buộc tài nguyên
thực tế của đồ án (máy cá nhân RAM 16GB/VRAM 4GB trước khi thuê GPU):

| Quyết định | Tác dụng | Cơ sở |
|---|---|---|
| Lưu `seg` dạng class-ID (1 byte/pixel) thay vì one-hot trong cả rollout buffer lẫn replay buffer; one-hot hoá **on-the-fly trên GPU** ngay trước lớp conv (`PolicyBackbone.forward`) | Giảm bộ nhớ buffer ảnh **4 lần** so với one-hot uint8, **16 lần** so với one-hot float32 (bảng nhãn hiện tại có 4 lớp) | One-hot chỉ cần tồn tại trong batch nhỏ đưa qua mạng, không cần tồn tại cho toàn bộ buffer |
| `ReplayBuffer` không lưu trùng `obs`/`next_obs` (§3.3) | Giảm ~2× RAM phần ảnh của SAC buffer | Mô phỏng đúng chiến lược `optimize_memory_usage=True` của Stable-Baselines3 |
| Camera render 480×384 (khớp collector) nhưng **observation hạ xuống 240×192** = đúng độ phân giải IL đã train, bằng `resize_class_map()` có bảo tồn class mỏng | Loại bỏ lệch phân phối quan sát giữa IL và DRL, đồng thời giảm 4× RAM buffer (SAC 9.2GB → 2.2GB) | CARLA render trực tiếp ở độ phân giải cấu hình nên render thẳng 240×192 sẽ xoá vạch kẻ ngay lúc rasterize; render cao rồi hạ mẫu theo **độ phủ diện tích** thì giữ được vạch. `AdaptiveAvgPool2d` khiến mọi kích thước đều chạy mà **không báo lỗi**, nên lệch độ phân giải là lỗi im lặng — phải khoá bằng cấu hình |
| `batch_size` mặc định thấp (64 cho PPO, 32 cho SAC — SAC thấp hơn vì chạy song song 2 mạng Q) | Giới hạn trực tiếp đỉnh VRAM/RAM mỗi lần update, độc lập với độ phân giải camera | Neo theo `BATCH_SIZE=32` mà notebook IL dùng ổn định ở cùng độ phân giải trên GPU 16GB |
| Khuyến nghị `--device cpu` cho máy VRAM ≤4GB | Nhường toàn bộ VRAM cho CARLA (UE4 khuyến nghị ≥6GB) | `PolicyBackbone` rất nhỏ (5 lớp conv, 24–64 kênh); điểm nghẽn thông lượng luôn là bước tick CARLA (vật lý + render), không phải các phép nhân ma trận của mạng — train trên CPU gần như không làm chậm thông lượng tổng thể |

Nguyên lý chung: **điểm nghẽn tốc độ nằm ở mô phỏng (CARLA tick), không nằm ở mạng nơ-ron** —
vì vậy mọi tối ưu hoá bộ nhớ ưu tiên phía buffer/ảnh (chi phí thật sự lớn), còn phía tính toán
mạng có thể hy sinh (chạy CPU) mà không ảnh hưởng đáng kể thông lượng huấn luyện tổng thể.

### 3.6 Bảng siêu tham số tổng hợp

| Nhóm | Tham số | PPO | SAC |
|---|---|---|---|
| Chung | `gamma` | 0.99 | 0.99 |
| Chung | `warm_start` | true | true |
| Chung | `max_grad_norm` | 0.5 | 0.5 |
| Chung | `batch_size` | 64 | 32 |
| Chung | `total_steps` | 2 000 000 | 500 000 |
| Riêng | Learning rate | `learning_rate=3e-4` (chung actor+critic) | `actor_lr=critic_lr=alpha_lr=3e-4` (tách riêng) |
| Riêng | `n_steps` / `buffer_capacity` | 2048 (rollout) | 50 000 (replay) |
| Riêng | `gae_lambda` / `tau` | 0.95 | 0.005 |
| Riêng | `clip_range` / `target_entropy` | 0.2 | tự động = −2.0 |
| Riêng | `epochs` / `learning_starts` | 10 | 5000 |
| Riêng | `target_kl` / `train_freq` | 0.02 | 1 |

---

## 4. Đánh giá chất lượng mã nguồn

*(Tổng hợp từ rà soát mã nguồn `drl_training/` — phần thay đổi đang chờ commit và toàn bộ hệ
thống reward/tối ưu hoá liên quan.)*

### 4.1 Điểm mạnh

- **Phân tách trách nhiệm rõ ràng**: môi trường (`envs/`), đặc trưng quan sát (`policy/
  observation.py`), kiến trúc mạng (`policy/backbone.py`, `policy/actor_critic.py`,
  `sac/networks.py`), thuật toán update (`ppo/ppo_agent.py`, `sac/sac_agent.py`), buffer
  (`ppo/rollout_buffer.py`, `sac/replay_buffer.py`) mỗi phần một module riêng, ranh giới rõ.
- **"Hợp đồng" quan sát tự mô tả**: `ObservationContract` đọc `continuous_cols`/`norm_stats`/…
  trực tiếp từ checkpoint IL thay vì hard-code trùng lặp hai nơi (Kaggle notebook vs. local
  script không chạy chung Python runtime) — loại bỏ hẳn một lớp lỗi lệch pha kinh điển giữa
  hai môi trường huấn luyện tách biệt.
- **Fail-fast, không silent-fail**: `checkpoint_io.py`, `il_compat.py` đều raise lỗi rõ ràng
  (kèm gợi ý khắc phục bằng tiếng Việt) khi checkpoint thiếu trường hoặc shape lệch, thay vì
  bỏ qua âm thầm — đúng nguyên tắc quan trọng với warm-start (một phần mạng "quên" warm-start
  trong im lặng là lỗi rất khó phát hiện).
- **Tái sử dụng logic dùng chung giữa PPO/SAC** (`policy/il_compat.py`, `policy/backbone.py`)
  giúp hai thuật toán không thể âm thầm lệch nhau về cách diễn giải checkpoint IL theo thời
  gian khi code được sửa.
- **Refactor gần nhất** (di chuyển `build_vehicle_state`/`traffic_light_label` từ
  `envs/carla_lane_keep_env.py` sang `policy/observation.py`) là một cải thiện tái sử dụng
  hợp lý — cùng logic tính `lane_offset_m`/`heading_error_rad` giờ dùng chung được bởi cả
  `CarlaLaneKeepEnv` lẫn Bridge Server (`CarlaConsole`), giảm nguy cơ hai nơi tính lệch làn
  theo hai công thức khác nhau. Không có thay đổi hành vi (đối chiếu diff: `magnitude()` và
  `normalize_angle()` được thay bằng biểu thức tương đương nội tuyến).
- **Docstring giải thích quyết định thiết kế** (không chỉ mô tả "làm gì" mà cả "vì sao") xuất
  hiện nhất quán trong toàn bộ module — hiếm gặp và có giá trị lớn cho bảo trì lẫn viết báo cáo
  (tài liệu này khai thác trực tiếp nhiều đoạn rationale đã có sẵn trong code).

### 4.2 Điểm cần lưu ý / rủi ro tiềm ẩn

| Vị trí | Quan sát | Mức độ | Đề xuất |
|---|---|---|---|
| `envs/carla_lane_keep_env.py::_compute_reward` | Không có phạt nào cho việc đứng yên/lùi kéo dài (§2.5) — có thể là một điểm hội tụ cục bộ không mong muốn | Thấp (thiết kế, chưa xác nhận bằng thực nghiệm) | Theo dõi phân phối `terminate_reason=time_limit` kèm `episode_reward`≈0 khi có log thật; bổ sung phạt/thưởng theo quãng đường nếu xảy ra |
| `sac/sac_agent.py::__init__` | `target_entropy` mặc định hard-code `-2.0` thay vì suy ra từ `action_dim` truyền vào constructor | Rất thấp (action_dim luôn = 2 trong phạm vi đồ án hiện tại) | Nếu mở rộng action space sau này (ví dụ thêm hành động cho route command), nhớ cập nhật hằng số này theo `action_dim` thực tế thay vì để mặc định cứng |
| `README.md` / `manual_train_drl.md` | Toàn bộ module **chưa được chạy với CARLA server thật** tại thời điểm viết (tác giả ghi rõ trong code) | Cần xác nhận trước khi dùng số liệu | Bắt buộc chạy smoke test (đã có hướng dẫn sẵn) trước khi lấy số liệu đưa vào báo cáo |
| Toàn hệ thống | Trọng số reward (§2.6) là giá trị khởi tạo theo lý luận thiết kế, **chưa qua kiểm chứng thực nghiệm** trên CARLA thật | Cần thực nghiệm | Thực hiện quy trình tinh chỉnh ở §2.7, ghi lại các bộ trọng số đã thử trong báo cáo như một phần của ablation |

Không phát hiện lỗi logic (correctness bug) nghiêm trọng nào trong luồng tính reward, GAE,
cập nhật PPO/SAC, hay các buffer khi rà soát thủ công công thức so với tài liệu tham khảo gốc
(Schulman 2017; Haarnoja 2018) — các điểm ở bảng trên đều là rủi ro thiết kế/thực nghiệm cần
xác nhận bằng dữ liệu thật, không phải lỗi lập trình.

---

## 5. Hạn chế và hướng phát triển

- **Chưa kiểm thử với CARLA server thật** — toàn bộ thiết kế trong tài liệu này dựa trên rà
  soát logic tĩnh; số liệu định lượng (reward hội tụ, tỉ lệ va chạm, MAE lệch làn) cần lấy từ
  lần chạy thật trước khi đưa vào báo cáo cuối.
- **1 CARLA instance / 1 environment** — chưa song song hoá thu thập rollout qua nhiều world;
  có thể mở rộng bằng cách chạy nhiều `CarlaUE4.exe` trên các port khác nhau (code đã viết theo
  hướng mọi state cục bộ trong instance `CarlaLaneKeepEnv`, không có biến toàn cục, thuận lợi
  cho mở rộng này).
- **Chưa tích hợp điều hướng theo tuyến A\*** — phạm vi hiện tại chỉ bám làn; khi
  `router_plan/Global_Route_Planner.py` sẵn sàng, mở rộng tự nhiên là thêm
  `route_target_local_x/y` + one-hot `route_command` vào `ObservationContract` và reward (ví
  dụ thưởng tiến độ theo tuyến, phạt đi sai `route_command`), rồi train lại IL trước khi
  warm-start DRL (giữ nguyên tắc: IL và DRL luôn chung một observation contract).
- **Trọng số reward chưa được kiểm chứng thực nghiệm** (§2.6) — là hạng mục thực nghiệm ưu
  tiên cao nhất trước khi kết luận về hiệu năng PPO/SAC.
- **Chưa xử lý rõ ràng hành vi đứng yên/lùi kéo dài** (§2.5, §4.2) — cần theo dõi và có thể cần
  bổ sung thành phần reward mới nếu quan sát thấy trong thực nghiệm.

---

## 6. Tài liệu tham khảo

- Schulman, J. et al. (2017). *Proximal Policy Optimization Algorithms.* arXiv:1707.06347.
- Haarnoja, T. et al. (2018). *Soft Actor-Critic: Off-Policy Maximum Entropy Deep
  Reinforcement Learning with a Stochastic Actor.* ICML 2018 / arXiv:1801.01290.
- Haarnoja, T. et al. (2018). *Soft Actor-Critic Algorithms and Applications.*
  arXiv:1812.05905. (Automatic temperature tuning.)
- Fujimoto, S., van Hoof, H., Meger, D. (2018). *Addressing Function Approximation Error in
  Actor-Critic Methods.* arXiv:1802.09477. (Clipped Double-Q, nguồn gốc twin-Q trong SAC/TD3.)
- Pardo, F. et al. (2018). *Time Limits in Reinforcement Learning.* arXiv:1712.00378.
- Raffin, A. et al. — Stable-Baselines3 (tham chiếu cài đặt chuẩn cho PPO liên tục, clipped
  surrogate, và `ReplayBuffer(optimize_memory_usage=True)`).

---

*Tài liệu thiết kế, biên soạn dựa trên rà soát mã nguồn `drl_training/` tại commit
`a8885d7` và các thay đổi đang chờ commit (`envs/carla_lane_keep_env.py`,
`policy/observation.py`) — 2026-08-14.*
