# Kết quả thực nghiệm nhánh SAC — vì sao chọn PPO làm mô hình triển khai

> **Vị trí đề xuất trong báo cáo**: chèn vào chương Kết quả, ngay sau phần trình bày kết quả
> PPO. Tài liệu thiết kế đã đặc tả SAC ở §10 và so sánh lý thuyết ở §11; phần này báo cáo
> *cái đã đo được*. Độ dài ~1 trang. Nguồn số liệu: `drl_training_sac/runs/`.

## 1. Thiết kế thí nghiệm

Để so sánh có giá trị, hai nhánh phải chỉ khác nhau ở **thuật toán tinh chỉnh**. Toàn bộ các
thành phần còn lại được giữ nguyên: cùng checkpoint khởi tạo nóng `best_il_model.pth`, cùng
hợp đồng quan sát, cùng không gian hành động, cùng hàm phần thưởng ở chế độ `normalized`,
cùng điều kiện kết thúc episode, cùng bốn bản đồ huấn luyện (Town01–Town04) và cùng giao thức
đánh giá 30 episode/bản đồ ở chế độ tất định.

Nhánh SAC được chạy **sáu lần** với các siêu tham số khác nhau nhằm loại trừ khả năng kết quả
kém chỉ do chưa tinh chỉnh:

| Lần chạy | Khác biệt chính | Số bước |
|---|---|---|
| sac_a | `bc_coef` 2.5 | 60 000 |
| sac_b | `bc_coef` 1.0 | 60 000 |
| sac_c | thêm `critic_warmup_steps` 4000 | 21 151 |
| sac_d | thêm `gamma` 0.95, `explore_epsilon` 0.25 | 39 154 |
| sac_e | `bc_coef` 0.0 (SAC thuần) | 30 101 |
| sac_f | như sac_d, chạy đủ ngân sách | 60 000 |

## 2. Kết quả đánh giá

Đánh giá 30 episode mỗi ô, chế độ tất định. Độ lệch làn chỉ tính trên **đoạn đường thẳng** —
trong nút giao, waypoint tham chiếu nhảy sang nhánh cắt nên phép đo mất ý nghĩa (§13).

Cột SAC lấy từ `sac_b`, checkpoint SAC tốt nhất trong sáu lần chạy.

| Bản đồ | PPO v4 va chạm | SAC va chạm | PPO lệch làn (m) | SAC lệch làn (m) |
|---|---|---|---|---|
| Town01 | **0,0 %** | 83,3 % | 0,129 | 0,145 |
| Town03 | **16,7 %** | 43,3 % | 0,375 | **0,187** |
| Town04 | **33,3 %** | 36,7 % | 0,099 | **0,081** |
| Town05 | **13,3 %** | 43,3 % | 0,148 | 0,192 |
| Trung bình | **15,8 %** | 51,7 % | 0,188 | **0,151** |

Kết quả chia hai chiều rõ rệt: **SAC bám làn tốt hơn PPO khoảng 20 %** (0,151 so với 0,188 m)
nhưng **va chạm nhiều hơn 3,3 lần** (51,7 % so với 15,8 %). Vì an toàn là ràng buộc cứng của
bài toán điều khiển, PPO v4 được chọn làm mô hình triển khai.

## 3. Kết quả là giới hạn cấu trúc, không phải do chưa tinh chỉnh

Ba lần chạy có siêu tham số khác nhau đáng kể — sac_b (γ = 0,99; không thăm dò bổ sung),
sac_d (γ = 0,95; `explore_epsilon` 0,25) và sac_f (như sac_d, chạy đủ 60 000 bước) — cho kết
quả **không phân biệt được về mặt thống kê**. Kiểm định z hai tỉ lệ cho tỉ lệ va chạm và
kiểm định Welch t cho độ lệch làn, trên hai bản đồ, với hiệu chỉnh Bonferroni cho bốn phép
kiểm (ngưỡng |z|, |t| > 2,50):

| Phép so | z (va chạm) | t (lệch làn) | Kết luận |
|---|---|---|---|
| sac_b → sac_f, cùng 60 000 bước | +0,36 · +0,26 | −0,41 · −0,42 | không phân biệt được |
| sac_f, 30 000 → 60 000 bước | −1,04 · −0,40 | +0,51 · +0,43 | không phân biệt được |

*(mỗi ô ghi hai giá trị ứng với Town01 · Town05)*

Hàng thứ hai đồng thời cho thấy **30 000 bước cuối không mang lại cải thiện đo được nào** —
tức các lần chạy trước đó không thất bại vì bị cắt sớm.

## 4. Cơ chế: critic học được giá trị trạng thái nhưng không phân biệt được hành động

Nguyên nhân được xác định bằng một phép đo trực tiếp trên 256 quan sát thật thu từ chính môi
trường. Giữ nguyên trạng thái, quét lệnh lái toàn dải [−1, +1] và đo Q thay đổi bao nhiêu, so
với độ lệch của Q **giữa các trạng thái** khác nhau:

| Bước huấn luyện | 5 000 | 10 000 | 20 000 | 25 000 | 30 000 | 35 000 | 40 000 |
|---|---|---|---|---|---|---|---|
| sac_d | 1,6 % | 3,0 % | 6,2 % | 9,5 % | 10,3 % | 10,3 % | — |
| sac_f | 4,8 % | 1,4 % | 4,9 % | 8,7 % | 7,6 % | 6,6 % | 8,0 % |

Đại lượng này **bão hoà quanh 8–10 % từ bước 25 000–30 000** và không tăng thêm dù huấn luyện
tiếp 30 000 bước. Nghĩa là hơn 90 % những gì critic học được là *đang ở tình huống nào*
(≈ V(s)), không phải *nên làm gì trong tình huống đó*. Vì actor của SAC chỉ nhận gradient qua
∂Q/∂a, đó là trần của toàn thuật toán.

Đáng lưu ý, `explained_variance` của critic đạt 0,93–0,99 ở **mọi** lần chạy. Critic dự báo
return rất chính xác — nó giỏi đúng đại lượng không sử dụng được. Điều này cho thấy
`explained_variance`, tuy là thước đo đúng để thay cho `critic_loss`, vẫn **không** phát hiện
được dạng hỏng này; cần đo trực tiếp độ nhạy theo hành động.

## 5. Vì sao PPO không vướng cùng vấn đề

Khác biệt nằm ở nguồn tín hiệu gán công trạng cho hành động. PPO ước lượng advantage bằng
GAE(λ), tức **cộng dồn hàng chục phần thưởng thực tế** dọc quỹ đạo; hậu quả của một hành động
nằm sẵn trong lợi tức thực nghiệm. SAC bắt buộc phải học một hàm Q có đạo hàm theo hành động.

Trong bài toán này, mỗi hành động kéo dài 0,2 giây (`action_repeat` = 4) và chính sách tự
hiệu chỉnh ở bước kế tiếp, nên hậu quả dài hạn của **một** hành động vốn dĩ nhỏ — với
γ = 0,95, một hành động chiếm khoảng 5 % chân trời hiệu dụng. ∂Q/∂a nhỏ vì vậy là **tính chất
của bài toán**, không phải do critic học kém.

Thêm vào đó, PPO có vùng tin cậy (`clip_range` 0,2 và dừng sớm theo `target_kl`) nên chính
sách không thể rời xa điểm khởi tạo nóng trong một lần cập nhật. SAC tối ưu tự do; thực
nghiệm cho thấy sau 650 bước cập nhật actor, lệnh lái trôi từ −0,001 sang −0,274 và lệnh ga
từ +0,288 sang −0,372 — khởi tạo nóng bị xoá hoàn toàn. Phải bổ sung ràng buộc kiểu TD3+BC
mới giữ được, nhưng chính ràng buộc đó lại giới hạn mức cải thiện.

## 6. Kết luận

Với bài toán bám làn khởi tạo nóng từ học bắt chước trên CARLA, **PPO phù hợp hơn SAC**, và
lý do mang tính cấu trúc chứ không phải do lựa chọn siêu tham số: cơ chế gán công trạng của
SAC đòi hỏi hàm Q phân biệt được hành động, điều mà đặc tính chân trời của bài toán này không
cho phép đạt tới mức đủ dùng. Ưu thế lấy mẫu của phương pháp off-policy không bù được nhược
điểm đó.

Nhánh SAC vì vậy được giữ lại trong báo cáo như **nhánh đối chứng có kiểm soát**, cung cấp
căn cứ định lượng cho việc lựa chọn PPO thay vì một lựa chọn mặc định không giải thích.
