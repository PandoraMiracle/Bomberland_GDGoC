# Báo cáo Phân tích Tính toán `ret` trong MAPPO

## 1. Công thức chính xác của `ret`

Trong mã nguồn cũ, biến `ret` được in ra terminal (thông qua biến `mean_return`) là **trung bình cộng của tất cả các phần thưởng trên từng bước (step rewards)** của các tập dữ liệu đang được thu thập trong `rollout_length`. 

Công thức cụ thể trong mã nguồn:
```python
# episode_returns chứa danh sách các step rewards cho mỗi môi trường (env)
all_returns = [r for ep in episode_returns for r in ep]
mean_return = float(np.mean(all_returns)) if all_returns else 0.0
```
Như vậy, `ret` (mean_return) **KHÔNG PHẢI** là tổng phần thưởng của một tập (episode return), mà là **phần thưởng trung bình trên mỗi bước di chuyển** (Mean Step Reward).

## 2. Data Flow và Tensor Shapes

- **Môi trường (`vec_env.py`)**: Ở mỗi step, môi trường tính toán `dense_r = rb.compute_dense()` (đã nhân với hệ số `dense_coef`). Nếu kết thúc tập (done), `reward = dense_r + term_r` (term_r không nhân `dense_coef`). Nếu không, `reward = dense_r`. Giá trị `reward` này (kiểu float) được trả về.
- **Thu thập (`train_mappo.py`)**: Giá trị `reward` được thêm vào mảng `episode_returns[e_i]`. Đồng thời, nó được lưu vào buffer qua `buffer.add(..., reward, ...)`.
- **Buffer (`rollout_buffer.py`)**: `rewards` được lưu thành numpy array với shape `(T, E)` tương ứng `(rollout_length, num_envs)`.
- **GAE**: Dùng tensor `rewards` `(T, E)` cùng với `values` `(T, E)` để tính `advantages` `(T, E)` và `returns = advantages + values`.
- **Update (`ppo_update.py`)**: Dữ liệu được làm phẳng (flatten) thành shape `(T*E,)` trước khi đưa vào PPO. `advantages` ở đây sẽ được chuẩn hóa (normalize) bằng công thức `(adv - mean) / (std + 1e-8)`.

## 3. Vì sao `ret` có giá trị xấp xỉ 0 (≈ 0.001)?

Giá trị `ret` nhỏ xuất phát từ bản chất của metric **Mean Step Reward** trong một môi trường có số bước rất dài (lên đến 500 bước):
- **Phần lớn các bước không có sự kiện**: Agent chỉ di chuyển thông thường trên ô cỏ, không nhận được reward (0.0).
- **Hệ số Dense Reward**: Khi xảy ra sự kiện (như phá thùng +0.02), giá trị này bị nhân với hệ số `dense_coef` (ví dụ 0.5), nên step reward chỉ là +0.01.
- **Terminal Reward**: Reward quan trọng nhất (+1.0 cho hạng 1) chỉ được phát **một lần duy nhất** vào cuối episode.
- Khi lấy tổng reward của một episode (giả sử agent chơi rất tốt, đạt return 1.0) chia cho số bước sống sót (ví dụ 450 steps), thì trung bình step reward là `1.0 / 450 ≈ 0.0022`. 

Do đó, `ret` hiển thị ở mức `0.001` - `0.003` là hoàn toàn bình thường và phản ánh đúng phép tính.

## 4. `ret = 0.001` có đồng nghĩa với việc Model không học không?

**KHÔNG.**
Vì `ret` là trung bình của step reward, việc nó gần 0 chỉ là hệ quả toán học của phép chia cho số bước (episode length) quá lớn. Model vẫn có thể đang nhận được Episode Return rất cao (như thắng lợi +1.0 và diệt đối thủ) và đang hội tụ bình thường. Sự kiện quan trọng nhất là thắng (term_r) vẫn mang tín hiệu gradient cực mạnh qua GAE mà không bị triệt tiêu bởi độ dài tập.

## 5. Các Metric nên dùng để đánh giá hiệu suất học

Để theo dõi xem agent có đang tiến bộ hay không, thay vì dùng `ret` (mean step reward), cần sử dụng:
1. **`episode_return_mean` / `episode_return_min` / `episode_return_max`**: Tổng phần thưởng thực tế mà agent nhận được khi hoàn thành 1 màn chơi.
2. **`tracker_mean_kills`, `tracker_mean_boxes`, `tracker_mean_items`**: Số liệu in-game (tiêu diệt, phá thùng, ăn đồ) chứng tỏ agent đang học các kỹ năng hữu ích.
3. **`eval_metrics`**: Các chỉ số thu được sau khi chạy tập đánh giá (evaluate) khách quan với opponents.
4. **`value_mean`**: Nếu agent học được, dự đoán giá trị trung bình của `critic` sẽ tăng dần.

## 6. Kiểm tra ảnh hưởng của Checkpoint / Resume

Qua quá trình truy vết `vec_env.py`, `rollout_buffer.py`, và `ppo_update.py`:
- Cả `RewardBuilder` và hệ thống tính GAE/Returns **không sử dụng** bất kỳ bộ lọc trạng thái tích lũy (running statistics / normalizer) nào cho Returns hay Rewards (như cơ chế `VecNormalize` của Stable-Baselines3).
- **Advantages** được chuẩn hóa (normalize) trong mỗi lần cập nhật PPO theo từng batch của *hiện tại* `adv = (adv - adv.mean()) / adv.std()`.
- Do đó, việc dừng và **resume training từ checkpoint không gây mất mát thông tin chuẩn hóa** hoặc tạo ra cú sốc cho thuật toán, quá trình học có thể tiếp tục một cách mượt mà.

## 7. Các thay đổi Logging đã được bổ sung

Các log mới đã được thêm thành công mà **không thay đổi logic training hay phần thưởng**:
- Sửa file: `training/mappo/ppo_update.py`
  - Thêm các giá trị `advantage_mean`, `advantage_std` (trước và sau khi normalize).
  - Thêm `return_mean`, `return_std`, `return_min`, `return_max` tính toán từ GAE returns.
  - Thêm `value_mean`, `value_std` tính toán từ Value outputs.
- Sửa file: `training/mappo/train_mappo.py`
  - Khai báo list theo dõi `update_episode_returns` chứa tổng phần thưởng của các tập đã hoàn thành.
  - Trích xuất `raw_rewards` từ `buffer.rewards` để lấy thống kê `raw_reward_mean`, `raw_reward_sum`, `min`, `max`.
  - In ra các giá trị với độ phân giải **6 chữ số thập phân**, cung cấp cái nhìn chi tiết hơn về log.
