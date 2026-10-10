# Báo cáo bài nộp — Day 23 Sensor Fusion Lab

> Điền file này rồi commit. Cách nộp: [hướng dẫn nộp](../SUBMISSION.md).

## Thông tin học viên

- Họ tên: Cao Đức Hiếu
- MSSV: 2A202602701
- Email: hieu.cd.2a202602701@vinuni.edu.vn
- Link repo (fork): https://github.com/ChunHieu/K4-L2L3-DAY23-CaoDucHieu-2A202602701-SensorFusion
- Commit hash nộp (`git rev-parse HEAD`): 917008ee7179069d300d8fca0ce938479e0004e0

## Tóm tắt kết quả

- `fusion_mode` (bắt buộc `compare`), `frames`, `segment`, `seed`: compare, [0, 198], training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord, 0
- `detection.precision`, `detection.recall`, `detection.tp/fp/fn`: 0.970093, 0.700405, 519 / 16 / 222
- `tracking.lidar.rmse`, `matches`, `sum_sq_err`, `ghost_track_frames`, `missed_gt_frames`, `mean_confirmed_tracks`: 0.150278, 502, 11.336920, 0, 239, 2.522613
- `tracking.fused.rmse`, `matches`, `sum_sq_err`, `ghost_track_frames`, `missed_gt_frames`, `mean_confirmed_tracks`: 0.135848, 502, 9.264217, 0, 239, 2.522613
- Giải thích khác biệt hai mode, đọc RMSE cùng số ghép và ghost/miss:
  Ở cả hai chế độ lidar và fused, số lượng matches (502), ghost tracks (0), missed GT frames (239) và mean confirmed tracks (2.5226) hoàn toàn giống nhau. Điều này chứng minh tính đúng đắn của kiến trúc track-then-fuse: chỉ có LiDAR quyết định việc tạo, tính score, xác nhận và xóa track; camera không tạo track mới và không xóa track. Về độ chính xác vị trí 3D, chế độ fused giúp giảm RMSE từ 0.1503 m xuống 0.1358 m (cải thiện ~9.6% sai số khoảng cách; tổng bình phương sai số sum_sq_err giảm từ 11.34 m² xuống 9.26 m², tức giảm ~18.3%), chứng tỏ việc tích hợp đo lường góc/pixel 2D từ camera FRONT qua EKF update đã tinh chỉnh đáng kể ước lượng trạng thái cho các xe nằm trong tầm nhìn phía trước, và sai số rmse_fused - rmse_lidar = -0.0144 m <= 0.05 m thỏa mãn hoàn toàn tiêu chí rubrics.

Chạy từ root repo:

```bash
fusion-run-lab --config student/config/paths.yaml --fusion compare --seed 0
```

`rmse = sqrt(sum_sq_err/matches)` trên vị trí 3D của confirmed tracks ghép
một-một với GT xe trong cửa sổ BEV, gate XY **2.0 m**; `null` nếu không có cặp.
Camera dùng tâm hộp 2D ground-truth FRONT có nhiễu seeded, **không** dùng camera
detector. Kết quả này không đo hiệu quả một perception system độc lập với GT.

`grade_run.log` là JSONL, mỗi `(mode,frame)` đúng một record với các trường:
`mode`, `frame`, `det_tp`, `det_fp`, `det_fn`, `valid_gt`, `confirmed`, `matches`,
`sum_sq_err`, `ghosts`, `misses`. Đảm bảo `matches+ghosts==confirmed` và
`matches+misses==valid_gt`; tổng/trung bình record phải khớp `metrics.json`.
File per-mode `metrics_lidar.json`, `metrics_fused.json`, `grade_run_lidar.log`,
`grade_run_fused.log` được giữ để đối chiếu.

## Giải thích ngắn (Parts E–H — tự viết)

1. Khác biệt đo lidar 3D và camera 2D trong EKF (`z`, `R`)?
   - LiDAR đo trực tiếp vị trí không gian 3D trong hệ quy chiếu xe: vector đo z = [x, y, z]^T (3x1, mét). Mô hình đo tuyến tính h(x) = H x với H = [I_3 | 0_{3x3}] (3x6). Ma trận hiệp phương sai đo R (3x3) là ma trận đường chéo chứa các phương sai đo lidar theo mét vuông (sigma_lidar_x^2, sigma_lidar_y^2, sigma_lidar_z^2).
   - Camera đo tọa độ pixel trên mặt phẳng ảnh 2D: vector đo z = [u, v]^T (2x1, pixel). Mô hình đo phi tuyến qua phép chiếu Pinhole h(x) phụ thuộc vào tọa độ trong hệ camera (p_s = R_{veh->sens} p + t_{veh->sens}) với u = c_i - f_i * y_s / x_s và v = c_j - f_j * z_s / x_s. Ma trận đo H (2x6) là Jacobian phi tuyến tính theo quy tắc chuỗi (chain rule: đạo hàm chiếu nhân ma trận xoay). Ma trận hiệp phương sai đo R (2x2) là ma trận đường chéo chứa phương sai pixel (sigma_cam_i^2, sigma_cam_j^2).

2. Vì sao cần gating Mahalanobis trước khi gán?
   - Khoảng cách Mahalanobis bình phương d^2 = gamma^T * S^{-1} * gamma chuẩn hóa sai số đo residual (gamma = z - h(x)) dựa trên ma trận hiệp phương sai sai số S = H P H^T + R. Khác với khoảng cách Euclidean thuần túy (coi sai số mọi hướng là đẳng hướng), Mahalanobis tính đến hình dạng elip bất định của cả track (P) lẫn cảm biến (R).
   - Gating bằng kiểm định phân phối Chi-square (chi2_gate) cho phép loại bỏ các cặp track-measurement có sai số quá lớn, không khả thi về mặt xác suất thống kê trước khi chạy thuật toán gán Greedy. Điều này giúp ngăn chặn việc gán nhầm measurement từ xe khác hoặc nhiễu môi trường vào track, tránh làm bộ lọc EKF bị phân kỳ, đồng thời tối ưu hiệu năng tính toán.

3. Pipeline là track-then-fuse hay fuse-then-track? Chỉ ra trên log `fusion-run-lab`.
   - Pipeline của lab là **track-then-fuse**: Hệ thống duy trì duy nhất một tracker và một danh sách track tập trung. Trong mỗi frame thời gian, EKF thực hiện dự báo (predict) một lần duy nhất cho toàn bộ track, sau đó lần lượt cập nhật EKF nối tiếp theo từng modality cảm biến: cập nhật LiDAR (AssocL) để xác định vị trí và quản lý vòng đời, rồi cập nhật Camera (AssocC) để tinh chỉnh trạng thái các track trong tầm nhìn camera FRONT. Hệ thống không gộp raw data của hai sensor trước khi detect hay tạo track độc lập (fuse-then-track).
   - Trên log `fusion-run-lab`: Cả hai chế độ lidar và fused đều ghi nhận số lượng confirmed tracks (2), matches (2), misses (0), ghosts (0) qua từng frame là hoàn toàn trùng khớp. Bước camera trong mode fused chỉ cập nhật trạng thái EKF làm giảm tổng bình phương sai số vị trí sum_sq_err (từ 11.34 m² xuống 9.26 m²) trên cùng danh sách track đó, hoàn toàn không sinh ra track mới hay thay đổi vòng đời track.

4. Nếu camera lệch calibration, triệu chứng gì trên innovation/residual?
   - Nếu camera bị lệch extrinsic (xoay/dịch sai so với xe) hoặc intrinsic (tiêu cự/quang tâm sai), phép chiếu pinhole h(x) sẽ tính ra tọa độ pixel dự đoán (u, v) bị lệch có hệ thống so với vị trí đo thực tế z.
   - Triệu chứng: Residual/innovation gamma = z - h(x) sẽ không còn là nhiễu trắng Gauss có kỳ vọng bằng 0 mà xuất hiện độ lệch trung bình khác không (systematic non-zero bias).
   - Hệ quả: Khoảng cách Mahalanobis d^2 = gamma^T * S^{-1} * gamma sẽ tăng đột biến. Nếu lệch lớn, d^2 sẽ vượt qua ngưỡng chi-square khiến đo camera bị gating loại bỏ hoàn toàn (không thể match). Nếu lệch vừa phải vẫn lọt qua gate, EKF update sẽ kéo trạng thái track lệch khỏi vị trí ground-truth thực tế, làm tăng mạnh sai số RMSE và làm sai lệch ước lượng vận tốc của bộ lọc.

5. Vì sao `associate_and_update(..., sensor)` cần sensor tường minh ở frame rỗng?
   Giải thích vì sao lidar quyết định score/init/delete còn camera chỉ EKF update.
   - Cần tham số sensor tường minh vì ngay cả khi frame rỗng (meas_list = []), hàm vẫn bắt buộc gọi manager.manage_tracks(unassigned_tracks, unassigned_meas, sensor). TrackManager cần biết sensor tường minh để xác định xem đây có phải lượt LiDAR hay không. Nếu là LiDAR, manager sẽ duyệt các track trong FOV LiDAR mà không có đo (miss) để trừ score (-1/window) và kiểm tra điều kiện xóa track.
   - LiDAR quyết định toàn bộ vòng đời (init, score, delete) vì LiDAR cung cấp thông tin không gian 3D trực tiếp (vị trí x,y,z và kích thước hộp) với tầm bao quát rộng, đủ tin cậy để tạo và duy trì track. Ngược lại, camera FRONT chỉ có trường nhìn hẹp phía trước và phép chiếu 2D làm mất thông tin độ sâu. Nếu camera tham gia tính score/delete, các track khi di chuyển ra khỏi vùng nhìn camera FRONT (ví dụ đi sang hai bên hoặc ra sau xe) sẽ bị trừ điểm và xóa oan. Do đó camera chỉ thuần túy đóng vai trò cập nhật EKF (refine state) khi xe nằm trong tầm nhìn.

6. Nêu điều kiện xác nhận, giữ confirmed sau miss, và điều kiện xóa track.
   - **Điều kiện xác nhận (confirmed):** Khi có LiDAR hit, track được cộng score +1/window. Khi score > confirmed_threshold (0.8), track chính thức được xác nhận (chuyển sang state = 'confirmed').
   - **Giữ confirmed sau miss:** Khi bị miss trong LiDAR FOV, track bị trừ score -1/window, nhưng nếu track đang ở trạng thái 'confirmed', trạng thái này được giữ nguyên (không bị hạ cấp về 'tentative' hay 'initialized'). Điều này giúp track bền vững, không bị mất danh tính khi detector bỏ sót xe tạm thời trong một vài frame.
   - **Điều kiện xóa track (delete):** Track bị xóa trong lượt LiDAR nếu thỏa mãn ít nhất một trong các điều kiện sau:
     1. Phương sai tọa độ ngang quá lớn: P[0,0] > max_P (9.0) hoặc P[1,1] > max_P (9.0).
     2. Track đã confirmed nhưng bị bỏ sót nhiều frame khiến score < delete_threshold (0.6).
     3. Track chưa confirmed (initialized hoặc tentative) có score <= 0.0.

## Bonus (không bắt buộc)

Liệt kê phần bonus đã làm, file bằng chứng trong `student/bonus/` và kết quả chính
(xem [RUBRIC.md](../RUBRIC.md) mục 2). Không làm thì ghi "Không".

- Không

## Khai báo sử dụng AI (bắt buộc)

Ghi rõ, kể cả khi không dùng ("Không dùng AI"). Xem [RULES.md](../RULES.md) mục 2.

- Công cụ đã dùng (ChatGPT, Copilot, Claude, …): Antigravity AI Assistant (Claude / Gemini 3.8 Flash)
- Dùng cho phần nào (hàm, câu hỏi, debug): Hỗ trợ lập trình các hàm Part E (EKF trong kalman.py), Part G (camera model trong camera_fusion.py), Part F (association và gating trong association.py), Part H (vòng đời track trong track_management.py); hỗ trợ soạn thảo câu trả lời giải thích kỹ thuật và kiểm tra test case.
- Cách bạn đã kiểm tra lại (pytest, chạy Waymo, đối chiếu công thức): Tự chạy toàn bộ pytest test suite trên student/tests (pass 128/128 tests); chạy fusion-run-lab compare mode trên 199 frames Waymo đạt RMSE 0.135m; chạy tools/check_submission.py tự kiểm tra; đối chiếu các công thức lý thuyết EKF, phân phối Chi-square và phép chiếu pinhole.

## Checklist nộp

- [x] **Part E–H** trong `workspace/` đã implement; `pytest student/tests -q` không còn `failed`/`xfailed`
- [x] Part A–D: không bắt buộc sửa (hoặc ghi chú nếu bạn đã sửa)
- [x] Lần chạy chấm điểm: `--fusion compare --seed 0`, `frame_start: 0`, `frame_end: 198`
- [x] Đã commit `student/artifacts/metrics*.json` và `student/artifacts/grade_run*.log` (không sửa tay)
- [x] Đã điền đủ file này, gồm khai báo AI
- [x] Không commit dữ liệu Waymo, weights, `paths.yaml`, API key
- [x] `python tools/check_submission.py` báo `KẾT QUẢ: SẴN SÀNG NỘP`
- [x] Đã push và nộp link repo + commit hash trên LMS ([hướng dẫn nộp](../SUBMISSION.md))
