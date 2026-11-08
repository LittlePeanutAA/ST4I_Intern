# ST4I_Intern

Các project thực tập ở ST4I về kiểm tra lỗi sản phẩm công nghiệp, gộp từ các repo cũ. Mỗi folder là một repo trước đây, lịch sử commit vẫn được giữ. `SurfaceDefect` đã được viết lại hoàn toàn (2026-11), bản cũ vẫn xem được trong lịch sử commit.

| Folder | Nội dung |
|---|---|
| [SurfaceDefect](SurfaceDefect) | Phát hiện bất thường trên chi tiết nhựa bằng học không giám sát (PaDiM / PatchCore, anomalib 2), tối ưu OpenVINO cho CPU yếu; có kiểm tra rò rỉ dữ liệu, so sánh FP32/FP16/INT8 và trạm kiểm tra mẫu |
| [InspectContour](InspectContour) | Phát hiện lỗi hình dạng 2D từ contour bằng xử lý ảnh truyền thống (OpenCV, C++) |
