# Inspect Contour

Mục tiêu là sử dụng xử lý ảnh truyền thống để xác định được lỗi hình dạng 2D của một số loại sản phẩm công nghiệp có hình dáng không quá phức tạp. Chủ yếu sử dụng thư viện OpenCV.

## Trích xuất contour
Contour là một đặc trưng cực kỳ hữu ích đối với các ứng dụng liên quan đến hình dạng vật thể. Để trích xuất được contour chỉ bao gồm các đường biên hình dạng của vật thể,
ta cần phải xử lý ảnh và lọc contour. Lưu ý khi lọc ảnh ta chỉ nên sử dụng bộ lọc Bilateral, vì nó không làm biến đổi hình dạng vật thể như các bộ lọc khác.

## Đối sánh mẫu
Phương pháp xác định lỗi được sử dụng là đối sánh mẫu. Đường contour được trích xuất sẽ đem so sánh với đường contour của vật mẫu dựa trên chỉ số khoảng cách. Tại những vị trí trên đường contour
có khoảng cách so với mẫu lớn thì được xác định là lỗi.

## Tăng độ chính xác và tốc độ xử lý
Để tăng tốc độ tính toán, các phép toán được viết theo tập lệnh SSE2 của thư viện SIMD. Khi đó, các thanh ghi sẽ xử lý đồng thời nhiều phép toán tương tự như việc tính toán trên GPUs.

Để tăng độ chính xác, thuật toán subpixel được sử dụng. Khi đó, các điểm ảnh không còn là điểm nguyên nữa. Đường contour và khoảng cách của chúng sẽ được tính toán ở độ chính xác thập phân.

![](img.jpg)

### Lưu ý: Bởi vì tính bảo mật, cho nên file InspectContour.cpp đã bị xoá các phần chương trình chính.
