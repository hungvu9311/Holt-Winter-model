Tổng quan về mô hình dự đoán:
Mô hình dự đoán doanh thu được kết hợp từ 2 phương pháp:
  + Mô hình Holt-winter: mô hình này sẽ áp dụng cho các merchant có >= 13 tháng doanh thu 
  + Trung bình doanh thu: phương pháp này sẽ áp dụng cho các merchant có < 13 tháng doanh thu

Cấu trúc Project:
- Folder /data: chứa file input đầu vào 
- Folder /config: chứa đường dẫn và các biến sử dụng để train mô hình
- Folder /src: chứa các function sử dụng để xây mô hình, cụ thể như sau:
  + /src/data_handling: chứa các function để save, load dataset
  + /src/data_preprocessing: chứa các function tiền xử lý dữ liệu (e.g missing value, outlier, ...) và các model function
- Folder /prediction_model: 
  + file training.py : file train mô hình AutoARIMA --> tìm ra best param cho mỗi merchant_id
  + file main.py: file forecast revenue bằng mô hình AutoARIMA và phương pháp bình quân doanh thu các tháng liền trước

Các bước chạy script:
Bước 1: Chạy file training.py, bộ best param với mỗi merchant được lưu lại thành file pickle trong folder /trained_param
Bước 2: Chạy file main.py