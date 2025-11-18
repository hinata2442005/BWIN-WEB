from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np

# Khởi tạo ứng dụng Flask
app = Flask(__name__)

# Load mô hình SVM và Scaler đã lưu
try:
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
    print("Đã tải mô hình và scaler thành công.")
except FileNotFoundError:
    print("Lỗi: Không tìm thấy tệp model.pkl hoặc scaler.pkl. Hãy chạy train_model.py trước.")
    exit()

# Định nghĩa tên của các lớp
class_names = ['Ác tính (Malignant)', 'Lành tính (Benign)']

@app.route('/')
def home():
    """Render trang chủ."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Nhận dữ liệu từ form, dự đoán và trả về kết quả."""
    if request.method == 'POST':
        try:
            # Lấy dữ liệu từ form
            # Chúng ta dùng 'mean radius' và 'mean texture'
            radius_mean = float(request.form['radius_mean'])
            texture_mean = float(request.form['texture_mean'])
            smoothness_mean = float(request.form['smoothness_mean'])
            compactness_mean = float(request.form['compactness_mean'])
            symmetry_mean = float(request.form['symmetry_mean'])
            fractal_dimension_mean = float(request.form['fractal_dimension_mean'])
            radius_se = float(request.form['radius_se'])
            texture_se = float(request.form['texture_se'])
            smoothness_se = float(request.form['smoothness_se'])
            compactness_se = float(request.form['compactness_se'])
            symmetry_se = float(request.form['symmetry_se'])
            fractal_dimension_se= float(request.form['fractal_dimension_se'])
          
            # Tạo mảng numpy từ dữ liệu
            input_data = np.array([[radius_mean, texture_mean, smoothness_mean, compactness_mean, symmetry_mean, fractal_dimension_mean, radius_se, texture_se, smoothness_se, compactness_se, symmetry_se, fractal_dimension_se]])
            
            # Chuẩn hóa dữ liệu đầu vào
            input_scaled = scaler.transform(input_data)
            
            # Thực hiện dự đoán
            prediction = model.predict(input_scaled)
            prediction_proba = model.predict_proba(input_scaled)
            
            # Lấy kết quả
            result_class = class_names[prediction[0]]
            confidence = prediction_proba[0][prediction[0]] * 100
            
            # Trả kết quả về cho trang index.html
            return render_template('index.html', 
                                   prediction_text=f'Kết quả dự đoán: {result_class}',
                                   confidence_text=f'Độ tin cậy: {confidence:.2f}%')

        except Exception as e:
            # Xử lý lỗi nếu có
            return render_template('index.html', prediction_text=f'Lỗi: {str(e)}')

if __name__ == "__main__":
    # Chạy Flask development server
    app.run(host="0.0.0.0", port=5000, debug=True)
