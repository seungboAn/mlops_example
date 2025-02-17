from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# TensorFlow Lite 모델 로드
interpreter = tf.lite.Interpreter(model_path="cifar10_tuned_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess_image(image_data):
    # 이미지 전처리 (모델 입력 형태에 맞게 수정)
    image = tf.image.decode_jpeg(image_data, channels=3) # 이미지 디코딩
    image = tf.image.resize(image, (32, 32)) # 이미지 사이즈 조정
    image = tf.cast(image, tf.float32) / 255.0 # 정규화
    image = np.expand_dims(image, axis=0) # 배치 차원 추가
    return image

@app.route('/predict', methods=['POST'])
def predict():
    try:
        image_file = request.files['image']
        image_data = image_file.read()
        image = preprocess_image(image_data)

        # TensorFlow Lite 모델 추론
        interpreter.set_tensor(input_details[0]['index'], image)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])

        # 예측 결과 반환
        prediction = np.argmax(output)
        return jsonify({'prediction': int(prediction)})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8501)