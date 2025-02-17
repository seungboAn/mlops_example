import tensorflow as tf
from tensorflow import keras

load_path = 'mlops/best_model/1/model.keras'
model = keras.models.load_model(load_path)

# Keras 3 호환 형식으로 모델 재 저장
model.save('mlops/best_model/1/model_keras_v3.keras') #다른이름으로 저장

#재 저장된 모델 로드 확인
model_v3 = keras.models.load_model('mlops/best_model/1/model_keras_v3.keras')
model_v3.summary()

converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
tflite_model = converter.convert()

tflite_file_path = 'best_model.tflite'
with open(tflite_file_path, 'wb') as f:
    f.write(tflite_model)

interpreter = tf.lite.Interpreter(model_path=tflite_file_path)
interpreter.allocate_tensors()

signatures = interpreter.get_signature_list()
print(signatures)

classify_lite = interpreter.get_signature_runner('serving_default')
print(classify_lite)