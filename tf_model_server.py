import tensorflow as tf

# 모델 로드
model = tf.keras.models.load_model('mlops/best_model/1/model.keras')

# SavedModel로 저장
tf.saved_model.save(model, 'mlops/best_model/1')