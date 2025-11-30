# app.py

import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import numpy as np
import json
import os

# ====================================================================
# 1. تعريف المسارات والثوابت
# ====================================================================
MODEL_PATH = 'my_model.keras'  # ملف النموذج
LABELS_PATH = 'labels.json'    # ملف التصنيفات
IMAGE_SIZE = (128, 128)        # حجم الإدخال
NUM_CLASSES = 13               # عدد الفئات كما في labels.json

# ====================================================================
# 2. تعريف نموذج CNN محسّن للتعميم (Generalization)
# ====================================================================

def build_improved_generalization_model(num_classes=NUM_CLASSES):
    """
    نموذج CNN محسّن مع Dropout و L2 Regularization للتعامل مع البيانات الجديدة وغير المألوفة.
    """
    model = models.Sequential([
        layers.Input(shape=(128, 128, 3)),
        layers.Rescaling(1./255),  # Normalization

        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),

        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),

        layers.Conv2D(128, 3, activation='relu'),
        layers.MaxPooling2D(),

        layers.Flatten(),
        layers.Dropout(0.35),
        layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)),

        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy']
    )

    return model

# ====================================================================
# 3. وظائف تحميل النموذج والتصنيفات
# ====================================================================

@st.cache_resource
def load_keras_model():
    """تحميل نموذج Keras من ملف .keras"""
    if os.path.exists(MODEL_PATH):
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            st.success("✅ تم تحميل النموذج بنجاح!")
            return model
        except Exception as e:
            st.error(f"❌ خطأ في تحميل النموذج: {e}")
            return None
    else:
        st.warning("⚠️ ملف النموذج غير موجود. يمكنك إنشاء نموذج جديد للتجربة.")
        return build_improved_generalization_model()

@st.cache_data
def load_labels():
    """تحميل أسماء الفئات من labels.json"""
    if os.path.exists(LABELS_PATH):
        try:
            with open(LABELS_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"❌ خطأ في تحميل التصنيفات: {e}")
            return None
    else:
        st.warning("⚠️ ملف labels.json غير موجود. سيتم إنشاء أسماء افتراضية.")
        return [f"Class_{i}" for i in range(NUM_CLASSES)]

# ====================================================================
# 4. وظيفة التنبؤ
# ====================================================================

def predict_uploaded_image(image, model, class_names):
    """
    معالجة الصورة وإجراء التنبؤ باستخدام النموذج.
    """
    img = image.convert("RGB").resize(IMAGE_SIZE)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_index]
    confidence = np.max(predictions[0]) * 100

    return predicted_class, confidence

# ====================================================================
# 5. واجهة Streamlit
# ====================================================================

st.title("تطبيق my_model.keras مع تحسين التعميم (Generalization)")

st.markdown("""
هذا التطبيق يوضح كيفية استخدام نموذج CNN محسّن للتعميم.
يمكن للنموذج التعامل مع البيانات الجديدة وغير المألوفة (Unseen Data) بشكل أفضل
بفضل استخدام تقنيات مثل Dropout و L2 Regularization وطبقات CNN متعددة.
""")

# عرض نقاط مهمة حول التعميم
st.subheader("نقاط مهمة حول التعميم (Generalization):")
st.markdown("""
- **Dropout Layers:** تقلل الاعتماد على الخلايا الفردية، مما يجعل النموذج أقل عرضة للفرط في التخصيص (Overfitting) للبيانات التدريبية.

- **L2 Regularization:** تضبط الأوزان الكبيرة في الطبقات الكثيفة، مما يزيد من قدرة النموذج على التعامل مع بيانات جديدة.

- **Rescaling & Normalization:** تساعد النموذج على التكيف مع نطاقات ألوان مختلفة.

- **Data Augmentation (تدريب على بيانات متنوعة):** عند تدريب النموذج، إضافة تقنيات مثل التدوير، التكبير، والانعكاس ستجعل النموذج أكثر مرونة مع البيانات غير المألوفة.

- **اختبار على بيانات غير مألوفة (Unseen Data):** لتقييم قدرة النموذج على التعميم، يجب دائمًا اختبار النموذج على مجموعة بيانات لم يرها أثناء التدريب.
""")

# تحميل النموذج والتصنيفات
model = load_keras_model()
class_names = load_labels()

if model and class_names:
    st.info(f"النموذج جاهز للعمل ويحتوي على {len(class_names)} فئة.")
    
    uploaded_file = st.file_uploader("قم برفع صورة لاختبار التنبؤ:", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption='الصورة المرفوعة', width=300)
        
        if st.button("توقع باستخدام my_model.keras"):
            with st.spinner('جاري التنبؤ...'):
                predicted_class, confidence = predict_uploaded_image(image, model, class_names)
            
            st.success(f"النتيجة من my_model.keras:")
            st.metric(label="الفئة المتوقعة", value=predicted_class, delta=f"بثقة {confidence:.2f}%")
