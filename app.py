import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import json
import os

# ====================================================================
# 1. تعريف مسارات الملفات والثوابت
# ====================================================================

# يجب التأكد أن الملف موجود في نفس مسار هذا الكود (app.py)
MODEL_PATH = 'my_model.keras' 
LABELS_PATH = 'labels.json'
# حجم الإدخال المتوقع للنموذج، وهو 128x128 بكسل (3 قنوات ألوان)
IMAGE_SIZE = (128, 128)  

# ====================================================================
# 2. وظيفة تحميل النموذج (نقطة الربط الرئيسية)
# ====================================================================

# الديكور @st.cache_resource هو ما يربط ملف my_model.keras بالتطبيق بشكل فعال
# يضمن هذا التحميل مرة واحدة فقط، حتى لو تفاعل المستخدم مع التطبيق آلاف المرات.
@st.cache_resource
def load_keras_model():
    """تحميل نموذج Keras من my_model.keras."""
    try:
        # الدالة الأساسية لتحميل النموذج
        model = tf.keras.models.load_model(MODEL_PATH)
        st.success("✅ تم تحميل النموذج بنجاح!")
        return model
    except Exception as e:
        st.error(f"❌ خطأ حرج في تحميل النموذج: {e}")
        return None

@st.cache_data
def load_labels():
    """تحميل أسماء الفئات من labels.json."""
    try:
        with open(LABELS_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"❌ خطأ في تحميل التصنيفات: {e}")
        return None

# ====================================================================
# 3. وظيفة التنبؤ (استخدام النموذج المحمّل)
# ====================================================================

def predict_uploaded_image(image, model, class_names):
    """
    تقوم بمعالجة الصورة وإجراء التنبؤ.
    يتم تمرير النموذج المحمّل (model) إلى هذه الدالة.
    """
    
    # تحجيم الصورة
    img = image.convert("RGB").resize(IMAGE_SIZE)
    
    # تحويل إلى مصفوفة وإضافة بعد الدفعة
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # إجراء التنبؤ باستخدام النموذج المرتبط
    predictions = model.predict(img_array)
    
    # استخراج النتيجة
    predicted_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_index]
    confidence = np.max(predictions[0]) * 100
    
    return predicted_class, confidence

# ====================================================================
# 4. تشغيل التطبيق (للتجربة والربط)
# ====================================================================

st.title("تجربة ربط my_model.keras")

# يتم تحميل النموذج هنا. إذا نجح التحميل، model لن تكون None
model = load_keras_model()
class_names = load_labels()

if model and class_names:
    st.info(f"النموذج جاهز للعمل. يحتوي على {len(class_names)} فئة.")
    
    uploaded_file = st.file_uploader("قم برفع صورة لاختبار الربط:", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='الصورة المرفوعة', width=300)
        
        if st.button("توقع باستخدام my_model.keras"):
            with st.spinner('جاري التنبؤ...'):
                predicted_class, confidence = predict_uploaded_image(image, model, class_names)
            
            # عرض النتيجة باللغة العربية
            st.success(f"النتيجة من my_model.keras:")
            st.metric(label="الفئة المتوقعة", value=predicted_class, delta=f"بثقة {confidence:.2f}%")