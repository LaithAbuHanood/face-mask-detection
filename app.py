import os                      # للتعامل مع الملفات والمجلدات
import cv2                     # OpenCV لمعالجة الصور والفيديو
import numpy as np             # للتعامل مع المصفوفات
import tensorflow as tf        # مكتبة TensorFlow
from tensorflow.keras.models import load_model  # لتحميل الموديل المدرب
import requests                # لتحميل الملفات من الإنترنت
import streamlit as st         # Streamlit لبناء واجهة المستخدم
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase  # لتشغيل الكاميرا داخل Streamlit

# --------------------------
# تحميل الموديل من Hugging Face إذا لم يكن موجودًا محليًا
# --------------------------
model_path = "mask_model.keras"                 # اسم ملف الموديل محليًا
if not os.path.exists(model_path):              # إذا الملف غير موجود
    url = "https://huggingface.co/spaces/Laithhanood/mask-detection-model222/resolve/main/mask_model.keras"  # رابط الموديل
    response = requests.get(url)                # تحميل الموديل من الإنترنت
    with open(model_path, "wb") as f:           # فتح ملف جديد للكتابة
        f.write(response.content)               # حفظ الموديل على الجهاز

# تحميل الموديل بدون compile لتفادي مشاكل Optimizer
model = load_model(model_path, compile=False)  # تحميل الموديل من الملف

# --------------------------
# إعداد أسماء الفئات
# --------------------------
categories = ["with_mask", "without_mask"]      # تصنيفات المخرجات: مع/بدون كمامة

# --------------------------
# دالة تحسين التنبؤ عند وجود نظارات أو تغطية جزئية
# --------------------------
def predict_with_augmentation(img):             # دالة التنبؤ مع تحسين للتغلب على أخطاء النظارات
    augmented_images = [                        # إنشاء نسخ من الصورة للتنبؤ المتوسط
        cv2.resize(img, (128,128)),             # النسخة الأصلية بحجم 128x128
        cv2.resize(cv2.flip(img, 1), (128,128)),        # انعكاس أفقي
        cv2.resize(cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), (128,128)), # تدوير 90 درجة
    ]
    predictions = []                            # لتخزين التنبؤات لكل نسخة
    for aug_img in augmented_images:            # لكل نسخة
        aug_img = aug_img / 255.0               # تطبيع قيم الصورة بين 0 و 1
        aug_input = np.expand_dims(             # إضافة بعد batch
            aug_img.astype(np.float32), axis=0
        )
        pred = model.predict(aug_input)        # التنبؤ بالفئة
        predictions.append(pred)                # تخزين التنبؤ
    avg_pred = np.mean(predictions, axis=0)    # أخذ متوسط التنبؤات
    class_index = np.argmax(avg_pred)          # اختيار أعلى احتمال
    return categories[class_index]             # إعادة اسم الفئة

# --------------------------
# واجهة Streamlit
# --------------------------
st.title("Face Mask Detection System")           # عنوان التطبيق
st.write("Detect whether a person is wearing a face mask (improved for glasses)")  # وصف للتطبيق

# ==========================
# 📷 رفع صورة
# ==========================
st.subheader("Upload Image")                     # عنوان فرعي للقسم
uploaded_file = st.file_uploader(                # عنصر رفع صورة
    "Choose an image",                           # نص للمستخدم
    type=["jpg", "png", "jpeg"]                  # أنواع الملفات المسموحة
)

if uploaded_file is not None:                    # إذا المستخدم رفع صورة
    file_bytes = np.asarray(                     # تحويل الملف إلى array
        bytearray(uploaded_file.read()), dtype=np.uint8
    )
    img = cv2.imdecode(file_bytes, 1)            # قراءة الصورة باستخدام OpenCV

    st.image(                                    # عرض الصورة على Streamlit
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB),   # تحويل ألوان الصورة من BGR إلى RGB
        caption="Uploaded Image",               # عنوان للصورة
        width=400                                # عرض الصورة
    )

    class_label = predict_with_augmentation(img) # عمل التنبؤ باستخدام الدالة المعدلة
    st.success(f"Prediction: {class_label}")     # عرض النتيجة للمستخدم

# ==========================
# 🎥 الكاميرا المباشرة
# ==========================
st.subheader("Live Webcam Detection")            # عنوان فرعي للقسم

class VideoProcessor(VideoTransformerBase):       # كلاس لمعالجة الفيديو
    def transform(self, frame):                  # دالة معالجة كل فريم
        img = frame.to_ndarray(format="bgr24")   # تحويل الفريم إلى مصفوفة صورة

        label = predict_with_augmentation(img)   # التنبؤ بالفئة لكل فريم

        cv2.putText(                             # كتابة نتيجة التنبؤ على الفريم
            img,                                 # الصورة
            label,                               # النص المراد كتابته
            (20, 40),                             # موقع النص على الفريم
            cv2.FONT_HERSHEY_SIMPLEX,            # نوع الخط
            1,                                   # حجم الخط
            (0, 255, 0),                         # لون النص (أخضر)
            2                                    # سماكة النص
        )

        return img                               # إعادة الفريم المعدل

webrtc_streamer(                                 # تشغيل الكاميرا داخل Streamlit
    key="mask-detection",                        # مفتاح مميز للعنصر
    video_processor_factory=VideoProcessor,      # تمرير كلاس المعالجة
    media_stream_constraints={"video": True, "audio": False}  # تشغيل الفيديو فقط بدون صوت
)
