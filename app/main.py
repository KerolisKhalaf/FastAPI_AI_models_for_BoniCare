from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
import pickle
import tensorflow as tf
from PIL import Image
import io

app = FastAPI(title="BoniCare AI Service")

# ---------------------------
# تعريف نماذج البيانات (Schemas)
# ---------------------------
class LowerBackInput(BaseModel):
    features: List[float]

# ---------------------------
# تحميل الموديلات
# ---------------------------
try:
    # تحميل موديل الظهر
    with open("models/lower_back_model.pkl", "rb") as f:
        lower_back_model = pickle.load(f)

    # تحميل موديل الكسور
    bone_model = tf.keras.models.load_model("models/model_bone_fracture_fixed.keras")
    print("Models loaded successfully! 🚀")
except Exception as e:
    print(f"Error loading models: {e}")

# ---------------------------
# الصفحة الرئيسية (Health Check)
# ---------------------------
@app.get("/")
def home():
    return {"status": "online", "message": "BoniCare AI Service Running 🚀"}

# ---------------------------
# توقع مشاكل أسفل الظهر
# ---------------------------
@app.post("/lower-back")
def predict_lower_back(data: LowerBackInput):
    # التأكد من وجود 12 ميزة
    if len(data.features) != 12:
        raise HTTPException(status_code=400, detail="يجب إرسال 12 ميزة")

    try:
        # تحويل لـ numpy array وتغيير النوع لـ float64 عشان يطابق تدريب الموديل
        feat_array = np.array([data.features], dtype=np.float64)
        
        # التوقع
        prediction = int(lower_back_model.predict(feat_array)[0])
        
        # حساب احتمالية الثقة
        probabilities = lower_back_model.predict_proba(feat_array)[0]
        confidence = float(probabilities[prediction])

        # الترتيب ده هو اللي الموديل اتعلم عليه في كود التدريب
        labels = ["Normal", "Herniated Disc", "Spinal Stenosis"]

        return {
            "prediction": prediction,
            "label": labels[prediction],
            "confidence": confidence,
            "probabilities": probabilities.tolist(),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"حدث خطأ في الموديل: {str(e)}")

# ---------------------------
# توقع كسر العظام (صور)
# ---------------------------
@app.post("/bone-fracture")
async def predict_bone_fracture(file: UploadFile = File(...)):
    # 1. التحقق من نوع الملف المرسل
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="يجب رفع صورة صحيحة بصيغة JPG أو PNG")

    try:
        # 2. قراءة الصورة ومعالجتها بنفس طريقة تدريب الموديل
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # تحجيم الصورة لـ 224x224 كما تم في كود MobileNetV2
        image = image.resize((224, 224))

        # تحويل الصورة لـ Array وتقسيمها على 255 (Normalization)
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # 3. التوقع من موديل TensorFlow
        raw_pred = bone_model.predict(img_array)[0][0]

        # 4. الربط الصحيح بناءً على ترتيب المجلدات الأبجدي في التدريب:
        # 0 = Fractured (كسر)
        # 1 = Not Fractured (سليم)
        
        if raw_pred > 0.5:
            # القيمة قريبة من 1 إذن الحالة سليمة
            label = "NOT FRACTURED"
            confidence = raw_pred * 100
        else:
            # القيمة قريبة من 0 إذن يوجد كسر
            label = "FRACTURED"
            # بنطرح من 1 عشان نجيب نسبة اليقين بوجود الكسر
            confidence = (1 - raw_pred) * 100

        return {
            "prediction_score": float(raw_pred),
            "label": label,
            "confidence": f"{confidence:.2f}%",
            "status": "success"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطأ في معالجة الصورة: {str(e)}")