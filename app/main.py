from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response, JSONResponse, FileResponse
from .ml_model import generate_character_image
import os

app = FastAPI(
    title="캐릭터 변환 API",
    description="얼굴 이미지를 업로드하면 간단한 캐릭터 스타일 이미지로 변환합니다.",
    version="1.0.0"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@app.get("/")
def read_root():
    return FileResponse(os.path.join(BASE_DIR, "static", "index.html"))

@app.post("/generate-character/")
async def generate_character(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다.")
    
    try:
        # 업로드된 이미지 읽기
        image_bytes = await file.read()
        
        # ML 모델 처리를 통해 캐릭터 이미지 바이트 얻기
        result_bytes = generate_character_image(image_bytes)
        
        # 이미지로 반환
        return Response(content=result_bytes, media_type="image/jpeg")
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"서버 내부 오류가 발생했습니다: {str(e)}")
