import torch
from PIL import Image
import io

# 전역(Global) 공간에서 AI 모델을 장전(Load)하여 메모리에 올려둡니다.
# 로컬 서버 부팅 후 최초 1회 요청 시, Github에서 모델 가동에 필요한 가중치를 다운받으며 시간이 걸릴 수 있습니다.
device = "cpu"

try:
    # 전세계적으로 활발히 쓰이는 AnimeGANv2 딥러닝 모델 (사전학습 가중치: face_paint_512_v2)
    # CPU 환경에서도 적당한 속도로 동작하도록 구성합니다.
    model = torch.hub.load("bryandlee/animegan2-pytorch:main", "generator", pretrained="face_paint_512_v2", device=device)
    model.eval() # 학습이 아닌 추론(Inference) 모드로 변경

    # 이미지를 512px에 맞춰 자동으로 자르고 텐서(Tensor)로 변환한 뒤 다시 사진으로 뽑아주는 편리한 래퍼 함수
    face2paint = torch.hub.load("bryandlee/animegan2-pytorch:main", "face2paint", size=512, device=device)
except Exception as e:
    print(f"딥러닝 모델 로딩 실패!: {e}")
    model = None
    face2paint = None

def generate_character_image(image_bytes: bytes) -> bytes:
    if model is None or face2paint is None:
        raise RuntimeError("서버에서 AI 딥러닝 모델을 초기화하지 못해 작업을 수행할 수 없습니다.")

    try:
        # 사용자가 올린 바이너리 데이터를 PIL 이미지 도화지로 엽니다 (딥러닝 라이브러리의 기본 포맷)
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise ValueError("이미지를 읽어 들일 수 없습니다.")

    # 이 한 줄이 바로 진짜 AI(머신러닝) 연산이 수행되는 핵심 부분입니다.
    # 인공신경망에 원본 이미지를 통과시켜 만화풍의 새로운 이미지를 '창조'해냅니다.
    out_img = face2paint(model, img)

    # 완성된 그림을 다시 서버에서 전송할 수 있도록 바이트(Bytes) 배열로 포장합니다.
    img_byte_arr = io.BytesIO()
    out_img.save(img_byte_arr, format='JPEG', quality=95)
    
    return img_byte_arr.getvalue()
