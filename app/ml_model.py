import cv2
import numpy as np

def generate_character_image(image_bytes: bytes) -> bytes:
    # 1. 파일 바이트를 numpy 배열로 변환 후 이미지로 디코딩
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("이미지를 디코딩할 수 없습니다.")

    # [최적화] 이미지가 너무 클 경우 처리 속도를 위해 리사이징 (최대 긴 변 800px 기준)
    h, w = img.shape[:2]
    max_dim = 800
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    # --- 길거리 캐리커처 (평면 마커펜 느낌) 알고리즘 ---

    # 1. 또렷한 펜/마커스케치 윤곽선 생성
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)  # 잔선 제거 방지용 블러
    
    # 어댑티브 쓰레시홀드로 특징을 잘 잡는 강한 선 생성
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 4)
    # 지저분한 노이즈 선 다시 한 번 다듬기
    edges = cv2.medianBlur(edges, 3)

    # 2. 입체감(Shading) 완벽 제거: 색상 평탄화 (마커 채색 느낌)
    # 강력한 양방향 필터를 반복 적용하여 그라데이션(음영)을 최대한 민무늬로 밀어버림
    color = cv2.bilateralFilter(img, 9, 300, 300)
    for _ in range(2):
        color = cv2.bilateralFilter(color, 9, 300, 300)
        
    # 포스터 리제이션(Posterization): 색 공간을 64단위 덩어리로 끊어서 입체감을 박살냄.
    # 그라데이션이 사라지고 2D 카툰처럼 평면적인 면으로 구성됩니다.
    color = (color // 64) * 64 + 32  
    
    # 3. 색상 면 위에 검은 라인 드로잉 덮어씌우기
    caricature = cv2.bitwise_and(color, color, mask=edges)

    # 4. 마커펜 채색의 번짐 효과를 살짝 더해 손그림 질감 유도
    caricature = cv2.edgePreservingFilter(caricature, flags=1, sigma_s=20, sigma_r=0.2)

    # 처리된 이미지를 다시 byte로 인코딩하여 반환
    success, encoded_image = cv2.imencode('.jpg', caricature, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not success:
        raise ValueError("결과 이미지를 인코딩하는데 실패했습니다.")

    return encoded_image.tobytes()
