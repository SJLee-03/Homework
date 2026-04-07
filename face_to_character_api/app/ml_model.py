import cv2
import numpy as np

def generate_character_image(image_bytes: bytes) -> bytes:
    # 1. 파일 바이트를 numpy 배열로 변환 후 이미지로 디코딩
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("이미지를 디코딩할 수 없습니다.")

    # 2. 아주 가벼운 캐릭터화(Cartoonize) 처리
    # 실제 딥러닝 모델(AnimeGAN 등) 대신, CPU에서도 매우 빠르게 동작하는 OpenCV 필터 사용
    
    # 엣지(윤곽선) 추출을 위해 그레이스케일 변환 및 블러
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    
    # 만화 같은 굵은 윤곽선 생성
    edges = cv2.adaptiveThreshold(
        gray, 255, 
        cv2.ADAPTIVE_THRESH_MEAN_C, 
        cv2.THRESH_BINARY, 9, 9
    )
    
    # 양방향 필터(Bilateral Filter)를 이용해 텍스처를 밀어서 부드럽게 만들고 색을 단순화
    color = cv2.bilateralFilter(img, 9, 300, 300)
    
    # 색상 단순화된 이미지에 윤곽선 마스크 씌우기
    cartoon = cv2.bitwise_and(color, color, mask=edges)

    # 3. 처리된 이미지를 다시 byte로 인코딩하여 반환
    success, encoded_image = cv2.imencode('.jpg', cartoon)
    if not success:
        raise ValueError("결과 이미지를 인코딩하는데 실패했습니다.")

    return encoded_image.tobytes()
