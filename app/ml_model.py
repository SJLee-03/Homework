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
        h, w = img.shape[:2]

    # 2. 캐리커처 변신: 과장 효과(형태 왜곡) 적용
    # 이미지 중앙부(얼굴)가 볼록하게 튀어나오는 Fisheye(어안) 렌즈 왜곡 효과
    # 카메라 매트릭스 가설 설정
    K = np.array([[w, 0, w/2],
                  [0, h, h/2],
                  [0, 0, 1]], dtype=np.float32)
    # 왜곡 계수 설정: 중앙을 확대하고 가장자리를 둥글게 밀어냄
    D = np.array([-0.08, 0.03, 0, 0], dtype=np.float32) 
    
    # 왜곡 맵 적용 (가장자리는 자연스럽게 늘리거나 반사경계로 처리)
    map1, map2 = cv2.initUndistortRectifyMap(K, D, None, K, (w,h), cv2.CV_32FC1)
    warped_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # 3. 회화적 팝아트 질감(Posterization + Stylization) 적용
    # 색상을 단순화 시켜 유화/팝아트 포스터 같은 거친 질감 느낌 유도
    quantized = warped_img // 32 * 32

    # Stylization: 수채화 물감이 번진 듯한 터치와 연필 스케치가 가미된 효과 제공
    # sigma_s: 필터 이웃 크기, sigma_r: 색상 균일도
    cartoon = cv2.stylization(quantized, sigma_s=40, sigma_r=0.3)

    # 윤곽선 일부만 아주 약하게 추출하여 그림체를 선명하게 보정 (선택 사항)
    gray = cv2.cvtColor(quantized, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 7)
    
    # stylization 결과 위에 부드러운 스케치 윤곽선을 살짝 겹침
    caricature = cv2.bitwise_and(cartoon, cartoon, mask=edges)

    # 4. 처리된 이미지를 다시 byte로 인코딩하여 반환
    success, encoded_image = cv2.imencode('.jpg', caricature, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not success:
        raise ValueError("결과 이미지를 인코딩하는데 실패했습니다.")

    return encoded_image.tobytes()
