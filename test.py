import cv2
import mediapipe as mp
import numpy as np
import os

# --- cau hinh ---
# ten file anh (nho de cung thu muc voi file code)
IMG_NEUTRAL = "neutral.jpg"
IMG_POINTING = "pointing.jpg"
IMG_THUMBSUP = "thumbsup.jpg"
IMG_THINKING = "thinking.jpg"

# nguong nhan dien cu chi
THUMB_TIP_Y_THRESHOLD = 0.05  # dau ngon cai phai cao hon khop ip mot ti thi moi tinh la like
FINGER_FOLD_THRESHOLD = 0.02 # check khoang cach y tuong doi de xem ngon tay co gap ko
THINKING_Y_THRESHOLD = 0.4   # dau ngon tro phai o top 40% man hinh moi tinh la dang suy nghi

# --- setup may cai cua mediapipe ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_face_detection = mp.solutions.face_detection
mp_drawing_styles = mp.solutions.drawing_styles

def load_and_resize_image(filename, target_width, target_height):
    """
    load anh va resize theo kich thuoc mong muon.
    neu khong thay anh thi tao cai placeholder den xi de bao loi.
    """
    if os.path.exists(filename):
        img = cv2.imread(filename)
        if img is None: # file co do nhung cv2 khong doc duoc
             print(f"Error: Could not decode {filename}. Using placeholder.")
             return create_placeholder(target_width, target_height, f"Error: {filename}")
        
        return cv2.resize(img, (target_width, target_height))
    else:
        print(f"Warning: {filename} not found. Using placeholder.")
        return create_placeholder(target_width, target_height, f"Missing: {filename}")

def create_placeholder(width, height, text):
    """tao cai anh nen den co chu o giua"""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    
    text_x = (width - text_size[0]) // 2
    text_y = (height + text_size[1]) // 2
    
    cv2.putText(img, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
    return img

def is_finger_folded(landmarks, tip_idx, pip_idx, wrist_idx=0):
    """
    tra ve true neu ngon tay dang gap (dau ngon tay gan co tay hon la khop pip).
    """
    wrist = landmarks[wrist_idx]
    tip = landmarks[tip_idx]
    pip = landmarks[pip_idx]
    
    dist_tip_wrist = (tip.x - wrist.x)**2 + (tip.y - wrist.y)**2
    dist_pip_wrist = (pip.x - wrist.x)**2 + (pip.y - wrist.y)**2
    
    return dist_tip_wrist < dist_pip_wrist

def is_finger_extended(landmarks, tip_idx, pip_idx, wrist_idx=0):
    """
    tra ve true neu ngon tay dang duoi thang.
    """
    return not is_finger_folded(landmarks, tip_idx, pip_idx, wrist_idx)

def is_thumb_up(landmarks):
    """
    check xem co phai dang like khong.
    - dau ngon cai phai cao hon khop ip.
    - check ca vi tri thang hang nua.
    """
    tip = landmarks[4]
    ip = landmarks[3]
    mcp = landmarks[2]
    # check x thoai mai thoi, y thi phai chat (dau ngon cao hon khop)
    return tip.y < ip.y - THUMB_TIP_Y_THRESHOLD and ip.y < mcp.y and abs(tip.x - ip.x) < 0.2

def main():
    # khoi tao cai mediapipe hands
    hands = mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
        max_num_hands=1
    )
    face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # doc thu 1 frame de lay kich thuoc
    success, frame = cap.read()
    if not success:
        print("Error: Could not read from webcam.")
        return

    h, w, _ = frame.shape
    
    # load truoc may cai anh meme (resize bang luon frame webcam)
    # luu y: neu webcam doi do phan giai thi phai fix lai, nhung thuong thi ko sao
    memes = {
        "neutral": load_and_resize_image(IMG_NEUTRAL, w, h),
        "pointing": load_and_resize_image(IMG_POINTING, w, h),
        "thumbsup": load_and_resize_image(IMG_THUMBSUP, w, h),
        "thinking": load_and_resize_image(IMG_THINKING, w, h),
    }

    current_meme_key = "neutral"

    print("Meme Mirror Active. Press 'q' or 'ESC' to exit.")

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # lat nguoc hinh lai cho giong guong
        # xu ly anh rgb
        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # xu ly nhan dien
        results = hands.process(image_rgb)

        # trang thai mac dinh
        detected_state = "neutral"

        # nhan dien khuon mat
        face_results = face_detection.process(image_rgb)
        face_bbox_rel = None # toa do tuong doi (xmin, ymin, width, height)

        if face_results.detections:
            for detection in face_results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = image.shape
                x, y, w_box, h_box = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                cv2.rectangle(image, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
                face_bbox_rel = (bboxC.xmin, bboxC.ymin, bboxC.width, bboxC.height)
                break # chi lay cai mat dau tien thoi

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # ve may cai diem tren tay
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                # lay list diem cho de xu ly
                lm = hand_landmarks.landmark

                # --- logic xu ly cu chi ---
                
                # index ngon tay:
                # cai: 1-4, tro: 5-8, giua: 9-12, ap ut: 13-16, ut: 17-20
                
                # check xem ngon tay co gap khong (tinh khoang cach toi co tay)
                # index: 8,6; giua: 12,10; ap ut: 16,14; ut: 20,18
                index_folded = is_finger_folded(lm, 8, 6)
                middle_folded = is_finger_folded(lm, 12, 10)
                ring_folded = is_finger_folded(lm, 16, 14)
                pinky_folded = is_finger_folded(lm, 20, 18)
                
                # check like: phai du ca ngon cai lan vi tri
                is_thumbs_up_gesture = is_thumb_up(lm)
                
                # check chi tay (ngon tro duoi, may ngon kia gap)
                is_pointing_gesture = not index_folded and middle_folded and ring_folded and pinky_folded
                
                # check dang nghi: chi tay + dau ngon tro gan mat
                index_in_face = False
                if face_bbox_rel:
                    fx, fy, fw, fh = face_bbox_rel
                    ix, iy = lm[8].x, lm[8].y
                    # ve dau ngon tro de debug
                    ih, iw, _ = image.shape
                    cv2.circle(image, (int(ix*iw), int(iy*ih)), 5, (255, 0, 255), -1)
                    
                    # kiem tra xem ngon tro co nam trong vung mat khong (co mo rong vung them 10-20%)
                    if (fx - 0.1) < ix < (fx + fw + 0.1) and (fy - 0.1) < iy < (fy + fh + 0.1): 
                       index_in_face = True
                
                # chot trang thai
                if is_thumbs_up_gesture and index_folded and middle_folded and ring_folded and pinky_folded:
                    detected_state = "thumbsup"
                elif is_pointing_gesture:
                    if index_in_face:
                        detected_state = "thinking"
                    else:
                        detected_state = "pointing"
                
                # hien thong tin debug len man hinh
                debug_y = 50
                cv2.putText(image, f"Index Folded: {index_folded}", (10, debug_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.putText(image, f"Thumb Up: {is_thumbs_up_gesture}", (10, debug_y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.putText(image, f"Pointing: {is_pointing_gesture}", (10, debug_y+40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.putText(image, f"Index In Face: {index_in_face}", (10, debug_y+60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # chon anh meme tuong ung
        # neu kich thuoc frame thay doi thi resize lai meme, binh thuong thi ko can
        meme_img = memes.get(detected_state, memes["neutral"])
        
        # dam bao kich thuoc khop nhau (de phong loi resize ao ma)
        if meme_img.shape != image.shape:
             meme_img = cv2.resize(meme_img, (image.shape[1], image.shape[0]))

        # ghep 2 hinh lai
        combined_display = np.hstack((image, meme_img))
        
        # them nhan de debug
        cv2.putText(combined_display, f"State: {detected_state}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Meme Mirror', combined_display)

        key = cv2.waitKey(5) & 0xFF
        if key == ord('q') or key == 27: # 27 la nut esc
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()