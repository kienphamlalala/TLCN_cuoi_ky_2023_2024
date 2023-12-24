import streamlit as st
import numpy as np
import cv2 
import joblib
import base64
from tensorflow.keras.models import load_model
import mediapipe as mp
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPooling1D
from tensorflow.keras.callbacks import TensorBoard


st.set_page_config(
    page_title="Hand Gesture Recognition",
    page_icon="🧊",
    initial_sidebar_state="expanded"
)

with open('./css/home.css') as f:
    css = f.read()

st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

colors=[(245,117,16),(117,245,16),(16,117,245),(16,245,235),(255, 0, 0), (0, 255, 0), (0, 0, 255), (128, 128, 0),(0, 0, 255), (128, 128, 0)]
def pro_viz(res,actions,input_frame,colors):
    output_frame=input_frame.copy()
    for num,prob in enumerate(res):
        if num < len(colors):
            cv2.rectangle(output_frame,(0,60+num*40),(int(prob*100),90+num*40),colors[num],-1)
            cv2.putText(output_frame,actions[num],(0,85+num*40),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)

    return output_frame

mp_holistic=mp.solutions.holistic #model holistic
mp_drawing=mp.solutions.drawing_utils #vẽ các utilities

def draw_landmark(image,result):
    mp_drawing.draw_landmarks(image,result.right_hand_landmarks,mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image,result.left_hand_landmarks,mp_holistic.HAND_CONNECTIONS)

def draw_styled_landmark(image,result):
    mp_drawing.draw_landmarks(image,result.right_hand_landmarks,
                              mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80,110,10),thickness=1,circle_radius=3))
    mp_drawing.draw_landmarks(image,result.left_hand_landmarks,
                          
                              mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80,110,10),thickness=1,circle_radius=3))
# Đoạn mã nhận diện cử chỉ tay và các hàm khác ở đây...
def mediapipe_detection(image,model):
    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #chuyển màu sắc TỪ BGR TO RGB
    image.flags.writeable=False
    result=model.process(image) #model xử lý + xác định
    image.flags.writeable=True
    image=cv2.cvtColor(image, cv2.COLOR_RGB2BGR) #CHUYỂN NGƯỢC LẠI
    return image,result

def extract_result(result):
    if  result.left_hand_landmarks:
    # Trích xuất tọa độ x, y, z từ result.left_hand_landmarks
        lh = np.array([[res.x, res.y, res.z] for res in result.left_hand_landmarks.landmark]).flatten()
        # flatten flatten() để biến mảng 2D này thành một mảng 1D.
    else:
        # Nếu result hoặc result.left_hand_landmarks không có giá trị, trả về mảng chứa 0
        lh = np.zeros(21 * 3)
    if  result.right_hand_landmarks:
        # Trích xuất tọa độ x, y, z từ result.left_hand_landmarks

        rh = np.array([[res.x, res.y, res.z] for res in result.right_hand_landmarks.landmark]).flatten()
    else:
        # Nếu result hoặc result.left_hand_landmarks không có giá trị, trả về mảng chứa 0
        rh = np.zeros(21 * 3)
    return np.concatenate([lh,rh])

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: contain;
    }
     .css-vk3wp9 e1fqkh3o11 {
        background-color: green;
    }
    .css-10trblm {
    color:black;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)
# set_background('assets/wall.png')




model = load_model("../models/best_CNN_LSTM.h5")

# /////////////////////////////////////////
def adjust_gamma(image, gamma=1.0):
    try:
        M, N = image.shape
        img_out = np.zeros((M, N), np.uint8)
    except:
        M, N, C = image.shape
        img_out = np.zeros((M, N, C), np.uint8)

    L = 256  # Số mức xám

    c = np.power(L - 1, 1 - gamma)

    for x in range(0, M):
        for y in range(0, N):
            r = image[x, y]
            s = c * np.power(r, gamma)
            img_out[x, y] = s

    return img_out


# /////////////////////////////////////////




def main():
    # Tiêu đề ứng dụng
    actions = ["A", "B", "C", "D", "E", "F", "G", "H","have a nice day","read"]
    actions = np.array(actions)
    sequence=[]
    sentence=[]

    threshold = 0.8  # Ngưỡng

    st.title("Hand Gesture Recognition")
    st.write("---")
    # Khởi tạo biến cap
    cap = cv2.VideoCapture(0)
    frame_placeholder = st.empty()
    stop_button_pressed = st.button("Stop")
    brightness_option = st.checkbox("Bật/Tắt Điều chỉnh chói sáng")
    # Loop chính của ứng dụng
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened() and not stop_button_pressed:


            # Đọc phản hồi từ camera
            ret, frame = cap.read()
            if not ret:
                print('No frames grabbed!')
                break
            # Kiểm tra xem frame có rỗng hay không
            if brightness_option:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.equalizeHist(frame)
                # frame = adjust_gamma(frame, gamma=5.0)
                # frame= adjust_log(frame)
            # Xử lý cử chỉ và vẽ kết quả
            image, result = mediapipe_detection(frame, holistic)
            if image is None or result is None:
                continue  # Bỏ qua lần lặp hiện tại và tiếp tục vòng lặp

            draw_styled_landmark(image, result)
            # Logic áp dụng
            keypoints = extract_result(result)
            sequence.insert(0, keypoints)
            sequence = sequence[:30]

            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                # st.write(actions[np.argmax(res)])

            # Vẽ kết quả và thông báo
                if res[np.argmax(res)] > threshold:
                    # st.write(res[np.argmax(res)])
                    if len(sentence) > 0:
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                    else:
                        # st.write(np.argmax(res))
                        # st.write(actions[np.argmax(res)])
                        sentence.append(actions[np.argmax(res)])

            # if len(sentence) > 5:
            #     sentence = sentence[-5:]

            # Vẽ kết quả lên ảnh
                        
                frame_placeholder.image(pro_viz(res, actions, image, colors), channels="BGR", use_column_width=True)

# Chạy ứng dụng Streamlit
if __name__ == "__main__":
    main()




