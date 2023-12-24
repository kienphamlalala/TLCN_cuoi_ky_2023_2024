import streamlit as st
from streamlit_lottie import st_lottie
import requests
import base64 


st.set_page_config(
    page_title="Summary",
    page_icon="👋",
    layout="wide"
)

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
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)
set_background('assets/cat.png')


st.title("Summary")


with st.container():
    st.write("---")
    left_column,right_column=st.columns(2)
    with left_column:
        st.header("Kiến trúc model: CNN + LSTM")
        st.image("assets/model.png")
    with right_column:
        st.header("Biểu đồ mất mát giữa tập train và test:")
        st.image("assets/loss.jpg")

st.markdown("""
    <style>
    .css-18ni7ap{
opacity:0.1;
}
        /* Đổi màu chữ của tiêu đề thành màu đỏ */
        .css-10trblm{
            color: white;
        }

        /* Đổi màu chữ của nội dung thành màu xanh dương */
        .css-fg4pbf  {
            color: cyan;
        }
    </style>
""", unsafe_allow_html=True)






