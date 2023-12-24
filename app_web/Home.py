import streamlit as st
import base64

st.set_page_config(
    page_title="Home App",
    page_icon="👋",
)

def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img = get_img_as_base64("assets/fw.jpg")

# Đổi màu tiêu đề
custom_styles = f"""
<style>
    .title-container {{
        color: #DB3E3E !important;  
        font-size: 55px;
    }}
    
    .gray-header {{
        color: #542E2E !important; 
        font-size: 30px;
    }}

    .black-subheader {{
        color: black !important;
        font-size: 25px;
    }}
</style>
"""

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
    background-image: url("https://images.unsplash.com/photo-1501426026826-31c667bdf23d");
    background-size: 180%;
    background-position: top left;
    background-repeat: no-repeat;
    background-attachment: local;
}}

[data-testid="stSidebar"] > div:first-child {{
    background-image: url("data:image/png;base64,{img}");
    background-position: center; 
    background-repeat: no-repeat;
    background-attachment: fixed;
    font-size: 40px;
}}

[data-testid="stHeader"] {{
    background: rgba(0,0,0,0);
}}

[data-testid="stToolbar"] {{
    right: 2rem;
}}
</style>
"""

# Hiển thị màu sắc tiêu đề và nền
st.markdown(custom_styles, unsafe_allow_html=True)
st.markdown(page_bg_img, unsafe_allow_html=True)

# Tiêu đề với lớp CSS
st.markdown('<h1 class="title-container">TIỂU LUẬN CHUYÊN NGÀNH</h1>', unsafe_allow_html=True)

# Header với màu xám
st.markdown('<h2 class="gray-header">TÌM HIỂU THUẬT TOÁN HỌC SÂU VÀ NHẬN DẠNG CỬ CHỈ TAY QUA CAMERA</h2>', unsafe_allow_html=True)

# Subheader với màu đen
st.markdown('<h3 class="black-subheader">GVHD: PGS.TS Hoàng Văn Dũng</h3>', unsafe_allow_html=True)

with st.container():
    st.write("---")
    left_column,right_column=st.columns(2)
    with left_column:
        st.markdown('<h4 style="color: black; font-size:27px">Sinh Viên Thực Hiện:</h4>', unsafe_allow_html=True)
        st.markdown('<h4 style="color: black; font-size:24px">20133047 - Lương Gia Huy</h4>', unsafe_allow_html=True)
        st.markdown('<h4 style="color: black; font-size:24px">20133059 - Phạm Trung Kiên</h4>', unsafe_allow_html=True)
