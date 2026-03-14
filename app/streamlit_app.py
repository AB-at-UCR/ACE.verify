import streamlit as st
import os
import shutil
import services

temp_dir_path = 'app/temp'
if not os.path.exists(temp_dir_path):
  os.makedirs(temp_dir_path)
else: 
  shutil.rmtree(temp_dir_path)
  os.makedirs(temp_dir_path)


#st.set_page_config(layout="wide")
st.title("ACEVerify")
st.header("a deepfake detection tool")
st.divider()


# if st.button("Send balloons!"):
#   st.balloons()


uploaded_vid = st.file_uploader(
  "Upload video", type="mp4"
)

if uploaded_vid:
  col1, col2 = st.columns(2)
  raw_filename = uploaded_vid.name

  with col1:
    st.video(uploaded_vid, loop=True, autoplay=True, width=300)

  with col2:
    st.header("Original Video")
    st.markdown(
      f"""
      - Name: `{raw_filename}`
      """
    )
    st.space("small")
    run_button = st.button("Run Detection", type="primary")


  if run_button:
    st.divider()
    st.header("Detection Results")

    save_path = os.path.join(temp_dir_path, raw_filename)
    vid_saved = 0
    try:
      with open(save_path, 'wb') as f:
        f.write(uploaded_vid.getbuffer())
      vid_saved = 1
    except:
        st.error(f'Error uploading file to detection service.')

    if (vid_saved):
      file_name = raw_filename.removesuffix('.mp4')
      processed = services.process_vid(file_name, temp_dir_path)
      if processed:
        services.save_vid_to_h5(file_name, temp_dir_path)

      model = services.load_model()
      label, confidence, video, spec = services.predict(file_name, temp_dir_path, model)
      st.markdown(
        f"""
        - PREDICTION: `{label}`
        - CONFIDENCE: `{confidence:.2f}%`
        """
      )

      col3, col4, col5, col6 = st.columns(4)
      with col3:
        frame1 = os.path.join(temp_dir_path, f'{file_name}_processed_01.jpg')
        st.image(frame1, "frame 1")
        frame5 = os.path.join(temp_dir_path, f'{file_name}_processed_05.jpg')
        st.image(frame5, "frame 5")
        frame9 = os.path.join(temp_dir_path, f'{file_name}_processed_09.jpg')
        st.image(frame9, "frame 9")
        frame13 = os.path.join(temp_dir_path, f'{file_name}_processed_13.jpg')
        st.image(frame13, "frame 13")
        
      with col4:
        frame2 = os.path.join(temp_dir_path, f'{file_name}_processed_02.jpg')
        st.image(frame2, "frame 2")
        frame6 = os.path.join(temp_dir_path, f'{file_name}_processed_06.jpg')
        st.image(frame6, "frame 6")
        frame10 = os.path.join(temp_dir_path, f'{file_name}_processed_10.jpg')
        st.image(frame10, "frame 10")
        frame14 = os.path.join(temp_dir_path, f'{file_name}_processed_14.jpg')
        st.image(frame14, "frame 14")

      with col5:
        frame3 = os.path.join(temp_dir_path, f'{file_name}_processed_03.jpg')
        st.image(frame3, "frame 3")
        frame7 = os.path.join(temp_dir_path, f'{file_name}_processed_07.jpg')
        st.image(frame7, "frame 7")
        frame11 = os.path.join(temp_dir_path, f'{file_name}_processed_11.jpg')
        st.image(frame11, "frame 11")
        frame15 = os.path.join(temp_dir_path, f'{file_name}_processed_15.jpg')
        st.image(frame15, "frame 15")

      with col6:
        frame4 = os.path.join(temp_dir_path, f'{file_name}_processed_04.jpg')
        st.image(frame4, "frame 4")
        frame8 = os.path.join(temp_dir_path, f'{file_name}_processed_08.jpg')
        st.image(frame8, "frame 8")
        frame12 = os.path.join(temp_dir_path, f'{file_name}_processed_12.jpg')
        st.image(frame12, "frame 12")
        frame16 = os.path.join(temp_dir_path, f'{file_name}_processed_16.jpg')
        st.image(frame16, "frame 16")

      
      col7, col8 = st.columns(2)
      with col7:
        services.return_attention_map(model, video, 8)
      with col8:
        services.return_spec_vis(spec)
