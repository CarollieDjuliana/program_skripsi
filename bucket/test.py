import streamlit as st

# Inisialisasi status halaman aktif
status_halaman = 1

# Button untuk membuka Halaman 1
button_halaman_1 = st.sidebar.button("Halaman 1", key='Halaman 1', help='Buka Halaman 1')

# Button untuk membuka Halaman 2
button_halaman_2 = st.sidebar.button("Halaman 2", key='Halaman 2', help='Buka Halaman 2')

# Tampilkan konten berdasarkan halaman yang aktif
if button_halaman_1:
    status_halaman = 1
    button_halaman_2 = False
elif button_halaman_2:
    status_halaman = 2
    button_halaman_1 = False

# Menambahkan CSS custom untuk efek "aktif" pada button yang ditekan

# Tampilkan konten berdasarkan halaman yang aktif
if status_halaman == 1:
    st.markdown(
        """
        <style>
        .css-1v4eu6n {
            background-color: white !important; /* Warna background saat button aktif */
            color: black !important; /* Warna teks saat button aktif */
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.write("Konten Halaman 1")
elif status_halaman == 2:
    st.write("Konten Halaman 2")
