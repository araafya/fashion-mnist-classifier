# Fashion-MNIST Classifier

Aplikasi web Streamlit untuk mengklasifikasikan gambar pakaian ke dalam 10 kelas Fashion-MNIST
(T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot)
menggunakan model CNN (TensorFlow / Keras).

Dinnetza Araafya Yudetti (5054251018). Tugas 2 Deep Learning, Machine Learning.

## Isi repository
| File | Keterangan |
|---|---|
| `streamlit_app.py` | Aplikasi Streamlit (upload gambar → prediksi kelas dan probabilitas) |
| `model.keras` | Model CNN terbaik hasil training di notebook |
| `requirements.txt` | Package yang dibutuhkan |

## Preprocessing di aplikasi
Sama seperti di notebook: gambar diubah ke grayscale, di-resize ke 28×28, lalu dinormalisasi ke 0–1.
Karena data Fashion-MNIST berupa objek terang di latar gelap, warna gambar dengan latar terang dibalik secara otomatis.

## Menjalankan secara lokal
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```
