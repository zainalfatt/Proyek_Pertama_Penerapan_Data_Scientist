# Proyek Akhir: Menyelesaikan Permasalahan Perusahaan Jaya Jaya Maju

## Business Understanding
Jaya Jaya Maju merupakan salah satu perusahaan multinasional yang telah berdiri sejak tahun 2000. Ia memiliki lebih dari 1000 karyawan yang tersebar di seluruh penjuru negeri.  

### Permasalahan Bisnis
Permasalahan bisnis yang dihadapi oleh perusahaan Jaya Jaya Maju adalah kesulitan dalam mengelola karyawan yang menyebabkan berimbasnya tingginya attrition rate (rasio jumlah karyawan yang keluar dengan total karyawan keseluruhan) hingga lebih dari 10%. Hal tersebut mendorong manajer departemen HR untuk mencari tahu penyebab tingginya attrition rate sehingga dapat meminimalisasi kemungkinan terjadinya karyawan keluar.

### Cakupan Proyek
1. Menganalisis faktor penyebab tingginya attrition rate
2. Membuat model machine learning dan prediksi sederhana dan di-deploy pada streamlit
3. Membangun dashboard menggunakan Looker Studio

### Persiapan

Sumber data: https://github.com/dicodingacademy/dicoding_dataset/blob/main/employee/employee_data.csv

Setup environment:
1. Membuat environment baru bernama newenv
```
python -m venv newenv
```
2. Aktivasi environment
```
.\newenv\Scripts\activate
```
3. Menginstal package yang dibutuhkan
```
pip install -r requirements.txt
```

## Business Dashboard

<img src="image\mosaicnim-dashboard.png" alt="alt text" width="whatever" height="whatever">

berikut adalah tujuan dari dibuatnya dashboard tersebut yakni:
1. Total, Amount Stay, dan Amount Quit
    - Fungsi: Menyediakan ringkasan jumlah karyawan secara keseluruhan.
        - Total: Jumlah total karyawan.
        - Amount Stay: Jumlah karyawan yang tetap bekerja.
        - Amount Quit: Jumlah karyawan yang resign atau keluar.
2. Donut Chart (Persentase Bertahan vs Keluar)
    - Fungsi: Visualisasi persentase karyawan yang bertahan dibandingkan yang keluar.
        - Warna biru menunjukkan karyawan yang tetap.
        - Warna merah menunjukkan karyawan yang keluar.
    - Membantu melihat rasio attrition secara keseluruhan.
3. Pengaruh Work-Life Balance terhadap Keputusan Keluar
    - Fungsi: Menunjukkan hubungan antara tingkat Work-Life Balance dan keputusan karyawan untuk keluar.
        - Angka pada sumbu Y (1–4) merepresentasikan tingkat Work-Life Balance.
        - Warna biru menunjukkan karyawan yang tetap, dan merah yang keluar.
    - Bisa dilihat bahwa sebagian besar karyawan dengan WLB tinggi (nilai 3) cenderung bertahan.
4. Persentase Attrition by Department
    - Fungsi: Menunjukkan persentase attrition (pengunduran diri) berdasarkan departemen.
    - Berguna untuk mengidentifikasi departemen dengan tingkat turnover tinggi.
5. Pengaruh OverTime terhadap Attrition
    - Fungsi: Menganalisis hubungan antara lembur (OverTime) dan keputusan keluar.
    - Memberi insight bahwa lembur mungkin berkontribusi pada turnover.


Link: https://lookerstudio.google.com/reporting/2e66e277-7364-4310-850f-94980e3ae98c

## Menjalankan Sistem Machine Learning
Langkah-langkah menggunakan sistem machine learning berbasis Extra Trees Classifier	adalah sebagai berikut.

1. Membuka link 

2. Tampilan Awal
<center><img src="image\tampilan_utama.png" alt="alt text" width="whatever" height="whatever"></center>

3. Memilih "Predict Featur" pada taskbar di sisi kiri.

<center><img src="image\menu.png" alt="alt text" width="whatever" height="whatever"></center>

3. Mengisi data yang dibutuhkan. Perlu diperhatikan bahwa nilai jurusan atau Course tidak boleh 'None' serta terdapat batas minimum dan maksimum pada input numerik. Selain itu, pengguna harus menekan enter agar dapat menyimpan data numerik.

<center><img src="image\featur_predict.png" alt="alt text" width="whatever" height="whatever"></center>

4. Hasil prediksi akan tampil di bagian bawah.
<center><img src="image\hasil_predict.png" alt="alt text" width="whatever" height="whatever"></center>


```

```

## Conclusion
- Faktor yang paling memengaruhi karyawan untuk keluar adalah lembur (Overtime). Sebanyak 31,9% dari total karyawan yang lembur (307 orang) memutuskan untuk keluar (98 orang). Artinya, semakin sering karyawan lembur, semakin tinggi kemungkinan mereka mengalami burnout dan akhirnya resign.
- Sebagian besar karyawan dengan Work-Life Balance (WLB) yang baik (nilai 3) cenderung bertahan di perusahaan, sedangkan karyawan dengan WLB yang rendah (nilai 1 atau 2) memiliki proporsi keluar yang lebih tinggi. Ini menunjukkan bahwa keseimbangan kerja dan kehidupan pribadi memainkan peran penting dalam keputusan untuk bertahan.
- Karyawan yang mengalami attrition (keluar) cenderung terkonsentrasi pada departemen tertentu. Departemen Sales memiliki attrition rate tertinggi (20,69%), dibandingkan dengan Human Resources (15,79%) dan Research & Development (15,26%). Hal ini bisa disebabkan oleh tekanan target atau lingkungan kerja yang lebih kompetitif.

### Rekomendasi Action Items
Berikut beberapa rekomendasi tindakan (action items) yang dapat dilakukan oleh Jaya Jaya Maju untuk mengatasi tingginya tingkat attrition (pengunduran diri karyawan):
- Melakukan evaluasi terhadap kebijakan lembur (overtime), terutama pada divisi yang memiliki beban kerja tinggi. Perusahaan perlu meninjau apakah beban kerja masih dalam batas wajar dan memberikan kompensasi atau fleksibilitas waktu yang layak bagi karyawan yang sering lembur.
- Meningkatkan program Work-Life Balance, seperti opsi kerja fleksibel, cuti tambahan, atau kegiatan kesejahteraan (well-being). Hal ini penting untuk mempertahankan karyawan yang merasa keseimbangan hidupnya terganggu.
- Melakukan evaluasi khusus pada departemen dengan attrition rate tinggi, terutama Sales, untuk mengidentifikasi penyebab tingginya turnover. Evaluasi ini bisa mencakup beban kerja, tekanan target, gaya kepemimpinan, hingga peluang karier.
- Mengembangkan program retensi dan engagement, seperti pelatihan karier, coaching, dan pengakuan terhadap kinerja yang baik, agar karyawan merasa dihargai dan memiliki prospek jangka panjang di perusahaan.
- Mengadakan exit interview secara sistematis, guna mendapatkan wawasan langsung dari karyawan yang resign. Informasi ini bisa menjadi dasar dalam menyusun strategi HR yang lebih efektif.
