# Laporan Proyek Machine Learning Terapan - Sitti Khairunnisak
MEMBUAT MODEL SISTEM REKOMENDASI BUKU DENGAN METODE COLLABORATIVE FILTERING

## Domain Proyek
Buku merupakan informasi segala kebutuhan yang diperlukan, dimulai dari iptek, seni budaya, ekonomi, politik, sosial dan pertahanan keamanan dan lain-lain. Upaya membaca buku membuka wawasan dunia intelek sehingga dapat mengubah masa depan serta mencerdaskan akal, pikiran dan iman.Dengan membaca buku, selain pengetahuan akan semakin bertambah, pribadi akan semakin   kaya, yang kesemuannya jelas akan menurunkan efek negatif terhadap anak-anak, yakni kenakalan. Sedangkan anak yang tidak terbina minat bacanya sejak  dini  akan menghadapi  peluang  yang  semakin kecil untuk mengembangkan pengetahuan setinggi-tingginya. Namun berdasarkan laporan _Bank Dunia_, Indonesia merupakan negara yang memiliki minat baca sangat rendah. Hal tersebut sungguh disayangkan, mengingat sebagai negara besar, Indonesia memiliki potensi besar untuk menjadi negara yang unggul.[1]

Sistem rekomendasi sendiri telah digunakan secara luas oleh hampir semua area bisnis dimana seorang konsumen memerlukan informasi untuk membuat suatu keputusan. Terdapat dua pendekatan  yang umumnya digunakan dalam membuat sitem rekomendasi, yaitu _content based filtering_ dan _collaborative filtering_. _Content based filtering_ merupakan metode yang bekerja dengan  mencari kedekatan suatu item yang akan direkomendasikan ke _user_ dengan _items_ yang  telah  diambil  oleh pengguna sebelumnya berdasarkan kemiripan antar kontennya.Namun, sistem  rekomendasi berbasis  konten  ini masih  memiliki  kelemahan,  yaitu  karena semua informasi dipilih dan direkomendasikan berdasarkan   konten,maka    pengguna    tidak    mendapatkan rekomendasi pada jenis konten yang berbeda. Selain itu, sistem rekomendasi ini kurang efektif untuk pengguna pemula, karena  pengguna yang masih pemula tidak mendapat masukan dari pengguna sebelumnya. (Li, 2002) 
Pendekatan  lain  untuk  menutup kelemahan  dari _content  based  filtering_ dikembangkan, yaitu _collaborative filtering_. Sistem _collaborative filtering_ adalah metode yang digunakan untuk memprediksi kegunaan item berdasarkan penilaian pengguna sebelumnya. _Collaborative Filtering_ dapat digunakan untuk membuat sistem rekomendasi, akan tetapi perhitungan dalam algoritma sangat bergantung pada hasil rekomendasi. Seperti halnya skenario yang digunakan dalam perhitungan _similarity_, antara metode _pearson correlation_ dan _adjusted cosine similarity_ memberikan hasil yang berbeda. [2]

## Business Understanding
Dampak positif dari sistem rekomendasi buku untuk pembelajaran adalah
1.Pembelajaran yang Ditingkatkan: Sistem rekomendasi buku dapat memberikan rekomendasi yang _dipersonalisasi_ berdasarkan minat dan preferensi pengguna. Hal ini dapat membantu pelajar menemukan sumber daya yang relevan dan berkualitas tinggi pada pembelajaran mesin, sehingga menghasilkan pengalaman pembelajaran yang lebih efektif dan efisien.
2.Peningkatan Keterlibatan: Dengan menyarankan buku yang sesuai dengan minat pengguna, sistem rekomendasi dapat meningkatkan keterlibatan dan motivasi untuk mengeksplorasi dan mempelajari lebih lanjut tentang pembelajaran mesin. Hal ini dapat mengarah pada pemahaman dan penguasaan subjek yang lebih dalam
3.Menghemat waktu: Dengan sistem rekomendasi buku, pengguna dapat dengan cepat menemukan buku yang relevan dengan kebutuhan dan minat spesifik mereka. Ini menghemat waktu dibandingkan dengan mencari buku secara manual atau mengandalkan rekomendasi umum
4.Menemukan Perspektif Baru: Sistem rekomendasi dapat memperkenalkan pengguna pada buku-buku yang mungkin belum mereka temukan sendiri. Hal ini dapat memaparkan mereka pada perspektif, pendekatan, dan penulis berbeda di bidang pembelajaran mesin, sehingga memperluas pengetahuan dan pemahaman mereka
5.Beradaptasi dengan Gaya Belajar Individu: Sistem rekomendasi buku dapat mempertimbangkan gaya belajar pengguna dan merekomendasikan buku yang selaras dengan cara belajar pilihan. Pendekatan yang dipersonalisasi ini dapat meningkatkan pengalaman belajar dan memenuhi kebutuhan individu.

### Problem Statements
Permasalahan yang ada dalam proyek sistem Rekomendasi Buku ini adalah
- bagaimana proses sistim rekomedasi buku dengan _recommedeernet_?
- menggunakan metode apa dalam menentukan sistim rekomendasi?

### Goals
Proses penentuan rekomendasi pada sistim rekomendasi seperti yang _diimplementasikan_ dalam program _RecommenderNet_ dapat dilakukan dengan beberapa langkah, yaitu:
Input Data: Sistim data input berupa _user ID_ dan _book ID_ yang ingin direkomendasikan.
_Embedding_: _User ID_ dan _book ID_ diubah menjadi vektor _embedding_ yang merepresentasikan karakteristik dari _user_ dan _book_ tersebut. Pada program _RecommenderNet_, _embedding_ dilakukan menggunakan layer-layer _embedding_ yang telah didefinisikan.
_Dot Product_: Vektor _embedding user_ dan _book_ dihitung _dot product-nya_ untuk menghasilkan nilai rekomendasi.
Aktivasi: Nilai rekomendasi yang dihasilkan menggunakan fungsi aktivasi _sigmoid_ untuk menghasilkan nilai rekomendasi akhir.
_Output_: Sistem mengeluarkan rekomendasi berupa _book ID_ yang memiliki nilai rekomendasi tertinggi.

-menggunakan metode _Collaborative filtering_

## Data Understanding
Data yang digunakan _mengimport_ dari _kaggle_ , (https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset)
Dengan empat file, yaitu books, rating, user dan satu file png dan jumlah dari masing-masing file adalah
Jumlah data buku:  271360
Jumlah data penilaian :  340556
Jumlah data pengguna:  166 .

### Variabel-variabel pada cats and dog dataset adalah sebagai berikut:
Variabel dalam kumpulan data 
- _Book_ : kumpulan macam-macam buku, penulis buku, penerbit buku
- _User_ : merupaka data pengguna
- _Rating_ : penilaian buku dari berbagai pembaca 

## Data Preparation
Mengatasi _missing value_ dengan cara mengecek _datafarme_ apakah ada _missing_ atau tidak, dan jika terdapat _missing_ maka membersihkan _missing value_ dengan menggunakan fungsi _dropna_, setelah itu cek kembali dengan _clean_ .
Melakukan pengurutan buku berdasarkan ISBN kemudian memasukkannya ke dalam variabel _fix_ buku dan cek jumlahnya. 
Lalu membuat variabel _preparation_ dan membuang data _duplikat_. Selanjutnya, kita perlu melakukan konversi data series menjadi list. Dalam hal ini, kita menggunakan fungsi _tolist_ dari _library numpy_. terakhir membuat _dictionary_ untuk data.

## Modeling
Model ini menggunakan penyematan untuk mewakili pengguna dan buku, lalu menghitung produk titik penyematan pengguna dan buku untuk menghasilkan skor rekomendasi. Model ini juga mencakup istilah bias bagi pengguna dan penyematan buku. 
Metode call() kelas mengambil data masukan dan mengembalikan skor rekomendasi setelah menerapkan fungsi aktivasi sigmoid.
Kelas RekomendasirNet memiliki empat lapisan:
User_embedding: lapisan penyematan untuk ID pengguna.
User_bias: lapisan penyematan untuk istilah bias pengguna.
Book_embedding: lapisan penyematan untuk ID buku.
Book_bias: lapisan penyematan untuk istilah bias buku.
Metode call() kelas mengambil data masukan, yang terdiri dari dua kolom: ID pengguna dan ID buku. Metode ini kemudian mengambil penyematan pengguna dan buku serta istilah bias dari lapisan yang sesuai, menghitung produk titik penyematan pengguna dan buku, menambahkan istilah bias pengguna dan buku, dan menerapkan fungsi aktivasi sigmoid untuk menghasilkan skor rekomendasi.
Tujuan dari program ini adalah untuk mendefinisikan model jaringan saraf untuk sistem rekomendasi yang dapat menghasilkan rekomendasi berdasarkan ID pengguna dan buku. Performa spesifik model bergantung pada kumpulan data dan hyperparameter spesifik yang digunakan untuk melatih model.

## Evaluation
Fungsi plt.plot() digunakan untuk memplot nilai RMSE untuk kumpulan data pelatihan dan pengujian, yang disimpan dalam kamus history.history masing-masing di bawah kunci 'root_mean_squared_error' dan 'val_root_mean_squared_error'. 
Fungsi plt.title(), plt.ylabel(), dan plt.xlabel() digunakan untuk menambahkan judul dan label sumbu ke plot. 
Terakhir, fungsi plt.legend() digunakan untuk menambahkan legenda ke plot yang menunjukkan baris mana yang sesuai dengan data pelatihan dan mana yang sesuai dengan data pengujian.
Hasil yang didapatkan untuk val_root_mean_squared_error adalah 0.4354 dan aroot_mean_squared_error'0.4347.
Untuk _loss_ pelatihan 0.6836  dan _val_loss_ 0.6840.

Referensi: [1.] Djamal,A Rhamadanus. Maharani, Warih dan Kurniati, Angelina Prima (2010). Analisis dan Implementasi Metode Item-Based Clustering Hybrid Pada Recomender Sytem 
[2] Li, Qing  and  Kim, Byeong  Man  2002. An Approach for Combining Content-based  and  Collaborative Filters. Departement of Computer Sciences,Kumoh National Institute of Technology


**---Ini adalah bagian akhir laporan---**
