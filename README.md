# Laporan Proyek Machine Learning Terapan - Sitti Khairunnisak
MEMBUAT MODEL SISTEM REKOMENDASI BUKU DENGAN METODE COLLABORATIVE FILTERING

## Domain Proyek
Buku merupakan informasi segala kebutuhan yang diperlukan, dimulai dari iptek, seni budaya, ekonomi, politik, sosial dan pertahanan keamanan dan lain-lain. Upaya membaca buku membuka wawasan dunia intelek sehingga dapat mengubah masa depan serta mencerdaskan akal, pikiran dan iman.Dengan membaca buku, selain pengetahuan akan semakin bertambah, pribadi akan semakin   kaya, yang kesemuannya jelas akan menurunkan efek negatif terhadap anak-anak, yakni kenakalan. Sedangkan anak yang tidak terbina minat bacanya sejak  dini  akan menghadapi  peluang  yang  semakin kecil untuk    mengembangkan    pengetahuan setinggi-tingginya.Namun berdasarkan laporan Bank Dunia, Indonesia merupakan negara yang memiliki minat baca sangat rendah. Hal tersebut sungguh disayangkan, mengingat sebagai negara besar, Indonesia memiliki potensi besar untuk menjadi negarayang unggul.[1]

Sistem rekomendasi sendiri telah digunakan secara luas oleh hampir semua area bisnis dimana seorang konsumen memerlukan informasi untuk membuat suatu keputusan. Terdapat dua pendekatan  yang umumnya digunakan dalam membuat sitem rekomendasi, yaitu content based filtering dan collaborative filtering. Content   based filtering merupakan metode yang bekerja dengan  mencari kedekatan suatu item yang akan direkomendasikan ke user dengan items yang  telah  diambil  oleh penggunasebelumnya berdasarkan kemiripan antar kontennya.Namun, sistem  rekomendasi berbasis  konten  ini masih  memiliki  kelemahan,  yaitu  karena semua informasi dipilih dan direkomendasikan berdasarkan   konten,maka    pengguna    tidak    mendapatkan rekomendasi pada jenis konten yang berbeda.Selain itu, sistem rekomendasi ini kurang efektif untuk pengguna pemula, karena  pengguna yang masih pemula tidak mendapat masukan dari pengguna sebelumnya. (Li, 2002) 
Pendekatan  lain  untuk  menutup kelemahan  dari content  based  filtering dikembangkan, yaitu collaborative filtering. Sistem collaborative filtering adalah metode yang digunakan   untuk memprediksi kegunaan item berdasarkan penilaian pengguna sebelumnya. Collaborative Filtering dapat digunakan untuk membuat sistem rekomendasi, akan tetapi perhitungan dalam algoritma sangat bergantung pada hasil rekomendasi. Seperti halnya skenario yang digunakan dalam perhitungan similarity, antara metode pearson correlation dan adjusted cosine similarity memberikan hasil yang berbeda. [2]

## Business Understanding
Dampak positif dari sistem rekomendasi buku untuk pembelajaran adalah
1.Pembelajaran yang Ditingkatkan: Sistem rekomendasi buku dapat memberikan rekomendasi yang dipersonalisasi berdasarkan minat dan preferensi pengguna. Hal ini dapat membantu pelajar menemukan sumber daya yang relevan dan berkualitas tinggi pada pembelajaran mesin, sehingga menghasilkan pengalaman pembelajaran yang lebih efektif dan efisien.
2.Peningkatan Keterlibatan: Dengan menyarankan buku yang sesuai dengan minat pengguna, sistem rekomendasi dapat meningkatkan keterlibatan dan motivasi untuk mengeksplorasi dan mempelajari lebih lanjut tentang pembelajaran mesin. Hal ini dapat mengarah pada pemahaman dan penguasaan subjek yang lebih dalam
3.Menghemat waktu: Dengan sistem rekomendasi buku, pengguna dapat dengan cepat menemukan buku yang relevan dengan kebutuhan dan minat spesifik mereka. Ini menghemat waktu dibandingkan dengan mencari buku secara manual atau mengandalkan rekomendasi umum
4.Menemukan Perspektif Baru: Sistem rekomendasi dapat memperkenalkan pengguna pada buku-buku yang mungkin belum mereka temukan sendiri. Hal ini dapat memaparkan mereka pada perspektif, pendekatan, dan penulis berbeda di bidang pembelajaran mesin, sehingga memperluas pengetahuan dan pemahaman mereka
5.Beradaptasi dengan Gaya Belajar Individu: Sistem rekomendasi buku dapat mempertimbangkan gaya belajar pengguna dan merekomendasikan buku yang selaras dengan cara belajar pilihan mereka. Pendekatan yang dipersonalisasi ini dapat meningkatkan pengalaman belajar dan memenuhi kebutuhan individu
Penting untuk dicatat bahwa dampak positif ini didasarkan pada asumsi bahwa sistem rekomendasi buku dirancang dengan baik dan diterapkan secara efektif.

### Problem Statements
Permasalahan yang ada dalam proyekSistem Rekomendasi Buku ini adalah
- bagaimana proses sistem rekomedasi buku dengan recommedeernet?
- menggunakan metode apa dalam menentukan sistem rekomendasi?

### Goals
Proses penentuan rekomendasi pada sistem rekomendasi seperti yang diimplementasikan dalam program RecommenderNet dapat dilakukan dengan beberapa langkah, yaitu:
Input Data: Sistem menerima data input berupa user ID dan book ID yang ingin direkomendasikan.
Embedding: User ID dan book ID diubah menjadi vektor embedding yang merepresentasikan karakteristik dari user dan book tersebut. Pada program RecommenderNet, embedding dilakukan menggunakan layer-layer embedding yang telah didefinisikan.
Dot Product: Vektor embedding user dan book dihitung dot product-nya untuk menghasilkan nilai rekomendasi.
Penambahan Bias: Nilai rekomendasi yang dihasilkan dari dot product di atas ditambahkan dengan nilai bias untuk user dan book. Nilai bias ini merepresentasikan preferensi umum dari user dan popularitas umum dari book.
Aktivasi: Nilai rekomendasi yang dihasilkan dari penambahan bias di atas diaktivasi menggunakan fungsi aktivasi sigmoid untuk menghasilkan nilai rekomendasi akhir.
Output: Sistem mengeluarkan rekomendasi berupa book ID yang memiliki nilai rekomendasi tertinggi.
-menggunakan metode Collaborative filtering

## Data Understanding
Data yang digunakan _mengimport_ dari _kaggle_ , (https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset)
Dengan jumlah dari masing-masing file
Jumlah data buku:  271360
Jumlah data penilaian :  340556
Jumlah data pengguna:  166 dan satu file png.
sistem rekomendasi buku dapat dilakukan dengan beberapa langkah, yaitu
Data Input: Sistem menerima data input berupa user ID dan book ID yang ingin direkomendasikan.
User-Item Matrix: Data input diubah menjadi matriks user-item, di mana setiap baris merepresentasikan user dan setiap kolom merepresentasikan buku. Setiap entri pada matriks merepresentasikan rating yang diberikan oleh user untuk buku tersebut.
Similarity Function: Untuk melakukan rekomendasi, sistem menggunakan similarity function untuk menghitung kemiripan antara item-item pada matriks. Beberapa similarity function yang umum digunakan adalah cosine similarity dan Pearson correlation.
Model Training: Sistem melakukan training pada model dengan menggunakan data input dan similarity function yang telah ditentukan sebelumnya. Beberapa teknik yang umum digunakan untuk training model adalah collaborative filtering, content-based filtering, dan hybrid filtering.
Rekomendasi: Setelah model dilatih, sistem dapat melakukan rekomendasi dengan menghitung similarity antara item-item pada matriks dan mengeluarkan rekomendasi berupa item yang memiliki similarity tertinggi dengan item yang ingin direkomendasikan.

### Variabel-variabel pada cats and dog dataset adalah sebagai berikut:
Variabel dalam kumpulan data 
- Book : kumpulan macam-macam buku, penulis buku, penerbit buku
- User : merupaka data pengguna
- Rating : penilaian buku dari berbagai pembaca 

## Data Preparation
Data _Image generator_ digunakan untuk membuat gambar dari teks atau input data lainnya yang dapat membantu dalam pemrosesan data. _Image generator_ dapat membantu masalah pengolahan data, mampu membuat gambar tambahan dari yang sudah ada. Dengan menghasilkan gambar baru model dapat dilatih pada kumpulan data yang lebih besar dan beragam, yang dapat meningkatkan akurasinya. Langkah pertama adalah mengimpor pustaka yang diperlukan dengan cara _import_ _imagedatagenerator._
__Training set_ digunakan untuk melatih model dan mengoptimalkan parameter, sedangkan _testing set_ digunakan untuk menguji performa model yang telah dilatih pada data yang belum pernah dilihat sebelumnya. Rasio pembagian _dataset_ antara _training set_ dan _testing set_ adalah (90%:10%) menghasilkan _train_ 1821 dan hasil _validation_ 204 dari dua kelas.
_tensorflow_ untuk membuat dan melatih model.
_ImageDataGenerator_ dari _tensorflow.keras.preprocessing.image_ untuk augmentasi data dan menyiapkan generator data untuk pelatihan dan validasi dan menggunakan _ImageDataGenerator_ untuk melakukan augmentasi data pada gambar pelatihan. Beberapa augmentasi yang diterapkan meliputi _rescaling_ dengan nilai 1/255, _rotation_range_, 20 _horizontal_ dan _vertical_shearing_, 0.2 _zooming_, 0.1 
_width_shift_range_ 0.2 dan , _height_shift_range_ 0.2. Setelah itu, menggunakan _flow_from_directory_ untuk membuat generator pelatihan dan validasi. kita menentukan _class_mode_ yaitu _'categorical_'. Gambar juga diubah ukurannya menjadi 150x150 piksel menggunakan parameter target _size._
 
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
