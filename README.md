# Laporan Proyek Machine Learning Terapan - Sitti Khairunnisak
MEMBUAT MODEL SISTEM REKOMENDASI BUKU DENGAN METODE COLLABORATIVE FILTERING

## Domain Proyek
Buku merupakan informasi segala kebutuhan yang diperlukan, dimulai dari iptek, seni budaya, ekonomi, politik, sosial dan pertahanan keamanan dan lain-lain. Upaya membaca buku membuka wawasan dunia intelek sehingga dapat mengubah masa depan serta mencerdaskan akal, pikiran dan iman.Dengan membaca buku, selain pengetahuan akan semakin bertambah, pribadi akan semakin   kaya, yang kesemuannya jelas akan menurunkan efek negatif terhadap anak-anak, yakni kenakalan. Sedangkan anak yang tidak terbina minat bacanya sejak  dini  akan menghadapi  peluang  yang  semakin kecil untuk mengembangkan pengetahuan setinggi-tingginya. Namun berdasarkan laporan _Bank Dunia_, Indonesia merupakan negara yang memiliki minat baca sangat rendah. Hal tersebut sungguh disayangkan, mengingat sebagai negara besar, Indonesia memiliki potensi besar untuk menjadi negara yang unggul.[1]

Hal itu sangat berpengaruh dengan Potensi negara,terkait dengan rendahnya minat baca di Indonesia. Rendahnya minat baca masyarakat Indonesia menjadi salah satu faktor yang menyebabkan rendahnya kualitas sumber daya manusia di Indonesia. Oleh karena itu, meningkatkan minat baca masyarakat Indonesia dapat membantu meningkatkan kualitas sumber daya manusia di Indonesia dan berpotensi meningkatkan kemajuan negara.
Dengan Sistem rekomendasi buku dapat membantu mengatasi rendahnya minat baca di Indonesia dengan cara memberikan rekomendasi buku yang sesuai dengan minat atau kesukaan pembaca.
Hal ini dapat meningkatkan minat baca masyarakat Indonesia karena akan lebih tertarik untuk membaca buku yang sesuai dengan minat mereka
Selain itu, sistem rekomendasi buku juga dapat membantu para pembaca untuk lebih mudah mendapatkan informasi mengenai buku yang akan dibaca. Dengan adanya sistem rekomendasi buku, para pembaca tidak perlu lagi bingung memilih buku yang ingin dibaca dan dapat lebih mudah menemukan buku yang sesuai dengan minat mereka

Sistem rekomendasi sendiri telah digunakan secara luas oleh hampir semua area bisnis dimana seorang konsumen memerlukan informasi untuk membuat suatu keputusan. Terdapat dua pendekatan  yang umumnya digunakan dalam membuat sitem rekomendasi, yaitu _content based filtering_ dan _collaborative filtering_. _Content based filtering_ merupakan metode yang bekerja dengan  mencari kedekatan suatu item yang akan direkomendasikan ke _user_ dengan _items_ yang  telah  diambil  oleh pengguna sebelumnya berdasarkan kemiripan antar kontennya.Namun, sistem  rekomendasi berbasis  konten  ini masih  memiliki  kelemahan,  yaitu  karena semua informasi dipilih dan direkomendasikan berdasarkan   konten,maka    pengguna    tidak    mendapatkan rekomendasi pada jenis konten yang berbeda. Selain itu, sistem rekomendasi ini kurang efektif untuk pengguna pemula, karena  pengguna yang masih pemula tidak mendapat masukan dari pengguna sebelumnya. (Li, 2002) 
Pendekatan  lain  untuk  menutup kelemahan  dari _content  based  filtering_ dikembangkan, yaitu _collaborative filtering_. Sistem _collaborative filtering_ adalah metode yang digunakan untuk memprediksi kegunaan item berdasarkan penilaian pengguna sebelumnya. _Collaborative Filtering_ dapat digunakan untuk membuat sistem rekomendasi, akan tetapi perhitungan dalam algoritma sangat bergantung pada hasil rekomendasi. Seperti halnya skenario yang digunakan dalam perhitungan _similarity_, antara metode _pearson correlation_ dan _adjusted cosine similarity_ memberikan hasil yang berbeda. [2]

## Business Understanding
Dampak positif dari sistem rekomendasi buku untuk pembelajaran adalah
1.Pembelajaran meningkat dengan Sistem rekomendasi buku dapat memberikan rekomendasi yang _dipersonalisasi_ berdasarkan minat dan preferensi pengguna. Hal ini dapat membantu pelajar menemukan sumber daya yang relevan dan berkualitas tinggi sehingga menghasilkan pengalaman pembelajaran yang lebih efektif dan efisien.
2.Dengan menyarankan buku yang sesuai dengan minat pengguna, sehingga  dapat meningkatkan keterlibatan dan motivasi untuk mengeksplorasi dan mempelajari lebih lanjut tentang pembelajaran mesin. Hal ini dapat mengarah pada pemahaman dan penguasaan subjek yang lebih dalam.
3. sistem rekomendasi buku juga dapat membantu pengguna dengan cepat menemukan buku yang relevan dengan kebutuhan dan minat spesifik mereka. Ini menghemat waktu dibandingkan dengan mencari buku secara manual atau mengandalkan rekomendasi umum
4. Dapat memperkenalkan pengguna pada buku-buku yang mungkin belum mereka temukan sendiri. Hal ini dapat memaparkan mereka pada perspektif, pendekatan, dan penulis berbeda di bidang pembelajaran mesin, sehingga memperluas pengetahuan dan pemahaman 
5.Dan juga dapat mempertimbangkan gaya belajar pengguna dan merekomendasikan buku yang selaras dengan cara belajar pilihan. Pendekatan yang dipersonalisasi ini dapat meningkatkan pengalaman belajar dan memenuhi kebutuhan individu.

### Problem Statements
Permasalahan yang ada dalam proyek sistem Rekomendasi Buku ini adalah
- bagaimana proses sistem rekomedasi buku dengan _recommedernet_?
- Mengapa proyek ini mengembangkan dan menerapkan model sistem rekomendasi buku dengan menggunakan metode _Collaborative Filtering_?

### Goals
Solusi dari permasalahan tersebut adalah 
-Proses penentuan rekomendasi pada sistem rekomendasi seperti yang _diimplementasikan_ dalam program _RecommenderNet_ dapat dilakukan dengan beberapa langkah, yaitu:
Sistem data input berupa _user ID_ dan _book ID_ yang ingin direkomendasikan.
 _User ID_ dan _book ID_ diubah menjadi vektor _embedding_ yang merepresentasikan karakteristik dari _user_ dan _book_ tersebut. Pada program _RecommenderNet_, _embedding_ dilakukan menggunakan layer-layer _embedding_ yang telah didefinisikan.
Vektor _embedding user_ dan _book_ dihitung _dot product-nya_ untuk menghasilkan nilai rekomendasi.
Nilai rekomendasi yang dihasilkan menggunakan fungsi aktivasi _sigmoid_ untuk menghasilkan nilai rekomendasi akhir.
_Output_, Sistem mengeluarkan rekomendasi berupa _book ID_ yang memiliki nilai rekomendasi tertinggi.

-Proyek ini mengembangkan dan menerapkan model sistem rekomendasi buku dengan menggunakan metode _Collaborative Filtering_ karena membantu merekomendasikan buku yang sesuai dan mengatasi kesulitan dalam memilih buku. Dengan mengembangkan dan menerapkan model sistem rekomendasi buku menggunakan metode _Collaborative Filtering,_ proyek ini bertujuan untuk meningkatkan minat baca di Indonesia dengan memberikan rekomendasi buku yang relevan dan sesuai dengan minat pembaca

## Data Understanding
Data yang digunakan _mengimport_ dari _kaggle_ , (https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset)
Dengan empat file, yaitu books, ratings, users dan satu file png dan jumlah dari masing-masing file adalah
Jumlah data buku:  271360
Jumlah data penilaian :  340556
Jumlah data pengguna:  166.
Dalam pembuatan sistem rekomendasi buku, variabel-variabel dalam kumpulan data seperti Book, User, dan Rating akan digunakan untuk membangun model rekomendasi yang efektif.
Book (kumpulan macam-macam buku, penulis buku, penerbit buku), Penting untuk mengidentifikasi dan mengkategorikan buku berdasarkan genre, penulis, dan penerbit.
Rating, Penilaian buku oleh pembaca akan digunakan untuk mempelajari preferensi dan kesukaan pembaca.
User, Informasi tentang pengguna seperti yang akan digunakan untuk memahami minat dan perilaku pembaca.

### Variabel-variabel pada cats and dog dataset adalah sebagai berikut:
Variabel dalam kumpulan data sistem rekomendasi adalah
- _Book_ : kumpulan macam-macam buku, penulis buku, penerbit buku
- _User_ : merupaka data pengguna
- _Rating_ : penilaian buku dari berbagai pembaca
- Metode Collaborative Filtering: Metode ini digunakan untuk membangun model rekomendasi berdasarkan kesamaan preferensi antara pengguna dan pengguna lain dalam dataset.
- Algoritma: Algoritma digunakan untuk memproses data dan membangun model rekomendasi. 

## Data Preparation
Dalam pembuatan sistem rekomendasi buku, ada beberapa proses penting yaitu
Mengecek keberadaan _missing value_,  ini dilakukan untuk memastikan apakah terdapat _missing value_ dalam dataset. Jika terdapat , langkah selanjutnya adalah membersihkan _missing value_ agar tidak mempengaruhi kualitas rekomendasi. Dalam notebook, langkah ini dilakukan dengan menggunakan fungsi _isnull()_ untuk mengecek keberadaan _missing value_, kemudian menggunakan fungsi _dropna()_ untuk membersihkan _missing value._
Pengurutan buku berdasarkan ISBN, dilakukan untuk mengurutkan buku berdasarkan ISBN (International Standard Book Number). Pengurutan ini berguna untuk mempermudah proses rekomendasi berdasarkan kesamaan karakteristik buku.
Memasukkan buku yang telah diurutkan ke dalam variabel _fix_ buku. Setelah dilakukan pengurutan buku berdasarkan ISBN, buku-buku tersebut dimasukkan ke dalam variabel _fix_ buku. Tujuan dari langkah ini adalah untuk mempersiapkan data buku yang akan digunakan dalam proses _modelling._
_Duplikasi_ data yang dapat mempengaruhi kualitas rekomendasi dan menghasilkan rekomendasi yang tidak akurat. Oleh karena itu, data duplikat perlu dihapus sebelum dilakukan proses _modelling_. Dalam notebook, proses ini dilakukan pada data rating.
dan Pembuatan _dictionary_ yang digunakan untuk menyimpan informasi tentang buku dan pengguna. _Dictionary_ ini akan digunakan dalam proses _modelling_ untuk membangun model rekomendasi. Dalam notebook, proses ini dilakukan dengan membuat _dictionary_ untuk data buku dan data pengguna.

## Modeling
Mengenai cara kerja model _RecommenderNet_ yaitu 
Model _RecommenderNet_ membutuhkan data rating dari pengguna yang mencerminkan preferensi dan kesukaan mereka terhadap buku. 
Data rating yang dikumpulkan akan digunakan untuk membangun matriks rating. Matriks rating ini akan merepresentasikan hubungan antara pengguna dan buku. 
Sebelum dilakukan proses _modelling,_ data rating perlu diproses terlebih dahulu. Hal ini termasuk membersihkan data duplikat, mengisi _missing value_ jika ada, dan melakukan normalisasi data jika diperlukan. Tujuan dari pemrosesan data ini adalah untuk memastikan kualitas data yang digunakan dalam proses _modelling_.
Setelah data rating diproses, model _RecommenderNet_ akan dibangun menggunakan metode _Collaborative Filtering_. Metode ini akan mempelajari pola dan hubungan antara pengguna dan buku berdasarkan data rating yang ada. 
Dan Setelah model dibangun, langkah selanjutnya adalah melakukan evaluasi terhadap model untuk mengukur kualitas dan performanya. Evaluasi dilakukan dengan menggunakan metrik seperti _Root Mean Square Error (RMSE)_ untuk mengevaluasi sejauh mana model dapat memberikan rekomendasi yang akurat dan relevan.
Setelah model dievaluasi, model _RecommenderNet_ dapat digunakan untuk memberikan rekomendasi buku kepada pengguna. Rekomendasi ini didasarkan pada kesamaan preferensi dan kesukaan pengguna dengan pengguna lain dalam dataset. Model akan mengidentifikasi buku-buku yang disukai oleh pengguna lain dengan preferensi serupa dan merekomendasikannya kepada pengguna.

Model _RecommenderNet_ sesuai untuk metode _Collaborative Filtering_ karena model ini menggunakan data rating dari pengguna untuk mempelajari preferensi dan perilaku pembaca. Data rating ini digunakan untuk mengidentifikasi kesukaan dan preferensi pembaca terhadap buku tertentu. Dengan mempelajari preferensi dan perilaku pembaca, model _RecommenderNet_ dapat memberikan rekomendasi buku yang relevan dan sesuai dengan minat pembaca.
Hasil Top-N Recommendation Hasil pengujian sistem rekomendasi dengan pendekatan _Collaborative Filtering_ sebagai berikut:
|  no | Book Title                      |   
|-----|---------------------------------|
|  1  | Harper Mass Market Paperbacks   |   
|  2  | Putnam Publishing Group         |   
|  3  | Warner Books                    |   
|  4  | Basic Books                     |   
|  5  | HarperCollins (UK)              |   
|  6  | Fireside                        |  
|  7  | Ebury Press                     |   
|  8  | Pocket                          |  
| 9   | Goblinshead                     |   
| 10  | Santillana S.A. (Alfaguara)     |



## Evaluation
_RMSE (Root Mean Square Error)_ adalah metrik evaluasi yang digunakan untuk mengukur seberapa akurat model dalam memprediksi nilai.
Fungsi _plt.plot()_ digunakan untuk memplot nilai _RMSE_ untuk kumpulan data pelatihan dan pengujian, yang disimpan dalam kamus _history.history_ masing-masing di bawah kunci _'root_mean_squared_error'_ dan _'val_root_mean_squared_error'._ 
Fungsi _plt.title(), plt.ylabel()_, dan _plt.xlabel()_ digunakan untuk menambahkan judul dan label sumbu ke plot. 
Terakhir, fungsi _plt.legend()_ digunakan untuk menambahkan legenda ke plot yang menunjukkan baris mana yang sesuai dengan data pelatihan dan mana yang sesuai dengan data pengujian.
Hasil yang didapatkan untuk _val_root_mean_squared_error_ adalah 0.4351 dan _aroot_mean_squared_error_ '0.4351.
Untuk _loss_ pelatihan 0.6855  dan _val_loss_ 0.6844. Hasil _RMSE_ yang diperoleh menunjukkan hasil yang belom baik karena, Semakin kecil nilai _RMSE_, semakin baik performa model dalam memprediksi data. _RMSE_ yang rendah menunjukkan bahwa model memiliki tingkat akurasi yang tinggi.
ada beberapa cara untuk memperbaiki nilai _RMSE_ yang kurang baik
memeriksa Outlier yang dapat mempengaruhi nilai _RMSE_ secara signifikan. Menghapusnya atau memperlakukannya secara berbeda dapat meningkatkan akurasi model, Memilih fitur yang paling relevan dapat meningkatkan akurasi model dan mengurangi nilai _RMSE_
Mencoba algoritme yang berbeda juga dapat membantu menemukan algoritme yang paling sesuai untuk masalah tertentu dan mengurangi nilai _RMSE_
Meningkatkan jumlah data yang digunakan untuk melatih model dapat meningkatkan akurasinya dan mengurangi nilai _RMSE_
dan juga bisa menggunakan validasi silang dapat membantu mengevaluasi performa model dan mengurangi nilai _RMSE_
Gambar (2)  gambar hasil grafik plot ![image](https://github.com/sittikhairunnisak/Laporan/assets/132251307/67f7502a-4e17-4f0b-9cda-b932dcecc8b9)


Referensi: [1.] Djamal,A Rhamadanus. Maharani, Warih dan Kurniati, Angelina Prima (2010). Analisis dan Implementasi Metode Item-Based Clustering Hybrid Pada Recomender Sytem 
[2] Li, Qing  and  Kim, Byeong  Man  2002. An Approach for Combining Content-based  and  Collaborative Filters. Departement of Computer Sciences,Kumoh National Institute of Technology


**---Ini adalah bagian akhir laporan---**
