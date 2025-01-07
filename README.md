# Saudi Arabia Used Car Regression Prediction
Oleh : Glen Valencius

Presentation Deck : [View the Google Slides Presentation](https://docs.google.com/presentation/d/1b1LtrZdJf9H3O_uXwpsv4zk_prxyAwYF/edit#slide=id.p1)


## Introduction
Project ini bertujuan untuk memprediksi harga jual mobil bekas di platform Syarah.com menggunakan machine learning. Di dalam project ini kita akan mencari cara cleaning terbaik, model terbaik, dan parameter terbaik agar menghasilkan prediksi yang terbaik yang didasarkan dari fitur - fitur yang telah disediakan dalam dataset.



Daftar Isi
1. Business Problem Understanding
2. Data Understanding
3. Data Pre-Processing
4. Modelling
5. Evaluation
6. Conclusion
7. Recommendation


## 1. Business Problem Understanding
 - **Context**

Syarah.com adalah platform bisnis b2c marketplace yang menjual mobil bekas dan juga baru. Syarah.com menjual list mobilnya di website. Syarah.com sendiri beroperasi di Riyadh Saudi Arabia dan berdiri tahun 2015 hingga kini. Melansir dari website Syarah.com pada tahun 2022 Syarah App mendapatkan ranking top 10 most-used app di Saudi Arabia menurut Saudi Ministry of Communications and Information Technology.

Kelebihan dari Syarah.com adalah konsumen dapat membeli dan menjual mobil melalui website atau app dan mobil akan langsung dikirim ke tujuan.


- **Problem Statement**

Salah satu hal yang diperlukan adalah menentukan harga mobil bekas dengan keadaan yang berbeda - beda.

Jika ingin melakukan pengembangan bisnis dengan memperbanyak penjualan, **maka akan semakin banyak pula mobil bekas yang harus diberikan harga**. Bila hanya mengandalkan team maka tidak akan efisien. Oleh karena itu diperlukan sebuah model Machine Learning yang dapat memprediksi harga dari sebuah mobil bekas dengan melihat spesifikasi dari mobil tersebut untuk membantu team dalam menentukan harga mobil.

Untuk meng hire **Appraiser** membutuhkan **SAR 4,000 to SAR 8,000 per month** setiap orang bila ingin menambah pegawai, sedangkan dengan Model akan lebih menghemat pengeluaran perusahaan hanya dengan memantau model saja **Data Scientist SAR 8,000 to SAR 12,000**.

Dikarenakan kita akan memprediksi harga yang dalam bentuk numerical, kita akan menggunakan Model Regresi.

 - **Stakeholders**

1. Calon Penjual Mobil Bekas
    - Masalah : Penjual kesulitan untuk menentukan harga jual mobilnya
    - Dampak : Mobil tidak laku terjual
2. Team Appraiser
    - Masalah : Semakin banyak mobil yang akan dijual, semakin butuh banyak pegawai untuk melakukan analisa harga
    - Dampak : Semakin sulit pekerjaan dengan beban kerja yang tinggi
3. Perusahaan Syarah.com
    - Masalah : Perusahaan membutuhkan pengeluaran lebih untuk menghire Team Appraiser untuk menilai harga mobil
    - Dampak : Sulit untuk melakukan ekspansi jumlah mobil yang dijual


- **Goals**

Berdasarkan permasalah tersebut, Syarah,com perlu memiliki ‘tools’ yang dapat **memprediksi harga mobil sesuai spesifikasi dari mobil tersebut**. Variabel - variabel yang ada di dataset diharapkan dapat memberikan prediksi harga dengan akurasi yang baik dalam menentukan harga mobil.

Model ini dapar digunakan oleh 2 stakeholder yaitu Calon Penjual ketika ingin menjual dan diberikan harga perkiraan dan Team Appraiser sebagai referensi untuk menentukan harga mobil ketika selesai diinspeksi.

Target Metric :

| Metric | Target |
| --- | --- |
| MAE | < 10.000 SAR |
| MAPE | < 20 |
| R2 | > 0.70 |


- **Analytic Approach**

Hal yang perlu kita lakukan adalah menganalisa data untuk mendapatkan pola dari fitur - fitur yang ada, yang membedakan satu mobil dengan mobil lainnya.

Selanjutnya, kita akan membangun model regresi yang akan membantu perusahaan untuk dapat memprediksi harga mobil bekas yang akan dijual di platform yang berguna untuk team Appraiser dalam pekerjaan mereka.

- **Metric Evaluation**

detailnya sebagai berikut :
- MAE (Primary metric) : Baik digunakan karena satuannya mengikuti satuan asli data, digunakan jika data terdapat outlier karena tidak akan terpengaruh. Ini akan digunakan sebagai fokus untuk Machine Learning ini karena data terdapat outlier.
- MAPE (Secondary metric): Baik untuk mengetahui persentase error.
- R Square : digunakan untuk mengevaluasi performance sebuah model. Semakin mendekati 1 maka model cocok dengan data, tetapi tidak berarti prediksinya akan bagus
- RMSE : Membebankan hitungan error di data yang outlier, dapat lebih fokus untuk menurunkan error prediksi

Semakin kecil nilai MAE dan MAPE model yang dibuat akan semakin akurat dalam memprediksi harga mobil sesuai dengan limitasi fitur yang digunakan.

## 2. Data Understanding
- **Dataset**
  
Secara umum, kita bisa melihat bahwa:
* dataset Data Saudi Used Car memiliki 5624 row dan 11 kolom
* Data bila kolom Negotiable True akan membuat kolom Price 0
* Mileage in KM
* Pembagian Jenis Option : 
    - Standard: Model Dasar, yang paling murah dengan fitur paling sedikit
    - Full: Level tertinggi, yang mencakup sebagian besar atau semua fitur dan bisa lebih dari dua kali lipat harga model dasar.
    - Semi-full: Mobil dengan level menengah yang mencakup beberapa fitur mewah, seperti jendela elektrik, pendingin udara otomatis, dan sistem suara yang baik.


- **Feature**
  
  * Dataset merupakan data mobil bekas yang dijual di Syarah.com
  * Setiap baris data merepresentasikan informasi terkait mobil bekas.

**Attributes Information**

| **Attribute** | **Data Type** | **Description** |
| --- | --- | --- |
| Type | Object | Type of Car |
| Region | Object | The region in which the used car was offered for sale |
| Make | Object | Company Name |
| Gear Type | Object | Gear type size of used car|
| Origin | Object | Origin of used car |
| Options | Object | Options of used car (How many feature in the car) |
| Year | Integer | Manufacturing year |
| Engine_Size | Float | The engine size of used car |
| Mileage | Integer | Total distance a vehicle has traveled or its fuel efficiency (miles per gallon) |
| Negotiable | Booleans | True if the price is 0, that means it is negotiable |
| Price | Integer | Used car price. (in riyal) | 

<br>

## 3. Data Preprocessing
Langkah-langkah :

1. Menangani Missing Value: Tidak ada dapat NaN di dataset.
2. Menghapus Duplicate: Terdapat 4 data duplikat dan sudah didrop.
3. Menghapus Feature yang tidak berkaitan dengan prediksi yaitu Kolom Negotiable
4. Menghapus semua data dengan Target `Price` = 0
5. Menangani Outlier: Outlier pada kolom menggunakan teknik capping degan syarat berikut :
  * Price : 15.000 - 182.500 (Upperbound)
  * Year : 2003 - 2022 (Dilihat dari jumlah data dibawah 10 akan diexclude, karena kurang menggambarkan sebuah kelompok) 
  * Mileage : 376.000 diambil dari (Upperbound), ini didasari juga dengan riset bahwa sebuah mobil maksimum 320.000 km. Bila menggunakan Upperbord masih masuk akal.
6. Feature Engineering: Membagi beberapa fitur menjadi kelompok berikut :
  * Encoding
    - Ordinal : ['Option'] (terdapat ranking)
    - Binary Encoding : ['Type','Region','Make'] (Memilik banyak unik value)
    - One Hot : ['Gear_Type','Origin'] (memiliki sedikit unik value)

  * Scaling
    - Robust Scaller : ['Mileage','Engine_Size','Year']

- **Modelling**
Model yang akan digunakan adalah :
- KNeighbors Regressor
- DecisionTree Regressor
- Linear Regression
- XGB Regressor
- Stacking (Linear, Decision Tree, KNN, dan XGB)
- Catbooster Regressor


## 5. Evaluation
- **Model Terbaik**
Model dengan kinerja terbaik adalah Catbooster Regressor dengan parameter terbaik setelah melakukan tuning, yang mencapai metrik berikut:

| **Metric** | **Score** |
| --- | --- |
| MAE | 9837.29 |
| MAPE | 0.18 |
| R-Squared | 0.84 |
| RMSE | 14460.97 |


## 6. Conclusion

Berdasarkan model yang telah dibuat, fitur `Engine_Size`, `Year` dan `Mileage` yang paling berpengaruh terhadap `Price`.

Model terbaik adalah CatBooster dengan paramater yang sudah dilakukan tuning.

Dengan menggunakan matrik evaluasi RMSE, MAE dan MAPE. Bila dilihat dari nilai MAPE yang sudah dilakukan hyperparameter tuning, yaitu sebesar 18%. Kita dapat mengambil kesimpulan bahwa bila nanti model ini digunakan untuk memperkirakan data mobil yang baru dengan rentang harga Nilai Price minimum 15.000 dan max 182.500. Maka perkiraan rata-rata akan meleset kurang lebih sebesar 18% dari harga sebenarnya.

Tapi ada kemungkinan juga hasil prediksi meleset jauh karena bila dilihat dari visualisasi harga aktual dan prediksi dan juga residual terdapat bias yang terjadi. Hal ini bisa terjadi karena terbatasnya fitur pada dataset yang bisa merepresentasikan kondisi mobil, seperti kondisi luar mobil, berapa kali turun mesin, kondisi interior mobil, pernah tabrakan atau tidak dan lain lain.

Model ini masih bisa disempurnakan lagi sehingga menghasilkan prediksi yang lebih baik. Kedepannya kita dapat mencoba untuk membandingkan hasil prediksi model ini dengan peningkatan akurasi harga mobil dari Appraiser.

- **Model Baik digunakan Saat**
    - Memperkirakan harga mobil bekas yang fiturnya dalam rentang **Price** SAR 15.000 - 182.500 , **Mileage** dibawah 376.000 KM, **Year** 2003 - 2021 dan Tidak Negotiable

- **Model Tidak Baik digunakan Saat**
    - Memperkirakan harga mobil diluar spesifikasi di atas seperti mobil edisi terbatas, mobil antik atau mobil yang sudah dimodifikasi.

**Cost Calculation**

- Perusahaan Syarah.com :
    - Dengan adanya Model Machine Learning ini, perusahan dapat menghemat dengan tanpa menghire karyawan Appraiser. Dari misal perusahaan harus menghire 10 orang tambahan untuk menambah team Appraiser, Perusahaan hanya perlu 1 Data Science untuk memelihara dan mengupgrade Model Machine Learning ketika ada data baru.
    Perusahaan bisa menghemat dari 10 * SAR 8.000 (80.000 / bulan) menjadi 1 * SAR 12.000 (12.000 / bulan)
- Calon Penjual Mobil Bekas :
    - Calon Penjual dapat mendapatkan estimasi harga mobilnya diawal sebelum yakin untuk menjual mobilnya di Syarah.com. Ini akan meningkatkan kemungkinan calon penjual akan menjual mobilnya, dan juga akan menambahkan profit dari perusahaan Syarah.com itu sendiri dengan semakin banyaknya mobil yang bisa di listing.


## 7. Recommendation
**a. For Model**
1. Melakukan pengecekan fitur mana saja yang paling berpengaruh terhadap target ‘Price’, dengan demikian model yang dibuat akan lebih baik untuk memprediksi ‘Price’.
<br><br>
2. Mencoba untuk mengelompokan mobil dengan rentang harga tertentu sehingga kita bisa mengetahui model yang paling cocok untuk setiap kelompok harga mobil
<br><br>
3. Mencoba mengisi data mobil Negotiable dengan menggunakan Machine Learning/Deep Learning, sehingga data tersebut dapat digunakan.
<br><br>
4. Jika memungkinkan menambahkan fitur yang lebih menggambarkan kondisi mobil seperti kondisi luar mobil, berapa kali turun mesin, kondisi interior mobil, pernah tabrakan atau tidak  dan lainnya.
<br><br>
5. Melakukan penambahan data dengan data yang lengkap, karena di dataset ini walaupun di awal ada sekitar 5.600 data tetapi setelah melakukan cleaning yang bisa digunakan hanya 3.192 data yang kurang banyak untuk mendapatkan hasil prediksi yang baik.
<br><br>

**b. For Business**

1. Kedepannya model ini dapat digunakan sebagai pembanding untuk pengembangan model yang lain seperti memprediksi pengelompokan mobil mana saja yang paling banyak dibeli oleh pelanggan. Model tersebut bermanfaat untuk pengaturan mobil yang dijual di platform Syarah.com
<br><br>
2. Menambahkan fitur upload gambar di website Syarah.com ketika calon penjual akan menjual mobilnya. Ini akan mempermudah Team untuk memperkirakan hasil dan bisa juga mengimplementasikan Computer Vision.



