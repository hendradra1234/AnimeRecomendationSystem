# Laporan Proyek Machine Learning - Hendra

## Domain Proyek
### Lantar Belakang
Anime adalah animasi asal Jepang yang digambar dengan tangan maupun menggunakan teknologi komputer. Kata anime merupakan singkatan dari animation dalam bahasa Inggris, yang merujuk pada semua jenis animasi. Di luar Jepang, istilah ini digunakan secara spesifik untuk menyebutkan segala animasi yang diproduksi di Jepang. Meskipun demikian, tidak menutup kemungkinan bahwa anime dapat diproduksi di luar Jepang. Beberapa ahli berpendapat bahwa anime merupakan bentuk baru dari orientalisme.
<div>
    <img src="https://review1st.com/wp-content/uploads/2021/08/anime.jpeg" width="900"/>
</div>

Referensi: [Anime News Network Lexicon - Anime](https://www.animenewsnetwork.com/encyclopedia/lexicon.php?id=45)

## Business Understanding
Proyek ini dibangun untuk perusahaan dengan karakteristik bisnis sebagai berikut :

+ Perusahaan Streaming Video yang ingin memberikan rekomendasi Anime yang sesuai.
+ Perusahaan TV berbayar yang ingin memberikan rekomendasi Anime yang sesuai.
+ Marketplace Film dan Animasi yang ingin memberikan rekomendasi Anime yang sesuai.

### Problem Statement

+ Fitur apa yang paling berpengaruh terhadap hasil rekomendasi anime?
+ Bagaimana cara memproses data agar dapat dilatih dengan baik oleh model?
+ Apa Rekomedasi Anime yang diberikan ke Client Steaming tergantung pada anime dan genre yang di sukai?

### Goals

+ Mengetahui kombinasi fitur yang berpengaruh pada rekomendasi anime.
+ Membuat dataset bisa di gunakan untuk melatih model.
+ Membuat model yang bisa di gunakan untuk merekomendasikan anime ke pengunna baru dan pengunna lama berdasarkan anime dan genre yang di sukai.

### Solution Statement

+ Menganalisis data dengan melakukan univariate data analysis dan multivariate data analysis yang di kombinasikan dengan visualisasi Data untuk memahami relasi setiap variable untuk rekomedasi.
+ Memilih fitur yang diperlukan, mengabungkan dataset anime dan anime_rating, menyiapkan data untuk melatih model.
+ Membangun model Content-based Filtering dengan Sklearn TF-IDF untuk merekomendasikan anime ke user baru ataupun user lama berdasarkan anime dan genre yang di sukai.

## Data Understanding & Preprocessing

Dataset yang digunakan untuk penelitian ini mengunakan [Kaggle: Anime-Recomendation-Dataset](https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database). Terdapat dua dataset di dalam file kompresi `zip` Anime-Recomendation-Dataset, yaitu dataset `anime` dan `anime_rating`. Berikut merupakan deskripsi Dataset `anime` dan `anime_rating`:

+ Dataset `anime`:
    - Dataset beformat CSV.
    - Dataset memiliki total 12294 sample dengan 7 fitur.
    - Dataset memiliki total 1 fitur bertipe float(64), 2 fitur bertipe int64, dan 4 fitur bertipe object.
    - Terdapat missing value pada dataset.
    - Tidak terdapat Duplikasi data pada Dataset.

+ Dataset `anime_rating`:
    - Dataset beformat CSV.
    - Dataset memiliki total 7813737 sample dengan 3 fitur.
    - Dataset memiliki total 1 fitur bertipe int64.
    - Tidak ada missing value pada dataset.
    - Terdapat Duplikasi data pada Dataset.

### Variable Pada Dataset

+ Dataset `anime`:
    - `anime_id` = ID yang bersifat unik untuk setiap anime.
    - `name` = Judul anime.
    - `genre` =  genre untuk setiap judul anime.
    - `type` = Tipe penayangan anime, seperti TV, OVA, etc.
    - `episodes` = jumlah episode untuk setiap judul anime.
    - `rating` = Rata-rata rating setiap anime relative dengan jumlah user yang memberi rating.
    - `members` = Jumlah anggota komunitas untuk setiap judul anime.

+ Dataset `anime_rating`:
    - `user_id` = ID unik ID yang bersifat unik untuk setiap user.
    - `anime_id` = ID dari dataset anime yang diberi rating oleh user.
    - `rating` = Rating yang diberikan oleh user.


### Univariate Analysis

Univariate Analysis merupakan proses analisis setiap fitur yang dilakukan secara terpisah.

#### Analisis atribut pada dataset anime

|       | anime_id | rating   | members    |
|-------|----------|----------|------------|
| count | 12017.00 | 12017.00 | 12017.00   |
| mean  | 13638.00 |     6.48 | 18348.88   |
| std   | 11231.08 | 1.02     | 55372.50   |
| min   | 1.00     | 1.67     | 12.00      |
| 25%   | 3391.00  | 5.89     | 225.00     |
| 50%   | 9959.00  | 6.57     | 1552.00    |
| 75%   | 23729.00 | 7.18     | 9588.00    |
| max   | 34519.00 | 10.00    | 1013917.00 |

Dataset `anime` memiliki rating terendah 1.67 dan rating tertinggi 10 dengan rata-rata 6.48. Dataset ini juga memiliki jumlah anggota komunitas terendah 12 dan yang terbanyak mencapai 1013917 member dengan nilai rata-rata 18348 member. Perbedaan nilai `min` dan `max` dari jumlah anggota komunitas anime memang cukup jauh karena perbedaan popularitas setiap anime.

#### Analisis atribut numerik pada dataset anime_rating

|       | user_id    | anime_id   | rating     |
|-------|------------|------------|------------|
| count | 7813736.00 | 7813736.00 | 7813736.00 |
| mean  |   36727.96 |    8909.07 | 6.14       |
| std   | 20997.95   | 8883.95    | 3.73       |
| min   | 1.00       | 1.00       | -1.00      |
| 25%   | 18974.00   | 1240.00    | 6.00       |
| 50%   | 36791.00   | 6213.00    | 1552.00    |
| 75%   | 54757.00   | 14093.00   | 9.00       |
| max   | 73516.00   | 34519.00   | 10.00      |

Dataset `anime_rating` memiliki rating terendah yang diberikan user pada suatu anime adalah -1 dan rating tertinggi adalah 10. Rating -1 menandakan bahwa user menonton anime, namun tidak memberikan rating. Sample user yang tidak memberikan rating akan dihapus karena tidak berguna untuk Training model.
Sample dataset yang tidak di gunakan di hapus dengan mengunakan kode berikut:
```
df_anime_rating = df_anime_rating[~(df_anime_rating.rating == -1)]
```

+ Analisis fitur `genre` pada dataset anime:
    <div><img src="https://github.com/hendradra1234/TimeSeries/blob/main/anime-genre-distribution.png?raw=true" width="500"/></div>
    Dataset anime memiliki genre yang beragam. tertapi untuk mayoritas genre di dominasi genre `Hentai` dan di ikuti genre `Comedy`. Pattern Genre ini memang tidak mengejutkan karena 2 genre itu memang memiliki banyak peminat dan genre itu hampir selalu ada di setiap anime populer.

+ Analisis fitur `type` pada dataset anime:
    <div><img src="https://github.com/hendradra1234/TimeSeries/blob/main/anime-type-distribution.png?raw=true" width="500"/></div>

Dataset anime memiliki `type` yang di dominasi oleh media penanyangan TV dan OVA lalu di ikuti Movie. Pattern Distribusi ini bisa terbentuk karena market share penyiaran di dominasi TV dan OVA untuk anime yang rilis di era sebelum layanan Streaming via Internet menjamur.
Tipe Movie merupakan anime yang tayang dalam bentuk film. Original Video Animation (OVA) merupakan anime yang dirilis dalam bentuk fisik (CD, DVD, HD-DVD, Blu-ray, dll) tanpa mengunakan media penyiaran TV. Original Net Animation (ONA) merupakan anime yang tayang lebih dahulu melalui internet Streaming Service. Spesial merupakan episode anime yang durasinya hanya beberapa menit dan biasanya tidak cannon dengan storyline dan lebih difokuskan sebagai fans service.

+ Analisis distribusi data rata-rata `rating` anime:

    <div><img src="https://github.com/hendradra1234/TimeSeries/blob/main/anime-avg-raring-distribution.png?raw=true" width="500"/></div>

    mayoritas rata-rata rating anime pada dataset 'anime' tersebar pada range rating 4 hingga 8.

+ Analisis distribusi data `rating` yang diberikan pengguna:

    <div><img src="https://github.com/hendradra1234/TimeSeries/blob/main/user-anime-rating-distribution.png?raw=true" width="500"/></div>

    mayoritas rating yang diberikan pengguna berada pada range rating 5 sampai 10.

### Multivariate Analysis

Multivariate Analysis merupakan analisa data yang digunakan untuk menemukan relasi antara dua atau lebih fitur dalam dataset.

+ Top 5 anime terbesar berdasarkan jumlah komunitas:

    <div><img src="https://github.com/hendradra1234/TimeSeries/blob/main/top-5-by-community.png?raw=true" width="1000"/></div>

+ Top 5 anime berdasarkan jumlah rata-rata rating anime:

    <div><img src="https://github.com/hendradra1234/TimeSeries/blob/main/top-5-by-anime-avg-rating.png?raw=true" width="1000"/></div>


## Data preprocessing

+ Menghapus missing value pada dari dataset `anime` dan dataset `anime_rating`.
+ Menghapus duplikasi sample dari dataset `anime` dan dataset `anime_rating`.
+ Menghapus symbol pada `name` anime pada dataset `anime` dan dataset `anime_rating`.
+ Dataset Fussion atau mengabungkan dataset `anime` dan dataset `anime_rating`.
    - Plotting Top 5 anime dari `dataset_fussion` berdasarkan kontribusi rating pengguna:

    <div><img src="https://github.com/hendradra1234/TimeSeries/blob/main/Anime%20Rating%20Contribution%20from%20Fussion%20Dataset.png?raw=true" width="1000"/></div>


## Modeling and Result

### Content Based Filtering

Sistem yang dibangun oleh Penelitian ini adalah sistem rekomendasi berdasarkan fitur `name` dan `genre` anime yang berbasis `content based filtering`.
Sistem rekomendasi berbasis konten ini merupakan sistem yang merekomendasikan konten yang mirip dengan konten yang disukai pengguna sebelumnya. Apabila suatu konten memiliki karakteristik yang sama atau hampir sama dengan konten lainnya, maka kedua konten tersebut dapat dikatakan mirip.
Berikut merupakan Algoritma model yang di mamfaatkan untuk membuat sistem rekomedasi berbasis `content based filtering`:
+ TF-IDF
    `TF-IDF` atau `Term Frequency - Inverse Document Frequency` dapat didefinisikan sebagai perhitungan seberapa relevan sebuah kata dalam rangkaian atau korpus dengan sebuah teks. Makna meningkat secara proporsional dengan berapa kali dalam teks sebuah kata muncul tetapi dikompensasi oleh frekuensi kata dalam korpus (kumpulan data).
    TF-IDF digunakan pada sistem rekomendasi anime untuk menentukan representasi fitur penting dari setiap fitur `name` dan `genre` anime. TF-IDF dijalankan dengan mengunakan fungsi `tfidfvectorizer()` dari library `sklearn`.

    Setelah itu hasil dari `TF-IDF` akan di convert ke dalam bentuk matriks dengan fungsi `todense()`. Dibuat Dataframe baru untuk menunjukkan matriks `TF-IDF` untuk beberapa `anime` dan `genre`. Nilai matriks yang semakin tinggi menunjukkan semakin erat relasi antara `anime` dengan `genre` tersebut.


+ Cosine Similarity
    `Cosine Similarity` mengukur kesamaan antara dua vektor dan menentukan apakah kedua vektor menunjuk ke arah yang sama. Teknik ini bekerja dengan menghitung sudut cosinus antara dua vektor. Semakin kecil sudut cosinus antara dua vektor, semakin besar nilai kemiripan cosinusnya.
    Cosine similarity digunakan untuk menghitung derajat kesamaan antar anime. Untuk menjalankan cosine similarity digunakan fungsi `cosine_similarity` dari library sklearn.
    Tahap ini menghitung cosise similarity pada dataframe `tfidf_matrix` yang dihasilkan dari tahapan `TF-IDF` sebelumnya.
    <div><img src="https://miro.medium.com/max/1400/1*IhpY-6LYV75983THCpWo-w.png" width="500"/></div>

    [Referensi gambar](https://towardsdatascience.com/what-is-cosine-similarity-how-to-compare-text-and-images-in-python-d2bb6e411ef0)

    Dataframe baru dibuat untuk menghitung relasi kesamaan antar judul dan genre anime mengunakan hasil dari `cosine similarity`. Semakin tinggi nilai `cosline similarity`, maka kedua anime akan semakin memiliki kesamaan.

### Result
Fungsi `recommendations` dibuat untuk menemukan rekomendasi anime menggunakan `similarity` yang telah didefinisikan sebelumnya. Fungsi ini bekerja dengan cara mengambil anime dengan similarity terbesar dari index yang ada. Selanjutnya adalah menemukan rekomendasi yang mirip dengan anime Bleach, Swort Art Online dan Dragon Ball. Berikut merupakan top 5 rekomendasi dari anime tersebut:

- Bleach

    |index  |	name	                                        | genre                                                     |
    |-------|---------------------------------------------------|-----------------------------------------------------------|
    |0	    |Bleach Movie 3: Fade to Black - Kimi no Na wo Yobu | Action, Comedy, Shounen, Super Power, Supernatural        |
    |1	    |Bleach Movie 4: Jigoku-hen                         | Action, Comedy, Shounen, Super Power, Supernatural        |
    |2	    |Code:Breaker                                       | Action, Comedy, School, Shounen, Super Power, Supernatural|
    |3	    |Yozakura Quartet: Tsuki ni Naku                    | Action, Comedy, Magic, Shounen, Super Power, Supernatural |
    |4	    |Yozakura Quartet: Hoshi no Umi                     | Action, Comedy, Magic, Shounen, Super Power, Supernatural |

- Sword Art Online

    |index  |	name	                                        | genre                                         |
    |-------|---------------------------------------------------|-----------------------------------------------|
    |0	    |Sword Art Online II                                | Action, Adventure, Fantasy, Game, Romancel    |
    |1	    |Sword Art Online: Extra Edition                    | Action, Adventure, Fantasy, Game, Romance     |
    |2	    |Sword Art Online II: Debriefing                    | Action, Adventure, Fantasy, Game              |
    |3	    |Bakugan Battle Brawlers                            | Action, Fantasy, Game                         |
    |4	    |Monster Strike: Mermaid Rhapsody                   | Action, Fantasy, Game                         |


- Dragon Ball

    |index  |	name	                                        | genre                                                     |
    |-------|---------------------------------------------------|-----------------------------------------------------------|
    |0	    |Dragon Ball Z Movie 11: Super Senshi Gekiha!! Katsu no wa Ore da | Action, Adventure, Comedy, Fantasy, Martial Arts, Shounen, Super Power        |
    |1	    |Dragon Ball Z                                      | Action, Adventure, Comedy, Fantasy, Martial Arts, Shounen, Super Power        |
    |2	    |Dragon Ball Super                                  | Action, Adventure, Comedy, Fantasy, Martial Arts, Shounen, Super Power|
    |3	    |Dragon Ball GT: Goku Gaiden! Yuuki no Akashi wa Suushinchuu |	Action, Adventure, Comedy, Fantasy, Martial Arts, Shounen, Super Power |
    |4	    |Dragon Ball Z Movie 15: Fukkatsu no F              | Action, Adventure, Comedy, Fantasy, Martial Arts, Shounen, Super Power |

model telah berhasil memberikan top 5 rekomendasi paling relavan tergantung anime yang di sukai, di sini anime yang di rekomendasikan memiliki relasi yang mendekati dengan genre dan nama anime itu sendiri.

### Evaluation
Model berhasil memberikan rekomendasi 5 series anime yang memiliki hubungan nama dan genre dengan Bleach, Sword Art Online dan Dragon Ball. Model menghasilkan akurasi 15/15 berdasarkan relasi `anime` dan `genre` yang di sukai, sehingga model memiliki tingkat presisi 100%.
<div><img src="https://dicoding-web-img.sgp1.cdn.digitaloceanspaces.com/original/academy/dos:819311f78d87da1e0fd8660171fa58e620211012160253.png" width="500"/></div>
