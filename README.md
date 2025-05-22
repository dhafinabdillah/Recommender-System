# Recommender-System
Membuat sistem rekomendasi dengan menggunakan Python, data yang akan kita gunakan di sini adalah database film dari imdb lengkap dengan metadatanya

Dalam kasus ini, kita akan menggunakan kombinasi antara rata-rata rating, jumlah vote, dan membentuk metric baru dari metric yang sudah ada, kemudian kita akan melakukan sorting untuk metric ini dari yang tertinggi ke terendah.

Sistem ini menawarkan rekomendasi yang umum untuk semua user berdasarkan popularitas film menggunakan formula dari IMDB dengan Weighted Rating sebagai berikut
![image](https://github.com/user-attachments/assets/a21ac4d5-9efb-4c24-a81d-9d67aebedb44)

dimana,
v: jumlah votes untuk film tersebut
m: jumlah minimum votes yang dibutuhkan supaya dapat masuk dalam chart
R: rata-rata rating dari film tersebut
C: rata-rata jumlah votes dari seluruh semesta film
