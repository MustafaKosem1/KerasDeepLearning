# Keras Deep Learning
Farklı farklı iskambil kartları resimlerinden oluşan bir veriseti kullanılarak derin öğrenme projesi gerçekleştirilmiştir.

Python Keras kullanılarak bir derin öğrenme modeli oluşturulmuştur. Bu model ilgili veriseti ile eğitilmiştir ve sonuçlar gözlenmiştir. Kartların sınıflandırılmasında ortalama %91-%92 arası başarı oranları yakanlanmıştır.

Modelde dropout, data augmentation gibi yöntemler kullanılarak modelin genelleme yeteneği arttırılmaya çalışılmıştır.

Modelde epoklar boyunca accuracy değerleri kaydedilip program sonunda grafik ile görselleştirilmektedir. Ayrıca bu epoklar arasındaki en iyi model de kaydedilmektedir. İstenirse daha sonra bu model üzerinden işlemler devam ettirilebilmektedir. Ek olarak gereksiz uzun eğitimi önlemek için early stopping mekanizması da eklenmiştir.

Veriseti 224x224 boyutlarında jpg resimlerden oluşmaktadır. 53 farklı sınıf içermektedir. Veriseti linki aşağıdadır:
https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification
