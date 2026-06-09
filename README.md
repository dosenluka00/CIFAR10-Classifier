\# CIFAR-10 Klasifikacija Slika



\## Opis projekta



Ovaj projekt implementira i uspoređuje tri različita pristupa klasifikaciji slika na CIFAR-10 skupu podataka:



\* Random Baseline

\* Multi-Layer Perceptron (MLP)

\* Convolutional Neural Network (CNN)



Cilj projekta nije bio isključivo postići što veću točnost, nego demonstrirati razumijevanje procesa razvoja modela strojnog učenja, od pripreme podataka i odabira arhitekture do evaluacije, automatizacije eksperimenata i osiguravanja ponovljivosti rezultata.



Konačni CNN model postiže približno \*\*78.46% točnosti na testnom skupu podataka\*\* koristeći isključivo vlastitu arhitekturu treniranu od nule, bez transfer learninga i bez korištenja predtreniranih težina.



Za detaljno objašnjenje arhitekturnih odluka, pretprocesiranja podataka, procesa treniranja i analize rezultata pogledati datoteku \*\*REPORT.md\*\*.


\---



\## Instalacija



Projekt je razvijen koristeći Python 3.13.



Instalacija potrebnih biblioteka:



```bash

pip install -r requirements.txt

```



\---



\## Struktura projekta



```text

cifar10\_classifier/

│

├── models/

├── utils/

│

├── outputs/

│

├── train.py

├── train\_model.py

├── run\_experiments.py

├── inference.py

├── config.py

│

├── requirements.txt

├── README.md

└── REPORT.md

```



\---



\## Pokretanje jednog modela



Treniranje modela koristeći vrijednosti definirane u konfiguraciji:



```bash

python train.py

```



Pokretanje specifičnog modela:



```bash

python train.py --model cnn

```



Primjeri:



```bash

python train.py --model random

python train.py --model mlp

python train.py --model cnn

```



\---



\## Broj epoha



Ako broj epoha nije naveden kroz naredbeni redak, koristi se zadana vrijednost definirana u konfiguraciji projekta.



Primjer eksplicitnog zadavanja:



```bash

python train.py --model cnn --epochs 50

```



\---



\## Pokretanje svih eksperimenata



Za automatsko treniranje i evaluaciju svih implementiranih modela:



```bash

python run_experiments.py

```



Skripta će redom:



1\. Pokrenuti Random Baseline

2\. Pokrenuti MLP model

3\. Pokrenuti CNN model

4\. Generirati usporednu tablicu rezultata



\---



\## Generirani izlazi



Za svaki model automatski se generira zaseban direktorij unutar mape `outputs`.



Primjer:



```text

outputs/

├── random/

├── mlp/

└── cnn/

```



Unutar svakog direktorija nalaze se:



\* najbolji spremljeni model (`\*.pth`)

\* matrica zabune (Confusion Matrix)

\* krivulja gubitka tijekom treniranja

\* krivulja validacijske točnosti

\* klasifikacijski izvještaj

\* JSON datoteka s metapodacima eksperimenta



\---



\## Reproducibilnost



Kako bi rezultati bili ponovljivi:



\* koristi se fiksni slučajni generator (`random seed`)

\* train/validation podjela je deterministička

\* validacijski skup ne koristi data augmentation

\* najbolji model automatski se sprema tijekom treniranja

\* koristi se Early Stopping mehanizam

\* koristi se Learning Rate Scheduler (`ReduceLROnPlateau`)



\---



\## Rezultati



| Model  | Parametri | Test Accuracy |

| ------ | --------: | ------------: |

| Random |         0 |         9.16% |

| MLP    |   789,258 |        50.79% |

| CNN    |   545,098 |        78.46% |



CNN postiže značajno bolju točnost od MLP modela unatoč manjem broju parametara, što potvrđuje prednost konvolucijskih arhitektura pri radu sa slikovnim podacima.



\---



\## Autor


Luka Došen



