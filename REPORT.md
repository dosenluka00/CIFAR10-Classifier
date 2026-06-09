# CIFAR-10 Klasifikacija slika

## Pregled

Ovaj projekt uključuje implementaciju i evaluaciju nekoliko pristupa klasifikacije slika na CIFAR-10 skupu podataka koristeći PyTorch. Cilj nije bio samo izmodelirati klasifikator, već usporediti više mogućih infrastruktura i na temelju usporedbe odabrati najprikladniju.

Evaluacija je izvršena na sljedećim modelima:

* Random baseline
* Multi-Layer Perceptron (MLP)
* Convolutional Neural Network (CNN)

Konačno rješenje sadrži CNN treniran od nule i postiže **78.46% točnost** na testnom skupu podataka, bez oslanjanja na transfer learning ili predtreniran backbone.

## Pristup

Jednostavni baseline modeli odabrani su s razlogom, netom prije implementacije CNN-a.

Aksiom za svaki model strojnog učenja je da mora biti bolji od nasumičnog pogađanja. Stoga je kao prvi baseline model implementiran random guesser, koji očekivano postiže accuracy od oko 10.

MLP služi kao klasični baseline za neuronske mreže. Slike su poravnane (*eng.flattened*) u vectore i propagirane kroz potpuno povezane slojeve. Iako ovaj model može korisno naučiti prepoznavanje uzoraka (*eng. pattern recognition*), on ignorira prostornu strukturu (*eng. spatial architecture*) slika.

Završni je model prilagođena CNN koja se sastoji od konvolucijskih slojeva (*eng. convolutional layers*), objedinjujućih slojeva (*eng. pooling layers*), i potpuno povezanih klasifikacijskih slojeva (*eng. fully connected classification layers*). Za razliku od MLP-a, CNN zadržava prostorne informacije (*eng. spatial information*) i uči ponovno iskoristive vizualne značajke(*eng. features*) kao što su rubovi, teksture i oblici.

Ova progresija iz MLP-a u CNN omogućava direktnu usporedbu kako infrastrukturne odluke utječu na performanse.

## CNN ili Transfer Learning?

Predtrenirani backbone poput ResNet18 ili EfficientNet-a bi vjerojatno postigao veću točnost. Međutim, namjerno ga nisam odabrao iz sljedećih razloga:

Najprije, CIFAR-10 skup podataka sadrži slike veličine 32×32 pixela, dok je većina predtreniranih ImageNet modela namijenjena za značajno veće input vrijednosti, stoga bi za korištenje transfer learninga morali dodatno prilagoditi podatke i veličinu slika.

Zatim, zadatak je bio pokazati razumijevanje i inženjerske prakse pri odabiru infrastrukture, a ne oslanjanje na eksterne predtrenirane komponente.

Naposljetku, transfer learning bi donio heuristiku naučenu od ImageNeta, što bi onemogućilo direktnu usporedbu sa baseline modelom, odnosno MLP-om. Treniranjem CNN-a od nule, utjecaj konvolucijske arhitekture može biti evaluiran direktno.

Unatoč svojoj jednostavnosti (malom broju parametara), konačni CNN uči značajne vizualne reprezentacije i postiže snažne performanse bez korištenja vanjskih skupova podataka ili predtreniranih težina.

## Pretprocesiranje podataka

Za vrijeme razvoja, nekoliko stavki je implementirano kako bi se poboljšala generalizacija modela na neviđenom skupu podataka.

Sve slike su normalizirane koristeći statističke vrijednosti CIFAR10 skupa podataka.

Proširenje podataka (*eng. data augmentation*) primjenjuje se tijekom treninga putem nasumičnih horizontalnih preokreta i nasumičnih izrezivanja s obrubom (*eng. padding*). Ovo omogućuje modelu iskustvo sa više inačica jedne slike i povećava robusnost.

Ključan implementacijski detalj je da su proširenja (*eng. augmentation*) primijenjena isključivo na skupu podataka za trening. Testni i validacijski skupovi podataka ostaju nepromijenjeni, kako bi osigurali da evaluacijske metrike daju uvid u stvarne performanse modela. Deterministička train-validation podjela (*eng. split*) je korištena kako bi osigurala ponavljanje eksperimenta.

## Faza treninga

Cjelokupan pipeline napredovao je za vrijeme razvoja.

Početni CNN postizao je oko 69% točnosti na testnom skupu podataka.

Nakon uvođenja normalizacije unosa (*eng. input*), performanse su se poboljšale.

Primjena tehnika proširenja podataka (*eng. data augmentation*) i osiguravanje čistog validacijskog pipeline-a, također je pozitivno utjecalo na performanse.

Finalna verzija dodatno je koristila validacijske checkpointe, rano zaustavljanje (*eng. early-stopping*) i adaptivno prilagođavanje stope učenja koristeći ReduceLROnPlateu tehniku.

Trening je određen za 50 epoha, ali uz automatsko prekidanje u trenutku kad performanse na validacijskom skupu podataka prestanu pokazivati napredak u 5 epoha. Najbolji model je zatim automatski pohranjen i korišten u konačnoj evaluaciji.

Prilagođavanje stope učenja (*eng. learning rate scheduler*) omogućilo je smanjenje stope učenja kad točnost na validacijskom skupu dostigne plato, omogućujući manju i finiju prilagodbu tijekom kasnijih faza učenja.

Ova su unaprjeđenja povećali točnost na testnom skupu podatka na 78.46%, bez ikakvih modifikacija arhitekture CNN-a.

## Rezultati

|Model|Parametersi|Test Accuracy|
|-|-:|-:|
|Random|0|9.16%|
|MLP|789,258|50.79%|
|CNN|545,098|78.46%|

Iz rezultata vidimo da CNN postiže znatno bolju točnost, pritom koristeći znatno manji broj parametara.

Unatoč tomu što MLP ima veći broj parametara, on tretira svaki piksel zasebno, nakon procesa poravnanja slika(*eng. image flattening*).

S druge strane, CNN koristi prostorni lokalitet (*eng. spatial locality*) i dijeljenje težina (*eng. weights sharing*), što ga čini značajno efikasnijim obzirom na broj parametara.

Ovaj rezultat ukazuje na to da je infrastruktura i razumijevanje iste, važnija od sirovog broja parametara kad se radi na skupovima podataka sa slikama.

## Analiza

Korištene metrike su točnost, matrica zabune i klasifikacijski report koji se sastoji od preciznosti, odziva i F1 metrika za svaku klasu.

Model najbolje performanse pokazuje na klasama sa specifičnim jakim globalnim strukturama, kao što su avion, broj, automobil, žaba i konj.

Performanse su slabije na klasama koje imaju životinje sličnih karakteristika, tu se najviše ističu zabune između pasa, mačaka, ptica i divljači.

Matrica zabune pokazuje da je par mačka-pas najčešća greška, što je bilo i za očekivati uzevši u obzir da je CIFAR10 skup podataka slika gdje su fino-zrnati (*eng. fine-grained*) detalji koji bi doprinjeli identifikaciji između navedenih klasa neprimjetni zbog veličine slika od 32x32 piksela.

Najviše grešaka se događa između semantički povezanih klasa, umjesto na potpuno nepovezanim klasama, što upućuje da je CNN dobro naučio važne vizualne značajke i da dobro generalizira. Za preostale je greške razlog struktura skupa podataka, umjesto nestabilnosti u fazi treniranja.

## Automatizacija

Zadatak uključuje pokretač eksperimenata (*eng. experiment runner*) sposoban za automatsko treniranje i evaluaciju svih modela.

Za svaki eksperiment, pipeline trenira model, pohranjuje najbolju kontrolnu točku, generira krivulje za vrijeme trening faze, matricu zabune i prijespomenuti report sa F1 i srodnim metrikama na razini klase.

Naposljetku se metapodatci eksperimenta pohranjuju u JSON formatu, što omugućuje konzistentnost i ponovljivost (*eng. reproduciblity*) eksperimenta.

## Inženjerske odluke

Završni model demonstrira da relativno mali CNN kreiran od nule može postići snažne performanse na zadanom skupu podataka, ukoliko je kombiniran sa pedantnim pretprocesiranjem, proširenjem (*eng. augmentation*), prilagođavanjem stope učenja i ranim zaustavljanjem.

Najznačajniji napredak nije ostvaren povećanjem kompleksnosti modela, nego unaprijeđivanjem samog pipeline. Počevši od malo manje od 70% točnosti, performanse su podignute na 78.46% na testnom skupu podataka isključivo kroz bolju pripremu podataka i korištenje dobrih praksi pri procesu treniranja.

