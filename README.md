# CodisTFM
**Conjunt de codis i dades usades per treball final de màster: Machine Learning per a l’optimització d’un model epidemiològic en tuberculosis.**

A continuació es descriuen els codis i objectes presents en aquest GitHub:

### Codi epidemiologic:
**main.c**: Fitxer que conté el simulador epidemiologic, ha estat modificat perquè llegís les dades d'entrada des d'un fitxer extern.

**list.h**: Fitxer adjunt al main.c que té alguns paràmetres i funcions que s'usen en el main.c.

**IBMheader.h**: Fitxer adjunt al main.c que té alguns paràmetres i funcions que s'usen en el main.c.

**simulacions.py**: Fitxer en python que a partir d'un fitxer 'sample.txt' va simulant cada un dels sets de paràmetres continguts en el fitxer de lectura en el simulador epidemiologic.

**executable**: Si l'executes s'encarrega de compilar tots els programes de C i que s'escrigui l'output en el fitxer 'results_new.xlsx'.

**results_new.xlsx**: Fitxer amb el conjunt de dades usat en els models de machine learning que s'han dut a terme.
### Mostreig:
**LHS.c**: Codi en c que construeix el mostreig usant Latin Hypercube Sampling.

**MC.c**: Codi en c que construeix el mostreig usant Monte Carlo.

### Models en Machine Learning:
**model_primer_any.py**: Model del primer any calculat pels diferents algoritmes de Machine Learning.

**model_primer_any_Nvars.py**: Model del primer any calculat pels diferents algoritmes de Machine Learning on es pot veure la dependencia amb el nombre de paràmetres d'entrada que s'usen per a l'entrenament.

**model_primer_any_Ndata.py**: Model del primer any calculat pels diferents algoritmes de Machine Learning on es pot veure la dependencia amb la quantitat de dades que s'usa com a entrenament.

**model_futurs_anys.py**: Model pels futus anys  calculat pels diferents algoritmes de Machine Learning.

**model_futurs_anys_Nvars.py**: Model pels futus anys calculat pels diferents algoritmes de Machine Learning on es pot veure la dependencia amb el nombre de paràmetres d'entrada que s'usen per a l'entrenament.

**model_futurs_anys_Ndata.py**: Model pels futus anys  calculat pels diferents algoritmes de Machine Learning on es pot veure la dependencia amb la quantitat de dades que s'usa com a entrenament.

**model_prediccio_futur.py**: Model per la predicció del futur calculat pels diferents algoritmes de Machine Learning.

**model_prediccio_futur_Nvars.py**: Model per la predicció del futur calculat pels diferents algoritmes de Machine Learning on es pot veure la dependencia amb el nombre de paràmetres d'entrada que s'usen per a l'entrenament.

**model_prediccio_futur_Ndata.py**: Model per la predicció del futur calculat pels diferents algoritmes de Machine Learning on es pot veure la dependencia amb la quantitat de dades que s'usa com a entrenament.

### Ajustar models:
**ajustar_GNB.py**: Calcula la millor configuració per l'algoritme de Machine Learning: Gaussian Naive Bayes.

**ajustar_KNC.py**: Calcula la millor configuració per l'algoritme de Machine Learning: KNeighbors Classifier.

**ajustar_LDA.py**: Calcula la millor configuració per l'algoritme de Machine Learning: Linear Discriminant Analysis.

**ajustar_LR.py**: Calcula la millor configuració per l'algoritme de Machine Learning: Logistic Regression.

**ajustar_RF.py**: Calcula la millor configuració per l'algoritme de Machine Learning: Random Forest.

**ajustar_SVC.py**: Calcula la millor configuració per l'algoritme de Machine Learning: Support Vector Machine.
