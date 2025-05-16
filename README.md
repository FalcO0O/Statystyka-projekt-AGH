# Statystyka-projekt-AGH

#### Repozytorium zawiera projekt przygotowany w ramach kursu Rachunek prawdopodobieństwa i statystyka, realizowanego na kierunku Informatyka na Akademii Górniczo-Hutniczej w Krakowie.

## MultinomialNaiveBayesClassifier

Klasyfikator `MultinomialNaiveBayesClassifier` implementuje model Naive Bayes dla danych kategorycznych, oparty na rozkładzie wielomianowym.

---

### Jak działa?

1. **Trening (fit)**:
   - Oblicza logarytmiczne prawdopodobieństwa a priori dla każdej klasy:
     ```text
     P(y) = liczba_przykładów_klasy_y / liczba_wszystkich_przykładów
     ```
   - Oblicza logarytmiczne prawdopodobieństwa warunkowe dla każdej wartości cechy względem klasy:
     ```text
     P(x_i | y) = liczba_przykładów_klasy_y_z_cechą_x_i / liczba_wszystkich_przykładów_klasy_y
     ```
   - Jeśli prawdopodobieństwo wynosi 0, przypisywana jest wartość (`epsilon`). Po podniesieniu jej do potęgi liczby (`e`), wynik jest bliski zera, co zapobiega błędom numerycznym. Typowe wartości `epsilon` mieszczą się w zakresie od -10 do -100 (w skali logarytmicznej).

2. **Predykcja (predict, predict_proba)**:
   - Oblicza sumę logarytmów prawdopodobieństw:
     - Logarytmu prawdopodobieństwa a priori.
     - Logarytmów warunkowych prawdopodobieństw dla każdej cechy w wierszu danych.
     - W przypadku braku wartości przypisuje wcześniej zadeklarowaną wartość (`epsilon`).
   - Zwraca klasę o najwyższym prawdopodobieństwie (`predict`) lub unormowane prawdopodobieństwa dla wszystkich klas (`predict_proba`).

---

### Przykład użycia

1. **Trening modelu**:
   ```python
   
   mnb = MultinomialNaiveBayesClassifier()
   mnb.fit(X_train, column_name="class", epsilon=-40)
    ```
### GaussianNaiveBayesClassifier

Klasyfikator `GaussianNaiveBayesClassifier` implementuje model Naive Bayes dla danych ilościowych, oparty na rozkładzie normalnym (Gaussa).

---

### Jak działa?

1. **Trening (`fit`)**:
   - Oblicza logarytmiczne prawdopodobieństwa a priori dla każdej klasy:
     ```text
     P(y) = liczba_przykładów_klasy_y / liczba_wszystkich_przykładów
     ```
   - Oblicza średnią i odchylenie standardowe dla każdej cechy względem klasy:
     - Średnia reprezentuje typową wartość cechy w danej klasie.
     - Odchylenie standardowe określa rozrzut wartości cechy wokół średniej.
   - Te parametry są przechowywane i używane w obliczeniach funkcji gęstości Gaussa.

2. **Funkcja gęstości Gaussa (`log_gaussian_density`)**:
   - Model oblicza logarytm gęstości rozkładu normalnego dla każdej cechy względem klasy:
     ```text
     log(P(x | y)) = -0.5 * log(2 * pi * sigma^2) - ((x - mu)^2) / (2 * sigma^2)
     ```
   - W przypadku wartości bardzo małych `sigma^2` (odchylenie standardowe bliskie 0), dodawana jest mała wartość, aby uniknąć błędów numerycznych.

3. **Predykcja (`predict`, `predict_proba`)**:
   - Oblicza sumę logarytmicznych prawdopodobieństw:
     - Logarytmów prawdopodobieństw a priori.
     - Logarytmów funkcji gęstości Gaussa dla każdej cechy.
     - W przypadku braku wartości przypisuje wcześniej zadeklarowaną wartość `epsilon`.
   - Zwraca:
     - Klasy o najwyższym prawdopodobieństwie (`predict`).
     - Unormowane prawdopodobieństwa wszystkich klas po ich unormowaniu (`predict_proba`).

---

### Przykład użycia

1. **Trening modelu**:
    ```python
   gnb = GaussianNaiveBayesClassifier()
   gnb.fit(X_train, column_name="class", epsilon=-40)
   ```
