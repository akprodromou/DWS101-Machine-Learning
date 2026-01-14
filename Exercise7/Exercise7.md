# DWS101 - Φυσικές Καταστροφές & Κοινωνικά Δίκτυα

## Project: Αυτοματοποιημένο Σύστημα Πρόληψης και Εκκένωσης μέσω Ανάλυσης Tweet

### Εισαγωγή

Στην παρούσα ανάλυση, αναπτύσσουμε ένα σύστημα τεχνητής νοημοσύνης που στοχεύει στην έγκαιρη ανίχνευση φυσικών καταστροφών μέσω της ανάλυσης αναρτήσεων στην πλατφόρμα X (Twitter). Η κρισιμότητα του συστήματος έγκειται στην ταχύτητα απόκρισης: αν μια ανάρτηση κριθεί ότι αναφέρεται σε πραγματική καταστροφή (target=1), το σύστημα δύναται να ενεργοποιήσει το πρωτόκολλο 112.

Στόχος είναι η δημιουργία ενός μοντέλου ταξινόμησης κειμένου που θα διακρίνει τις πραγματικές καταστροφές από μεταφορικές χρήσεις λέξεων (π.χ. "The concert was fire").

---

### Μεταφόρτωση Βιβλιοθηκών

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string

# Scikit-learn εργαλεία
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC, SVC
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# flow chart
import mermaid as md
from IPython.display import Image
```

---

## Εισαγωγή Δεδομένων και Διαχωρισμός

Θα χρησιμοποιήσουμε το dataset "Natural Language Processing with Disaster Tweets" από το Kaggle.

```python
# Φόρτωση δεδομένων (υποθέτουμε ότι το αρχείο έχει μεταφορτωθεί)
df = pd.read_csv("train.csv")

# Διαχωρισμός σε Train και Test (90-10%) με στρωματοποίηση (stratify)
# Η επιλογή stratify εξασφαλίζει ότι η αναλογία των κλάσεων παραμένει ίδια στα δύο σύνολα
train_df, test_df = train_test_split(df, test_size=0.10, stratify=df['target'], random_state=0)
```

---

### Ερώτημα 1: Οπτικοποίηση Target & Ισορροπία Dataset

```python
# Υπολογισμός πλήθους
train_counts = train_df['target'].value_counts()
test_counts = test_df['target'].value_counts()

# Δημιουργία Bar-plot
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
train_counts.plot(kind='bar', color=['#40798C', '#8c4053'], ax=ax[0])
ax[0].set_title('Κατανομή Target στο Train Set')
test_counts.plot(kind='bar', color=['#40798C', '#8c4053'], ax=ax[1])
ax[1].set_title('Κατανομή Target στο Test Set')
plt.show()
```

**Σχολιασμός:** Το dataset εμφανίζει μια ελαφριά ανισορροπία (περισσότερα target=0 από target=1), ωστόσο δεν θεωρείται εξαιρετικά ασύμμετρο (imbalanced) ώστε να απαιτεί τεχνικές όπως το SMOTE, αλλά η χρήση της μετρικής F1-Score είναι απαραίτητη.

---

## Ερώτημα 2-3-4: Προεπεξεργασία Δεδομένων

Για την προεπεξεργασία, θα ακολουθήσουμε τη ροή που περιγράφεται παρακάτω:

```python
# Αφαίρεση id, location
train_df = train_df.drop(['id', 'location'], axis=1)
test_df = test_df.drop(['id', 'location'], axis=1)

# Συμπλήρωση ελλιπών τιμών στο keyword
train_df['keyword'] = train_df['keyword'].fillna('null')
test_df['keyword'] = test_df['keyword'].fillna('null')

# Ένωση keyword και text
train_df['inputs'] = train_df['keyword'] + ' ' + train_df['text']
test_df['inputs'] = test_df['keyword'] + ' ' + test_df['text']
```

---

### Τεχνικές Καθαρισμού Κειμένου (Text Preprocessing)

Θα εφαρμόσουμε τις τεχνικές που βασίζονται στο αποθετήριο Deffro/text-preprocessing-techniques:

1. **Remove URLs**: Αφαίρεση συνδέσμων που δεν προσφέρουν σημασιολογική αξία στην ανίχνευση καταστροφής.
2. **Remove HTML tags**: Καθαρισμός ετικετών που προκύπτουν από web scraping.
3. **Remove Unicode/Emojis**: Αφαίρεση μη-ASCII χαρακτήρων για μείωση του θορύβου στα διανύσματα.
4. **Remove Punctuation**: Αφαίρεση σημείων στίξης που αυξάνουν τη διάσταση του λεξιλογίου χωρίς λόγο.

**Αιτιολόγηση:** Επιλέχθηκαν αυτές οι τεχνικές για να μειωθεί η "αραιότητα" (sparsity) του πίνακα TF-IDF και να επικεντρωθεί το μοντέλο στις λέξεις-κλειδιά.

```python
def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www.\S+', '', text)  # URLs
    text = re.sub(r'<.*?>', '', text)  # HTML
    text = re.sub(r'[^\x00-\x7f]', r'', text)  # Unicode
    text = text.translate(str.maketrans('', '', string.punctuation))  # Στίξη
    return text

train_df['inputs'] = train_df['inputs'].apply(clean_text)
test_df['inputs'] = test_df['inputs'].apply(clean_text)

# Μετατροπή σε Numpy Arrays
X_train_raw = train_df['inputs'].values
y_train = train_df['target'].values
X_test_raw = test_df['inputs'].values
y_test = test_df['target'].values
```

---

## Ερώτημα 5: Μετασχηματισμός TF-IDF

```python
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train_raw)
X_test_tfidf = vectorizer.transform(X_test_raw)
```

---

## Ερώτημα 6: Εκπαίδευση LinearSVC & Μετρικές

```python
lsvc = LinearSVC(random_state=0)
lsvc.fit(X_train_tfidf, y_train)
y_pred_lsvc = lsvc.predict(X_test_tfidf)

def get_metrics(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred)
    }

metrics_lsvc = get_metrics(y_test, y_pred_lsvc)
print("LinearSVC Metrics:", metrics_lsvc)
```

**Ποια μετρική είναι πιο σημαντική?**

Στο συγκεκριμένο project, η πιο σημαντική μετρική είναι το **Recall**.

**Αιτιολόγηση:** Σε ένα σύστημα που συνδέεται με το 112, το κόστος ενός "False Negative" (να συμβαίνει καταστροφή και το σύστημα να μην την εντοπίσει) είναι δυνητικά ανθρώπινες ζωές. Αντίθετα, ένα "False Positive" (λάθος συναγερμός) είναι διαχειρίσιμο.

---

## Ερώτημα 7: Μετασχηματισμός PCA

Το PCA απαιτεί πυκνούς πίνακες, οπότε θα μετατρέψουμε τα δεδομένα.

```python
# Διατήρηση 95% της πληροφορίας
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_tfidf.toarray())
X_test_pca = pca.transform(X_test_tfidf.toarray())

# Εκπαίδευση μοντέλων μετά το PCA
lsvc_pca = LinearSVC(random_state=0)
lsvc_pca.fit(X_train_pca, y_train)
y_pred_lsvc_pca = lsvc_pca.predict(X_test_pca)

svc_rbf = SVC(kernel='rbf', random_state=0)  # default gamma='scale'
svc_rbf.fit(X_train_pca, y_train)
y_pred_rbf = svc_rbf.predict(X_test_pca)
```

---

## Ερώτημα 8: Παράμετρος Gamma & Σύγκριση

**Θεωρητικό υπόβαθρο:**

* `'scale'` (default): 1 / (n_features * X.var()) → Προσαρμόζεται στη διακύμανση των δεδομένων.
* `'auto'`: 1 / n_features → Χρησιμοποιεί μόνο το πλήθος των χαρακτηριστικών.

```python
svc_rbf_auto = SVC(kernel='rbf', gamma='auto', random_state=0)
svc_rbf_auto.fit(X_train_pca, y_train)
y_pred_rbf_auto = svc_rbf_auto.predict(X_test_pca)

# Συλλογή μετρικών για τα 4 μοντέλα
results = {
    "LinearSVC": metrics_lsvc,
    "LinearSVC_PCA": get_metrics(y_test, y_pred_lsvc_pca),
    "SVC_RBF_Scale": get_metrics(y_test, y_pred_rbf),
    "SVC_RBF_Auto": get_metrics(y_test, y_pred_rbf_auto)
}

# Bar-plot σύγκρισης
res_df = pd.DataFrame(results).T
res_df.plot(kind='bar', figsize=(12, 6))
plt.title('Σύγκριση Μοντέλων (PCA-based)')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.legend(loc='lower right')
plt.show()
```

---

## Ερώτημα 9: Gaussian Naive Bayes (Χωρίς PCA)

```python
gnb = GaussianNB()

# Το GaussianNB απαιτεί πυκνό πίνακα (dense array)
gnb.fit(X_train_tfidf.toarray(), y_train)
y_pred_gnb = gnb.predict(X_test_tfidf.toarray())

metrics_gnb = get_metrics(y_test, y_pred_gnb)
print("GaussianNB Metrics (Original TF-IDF):", metrics_gnb)
```

---

### Τελικά Συμπεράσματα

Παρατηρούμε ότι ο **LinearSVC** στα αρχικά δεδομένα TF-IDF τείνει να υπερτερεί σε ταχύτητα και απόδοση. Η εφαρμογή του PCA, ενώ μειώνει τη διάσταση, συχνά οδηγεί σε μικρή απώλεια ακρίβειας λόγω της φύσης των κειμενικών δεδομένων (sparse features). Ο Naive Bayes λειτουργεί ως ένα καλό baseline, αλλά υστερεί στην κατανόηση των συσχετίσεων λέξεων σε σχέση με τους SVMs.