########## 2. Read and preprocess raw data ##########
import os
import re
import numpy as np
import pandas as pd
import tensorflow as tf
import nltk
import pickle
from tensorflow.keras.layers import Input, Embedding, Conv1D, Dense, Dropout, Concatenate, GlobalMaxPooling1D, Lambda
from nltk.corpus import wordnet, stopwords
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from spellchecker import SpellChecker
import argparse

parser = argparse.ArgumentParser(description="Sentiment Analysis Tool")
parser.add_argument('--project', type=str, default="caffe", help="Project name for dataset (default: caffe)")
parser.add_argument('--kfold', type=bool, default=False, help="Whether to perform K-fold cross-validation (default: False)")
args = parser.parse_args()

project = args.project
path = f"datasets/{project}.csv"
do_kfold = args.kfold  

########## 2. Read and preprocess raw data ##########
pd_all = pd.read_csv(path)
pd_all = pd_all.sample(frac=1, random_state=999)

pd_all['Title+Body'] = pd_all.apply(
    lambda row: row['Title'] + '. ' + row['Body'] if pd.notna(row['Body']) else row['Title'],
    axis=1
)

pd_tplusb = pd_all.rename(columns={
    "Unnamed: 0": "id",
    "class": "sentiment",
    "Title+Body": "text"
})
pd_tplusb.to_csv('Title+Body.csv', index=False, columns=["id", "Number", "sentiment", "text"])

########## 3. Load cleaned dataset ##########
datafile = 'Title+Body.csv'
data = pd.read_csv(datafile).fillna('')
lemmatizer = WordNetLemmatizer()

NLTK_stop_words_list = stopwords.words('english')
custom_stop_words_list = ['...']
final_stop_words_list = NLTK_stop_words_list + custom_stop_words_list

def remove_html(text):
    return re.sub(r'<.*?>', '', text)

def remove_emoji(text):
    emoji_pattern = re.compile("["u"\U0001F600-\U0001F64F"
                               u"\U0001F300-\U0001F5FF"
                               u"\U0001F680-\U0001F6FF"
                               u"\U0001F1E0-\U0001F1FF"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251""]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

spell = SpellChecker()

def correct_spelling(text):
    corrected = []
    for word in text.split():
        corrected_word = spell.correction(word)
        corrected.append(corrected_word if corrected_word is not None else word)
    return " ".join(corrected)

def handle_negation(text):
    negation_words = r"(?:not|no|never|none|nothing|nobody|n't)"
    negation_contractions = r"(?:don't|doesn't|didn't|can't|won't|wouldn't|shouldn't|couldn't|isn't|aren't|wasn't|weren't|mustn't|mightn't)"
    pattern = re.compile(rf"\b({negation_words}|{negation_contractions})\s+(\w+)", flags=re.IGNORECASE)
    return pattern.sub(r"\1_\2", text)

def remove_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in final_stop_words_list])

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def lemmatize_with_pos(text):
    tokens = word_tokenize(text)
    lemmatized = []
    for word, tag in pos_tag(tokens):
        wordnet_pos = get_wordnet_pos(tag)
        lemma = lemmatizer.lemmatize(word, pos=wordnet_pos)
        lemmatized.append(lemma)
    return " ".join(lemmatized)

def clean_str(string):
    string = re.sub(r"http\\S+", "", string)
    string = re.sub(r"[^A-Za-z0-9 ]+", " ", string)
    string = re.sub(r"\d+", "", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

import random
from nltk.corpus import wordnet

def synonym_replacement(sentence, n=2):
    words = sentence.split()
    new_words = words.copy()
    random_word_list = list(set([word for word in words if word not in final_stop_words_list]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms))
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break
    return ' '.join(new_words)

def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
            if synonym != word:
                synonyms.add(synonym)
    return synonyms

data['text'] = data['text'].apply(remove_html)
data['text'] = data['text'].apply(remove_emoji)
data['text'] = data['text'].apply(remove_stopwords)
print(1)
# data['text'] = data['text'].apply(correct_spelling)
print(2)
data['text'] = data['text'].apply(handle_negation)
print(3)
data['text'] = data['text'].apply(clean_str)
data['text'] = data['text'].apply(lemmatize_with_pos)

labels = data['sentiment'].values

########## 4. Tokenize and Convert Text to Sequences ##########
MAX_NUM_WORDS = 2000
MAX_SEQUENCE_LENGTH = 300
EMBEDDING_DIM = 100

tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(data['text'])

with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

sequences = tokenizer.texts_to_sequences(data['text'])
X = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

augment_times = 0
X_train_augmented = []
y_train_augmented = []

for i in range(len(X_train)):
    original_text = ' '.join([tokenizer.index_word.get(idx, '') for idx in X_train[i] if idx > 0])
    X_train_augmented.append(X_train[i])
    y_train_augmented.append(y_train[i])
    for _ in range(augment_times):
        augmented_text = synonym_replacement(original_text, n=2)
        augmented_seq = tokenizer.texts_to_sequences([augmented_text])
        augmented_pad = pad_sequences(augmented_seq, maxlen=MAX_SEQUENCE_LENGTH)
        X_train_augmented.append(augmented_pad[0])
        y_train_augmented.append(y_train[i])

X_train_augmented = np.array(X_train_augmented)
y_train_augmented = np.array(y_train_augmented)

class_weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

def reduce_mean_axis1(x):
    return tf.reduce_mean(x, axis=1)

input_layer = Input(shape=(MAX_SEQUENCE_LENGTH,))
embedding = Embedding(input_dim=MAX_NUM_WORDS, output_dim=EMBEDDING_DIM)(input_layer)
conv1 = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(embedding)
conv2 = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(conv1)
conv3 = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(conv2)
x = Concatenate()([conv1, conv3])
x = GlobalMaxPooling1D()(x)
x = Dropout(0.5)(x)
x = Dense(64, activation='relu')(x)
output_layer = Dense(1, activation='sigmoid')(x)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(loss='binary_crossentropy',
              optimizer=Adam(learning_rate=0.001),
              metrics=['accuracy'])
model.summary()

checkpoint = ModelCheckpoint("best_model.keras", monitor="val_accuracy", save_best_only=True, verbose=1, mode="max")

class_weights_augmented = compute_class_weight(
    "balanced", classes=np.array([0, 1]), y=y_train_augmented
)
class_weight_dict_augmented = {i: class_weights_augmented[i] for i in range(len(class_weights_augmented))}

history = model.fit(X_train_augmented, y_train_augmented, epochs=10, batch_size=64,
                    validation_data=(X_test, y_test),
                    class_weight=class_weight_dict_augmented, callbacks=[checkpoint])

best_model = load_model("best_model.keras")
best_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

y_pred = (best_model.predict(X_test) > 0.4).astype("int32")
y_pred_proba = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
auc = roc_auc_score(y_test, y_pred_proba)
print("\n=== Best Model Evaluation ===")
print(f"Accuracy:      {accuracy:.4f}")
print(f"Precision:     {precision:.4f}")
print(f"Recall:        {recall:.4f}")
print(f"F1 Score:      {f1:.4f}")
print(f"AUC:           {auc:.4f}")

if do_kfold:
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_no = 1
    accuracy_scores, precision_scores, recall_scores, f1_scores = [], [], [],[]

    for train_idx, val_idx in kfold.split(X, y):
        print(f"\n===== Stratified Fold {fold_no} =====")

        X_fold_train, X_fold_val = X[train_idx], X[val_idx]
        y_fold_train, y_fold_val = y[train_idx], y[val_idx]

        train_texts = [' '.join([tokenizer.index_word.get(idx, '') for idx in seq if idx > 0]) for seq in X_fold_train]
        fold_tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
        fold_tokenizer.fit_on_texts(train_texts)

        X_fold_train_augmented, y_fold_train_augmented = [], []

        for i in range(len(X_fold_train)):
            original_text = ' '.join([fold_tokenizer.index_word.get(idx, '') for idx in X_fold_train[i] if idx > 0])
            X_fold_train_augmented.append(X_fold_train[i])
            y_fold_train_augmented.append(y_fold_train[i])
            for _ in range(augment_times):
                augmented_text = synonym_replacement(original_text, n=2)
                augmented_seq = fold_tokenizer.texts_to_sequences([augmented_text])
                augmented_pad = pad_sequences(augmented_seq, maxlen=MAX_SEQUENCE_LENGTH)
                X_fold_train_augmented.append(augmented_pad[0])
                y_fold_train_augmented.append(y_fold_train[i])

        X_fold_train_augmented = np.array(X_fold_train_augmented)
        y_fold_train_augmented = np.array(y_fold_train_augmented)

        val_sequences = fold_tokenizer.texts_to_sequences([' '.join([fold_tokenizer.index_word.get(idx, '') for idx in seq if idx > 0]) for seq in X_fold_val])
        X_fold_val = pad_sequences(val_sequences, maxlen=MAX_SEQUENCE_LENGTH)

        class_weights_fold = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_fold_train_augmented)
        class_weight_dict_fold = {i: class_weights_fold[i] for i in range(len(class_weights_fold))}

        input_layer = Input(shape=(MAX_SEQUENCE_LENGTH,))
        embedding = Embedding(input_dim=MAX_NUM_WORDS, output_dim=EMBEDDING_DIM)(input_layer)
        conv1 = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(embedding)
        conv2 = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(conv1)
        conv3 = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(conv2)
        x = Concatenate()([conv1, conv3])
        x = GlobalMaxPooling1D()(x)
        x = Dropout(0.5)(x)
        x = Dense(64, activation='relu')(x)
        output_layer = Dense(1, activation='sigmoid')(x)

        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

        checkpoint = ModelCheckpoint(f"corrected_model_fold_{fold_no}.keras",
                                     monitor="val_accuracy",
                                     save_best_only=True, mode="max")

        model.fit(X_fold_train_augmented, y_fold_train_augmented, epochs=30, batch_size=64,
                  validation_data=(X_fold_val, y_fold_val),
                  class_weight=class_weight_dict_fold, callbacks=[checkpoint], verbose=1)

        best_model = load_model(f"corrected_model_fold_{fold_no}.keras",
                                custom_objects={"reduce_mean_axis1": reduce_mean_axis1})
        y_pred_fold = (best_model.predict(X_fold_val) > 0.5).astype("int32")

        accuracy_scores.append(accuracy_score(y_fold_val, y_pred_fold))
        precision_scores.append(precision_score(y_fold_val, y_pred_fold, average='macro', zero_division=0))
        recall_scores.append(recall_score(y_fold_val, y_pred_fold, average='macro', zero_division=0))
        f1_scores.append(f1_score(y_fold_val, y_pred_fold, average='macro', zero_division=0))

        print(f"\nFold {fold_no} results:")
        print(f"Accuracy: {accuracy_scores[-1]:.4f}")
        print(f"Precision: {precision_scores[-1]:.4f}")
        print(f"Recall: {recall_scores[-1]:.4f}")
        print(f"F1 Score: {f1_scores[-1]:.4f}")

        fold_no += 1

    print("\n=== Corrected Stratified K-Fold Cross-Validation Results ===")
    print(f"Average Accuracy:  {np.mean(accuracy_scores):.4f}")
    print(f"Average Precision: {np.mean(precision_scores):.4f}")
    print(f"Average Recall:    {np.mean(recall_scores):.4f}")
    print(f"Average F1 Score:  {np.mean(f1_scores):.4f}")
else:
    print("\nK-fold cross-validation skipped as --kfold is set to False.")
