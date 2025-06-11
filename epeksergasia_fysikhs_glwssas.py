import random # για παραδοτέο 1 α
from transformers import pipeline # για παραδοτέο 1 Β
from sentence_transformers import SentenceTransformer,util # για παραδοτέο 2 
import re # για παραδοτέο 2 custom
import logging # Για τις ενημερώσεις και τα warnings
logging.getLogger("transformers").setLevel(logging.ERROR)
import warnings 
warnings.filterwarnings('ignore') # Για τις ενημερώσεις και τα warnings
import matplotlib.pyplot as plt # Για οπτικοποίηση ενσωματομένων λέξεων
from sklearn.decomposition import PCA # Κλάση pca
import numpy as np
import gensim.downloader as api # Κατεβάζει την πρώτη φορά που εκτελείται , το μοντέλο

# Παραδοτέο 1

# A)
sentences = [
    "Hope you too, to enjoy it as my deepest wishes.",
    "Overall, let us make sure all are safe and celebrate the outcome with strong coffee and future targets."
]

synonyma = {
    "hope": ["wish","pray"],
    "enjoy": ["like"],
    "deepest":["best"],
    "wishes":["prayers"],
    "overall":["altogether","In the end"],
    "safe":["secured"],
    "celebrate":["enjoy"],
    "outcome":["result"],
    "strong":["rich","intense"],
    "future":["upcoming","next"],
    "targets":["goals","achievements"]
}

def cnstr(sentences):

    new_sentences = []

    for s in sentences:

        words = s.split()
        new_sen =[]

        for word in words: 
        
            key_word = word.lower().strip(".,!;")

            if key_word in synonyma:

                new_word = random.choice(synonyma[key_word])

                if word[0].isupper():

                    new_word = new_word.capitalize()

                new_sen.append(new_word)
            else:

                new_sen.append(word)

        new_sentences.append(" ".join(new_sen))
    return new_sentences

rec_sen = cnstr(sentences)

# Output των προτάσεων του δικού μου αυτόματου

print("Ανακατασκευή δικού μου αυτόματου")
print("-" * 100)
print("-" * 100)
print("-" * 100)
print(f"Ανακατασκευή της πρώτης πρότασης  : {rec_sen[0]}")
print(f"Ανακατασκευή της δεύτερης πρότασης  : {rec_sen[1]}")



# Β) 


keimeno1=[
    "Today is our dragon boat festival, in our Chinese culture, to celebrate it with all safe and great in our lives."," Hope you too, to enjoy it as my deepest wishes."," Thank your message to show our words to the doctor, as his next contract checking, to all of us."," I got this message to see the approved message."," In fact, I have received the message from the professor, to show me, this, a couple of days ago."," I am very appreciated the full support of the professor, for our Springer proceedings publication"    
]
keimeno2=[
    "During our final discuss, I told him about the new submission — the one we were waiting sincelast autumn, but the updates was confusing as it not included the full feedback from reviewer or maybe editor?"," Anyway, i believe the team, although bit delay and less communication at recent days, they really tried best for paper and cooperation."," We should be grateful, I mean all of us, for the acceptanceand efforts until the Springer link came finally last week, i think."," Also, kindly remind me please, if the doctor still plan for the acknowledgments section edit beforehe sending again."," Because I didn’t see that part final yet, or maybe I missed, I apologize if so. Overall, let us make sure all are safe and celebrate the outcome with strong coffee and future targets"
]


# 1)


paraphraser = pipeline("text2text-generation", model="tuner007/pegasus_paraphrase")


def pegasus(text, num_return_sequences=3):

    results = paraphraser(text, num_return_sequences=num_return_sequences, clean_up_tokenization_spaces=True)
    return [res['generated_text'] for res in results]

tel_keim1_1 = ''
tel_keim2_1 = ''

for s in keimeno1:

    phrase = pegasus(s)
    tel_keim1_1 += phrase[0]  

for s in keimeno2:

    phrase = pegasus(s)
    tel_keim2_1 += phrase[0]

print("-" * 100)
print("-" * 100)
print("-" * 100)

print(f"Παράφραση πρώτου κείμενου με Pegasus : {tel_keim1_1}")
print("-" * 50)
print(f"Παράφραση δεύτερου κείμενου με Pegasus : {tel_keim2_1}")

print("-" * 100)
print("-" * 100)
print("-" * 100)

# 2)

paraphraser = pipeline("text2text-generation", model="facebook/bart-large-cnn")

tel_keim1_2= ''
tel_keim2_2 = ''

for s in keimeno1:

    result = paraphraser(s, max_length=100, num_return_sequences=1)
    tel_keim1_2+= result[0]['generated_text']+"  "
for s in keimeno2:

    result = paraphraser(s, max_length=100, num_return_sequences=1)
    tel_keim2_2+= result[0]['generated_text']+ " "

print(f"Παράφραση πρώτου κείμενου με BART - large : {tel_keim1_2}")
print("-"*50)
print(f"Παράφραση δεύτερου κείμενου με BART - large : {tel_keim2_2}")

print("-" * 100)
print("-" * 100)
print("-" * 100)


# 3)

paraphraser = pipeline("text2text-generation", model="ramsrigouthamg/t5_paraphraser")

tel_keim1_3= ''
tel_keim2_3 = ''

for s in keimeno1:

    phrase = paraphraser(f"paraphrase: {s} </s>", max_length=60, num_return_sequences=1, clean_up_tokenization_spaces=True)
    tel_keim1_3 += phrase[0]['generated_text']

for s in keimeno2:

    phrase = paraphraser(f"paraphrase: {s} </s>", max_length=60, num_return_sequences=1, clean_up_tokenization_spaces=True)
    tel_keim2_3 += phrase[0]['generated_text']


print(f"Παράφραση πρώτου κειμένου με Ramsrigouthamg : {tel_keim1_3}")
print("-"*50)
print(f"Παράφραση δεύτερου κειμένου με Ramsrigouthamg : {tel_keim2_3}")

print("-" * 100)
print("-" * 100)
print("-" * 100)

# Παραδοτέο 2

keimeno_1_oloklhro=" Today is our dragon boat festival, in our Chinese culture, to celebrate it with all safe and great in our lives. Hope you too, to enjoy it as my deepest wishes.Thank your message to show our words to the doctor, as his next contract checking, to all of us. I got this message to see the approved message. In fact, I have received the message from the professor, to show me, this, a couple of days ago. I am very appreciated the full support of the professor, for our Springer proceedings publication"
keimeno_2_oloklhro="During our final discuss, I told him about the new submission — the one we were waiting since last autumn, but the updates was confusing as it not included the full feedback from reviewer or maybe editor? Anyway, I believe the team, although bit delay and less communication at recent days, they really tried best for paper and cooperation. We should be grateful, I mean all of us, for the acceptance and efforts until the Springer link came finally last week, I think. Also, kindly remind me please, if the doctor still plan for the acknowledgments section edit before he sending again. Because I didn’t see that part final yet, or maybe I missed, I apologize if so. Overall, let us make sure all are safe and celebrate the outcome with strong coffee and future targets"

def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)  
    text = text.lower()
    return text

keimeno_1_oloklhro= clean_text(keimeno_1_oloklhro)
keimeno_2_oloklhro= clean_text(keimeno_2_oloklhro)
tel_keim1_1= clean_text(tel_keim1_1)
tel_keim2_1= clean_text(tel_keim2_1)
tel_keim1_2= clean_text(tel_keim1_2)
tel_keim2_2= clean_text(tel_keim2_2)
tel_keim1_3= clean_text(tel_keim1_3)
tel_keim2_3= clean_text(tel_keim2_3)

model = SentenceTransformer('all-MiniLM-L6-v2') 
model2 = api.load("word2vec-google-news-300")



def cos_ypologismos(arxiko,teliko,minima):
    embending1 = model.encode(arxiko, convert_to_tensor=True)
    embending2 = model.encode(teliko, convert_to_tensor=True)
    similarity = util.cos_sim(embending1, embending2)
    print(f"Cosine similarity {minima}: {similarity.item():.4f}")

def pca(arxiko_text, teliko_text, model,label):
    
    arxiko_words = [word for word in arxiko_text.split() if word in model]
    teliko_words = [word for word in teliko_text.split() if word in model]

    arxiko_vectors = [model[word] for word in arxiko_words]
    teliko_vectors = [model[word] for word in teliko_words]

    all_vectors = np.vstack((arxiko_vectors, teliko_vectors))
 
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(all_vectors)     
    plt.figure(figsize=(15, 8))

    for i, word in enumerate(arxiko_words):

        plt.scatter(reduced[i, 0], reduced[i, 1], color='blue')
        plt.annotate(f'{word} (A)', (reduced[i, 0], reduced[i, 1]),fontsize=5)


    for j, word in enumerate(teliko_words):

        plt.scatter(reduced[i+1+j, 0], reduced[i+1+j, 1], color='red')
        plt.annotate(f'{word} (B)', (reduced[i+1+j , 0], reduced[i+1+j , 1]),fontsize=5)
  
    plt.title(f"Οπτικοποίηση PCA των embeddings λέξεων {label}")
    plt.show()



# Σύγκριση για ερώτημα Α από παραδοτέο 1

cos_ypologismos(sentences[0],rec_sen[0],"πρώτης πρότασης δικού μου αυτόματου")
cos_ypologismos(sentences[1],rec_sen[1],"δεύτερης πρότασης δικού μου αυτόματου")
pca(sentences[0],rec_sen[0],model2,"πρώτης πρότασης δικού μου αυτόματου")
pca(sentences[1],rec_sen[1],model2,"δεύτερης πρότασης δικού μου αυτόματου")
print("-" * 100)
print("-" * 100)
print("-" * 100)


# Για παραφράσεις pegasus

cos_ypologismos(keimeno_1_oloklhro,tel_keim1_1,"πρώτου κειμένου Pegasus")
cos_ypologismos(keimeno_2_oloklhro,tel_keim2_1,"δεύτερου κειμένου Pegasus")
pca(keimeno_1_oloklhro,tel_keim1_1,model2,"πρώτου κειμένου με Pegasus")
pca(keimeno_2_oloklhro,tel_keim2_1,model2,"δεύτερου κειμένου Pegasus")
print("-" * 100)
print("-" * 100)
print("-" * 100)


# Για παραφράσεις BART - large

cos_ypologismos(keimeno_1_oloklhro,tel_keim1_2,"πρώτου κειμένου BART -large")
cos_ypologismos(keimeno_2_oloklhro,tel_keim2_2,"δεύτερου κειμένου BART -large")
pca(keimeno_1_oloklhro,tel_keim1_2,model2,"πρώτου κειμένου BART -large")
pca(keimeno_2_oloklhro,tel_keim2_2,model2,"δεύτερου κειμένου BART -large")
print("-" * 100)
print("-" * 100)
print("-" * 100)


# Για παραφράσεις Ramsrigouthamg

cos_ypologismos(keimeno_1_oloklhro,tel_keim1_3,"πρώτου κειμένου Ramsrigouthamg")
cos_ypologismos(keimeno_2_oloklhro,tel_keim2_3,"δεύτερου κειμένου Ramsrigouthamg")
pca(keimeno_1_oloklhro,tel_keim1_3,model2,"πρώτου κειμένου Ramsrigouthamg")
pca(keimeno_2_oloklhro,tel_keim2_3,model2,"δεύτερου κειμένου Ramsrigouthamg")




