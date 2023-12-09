import streamlit as st
from time import sleep
from stqdm import stqdm
from annotated_text import annotated_text
from transformers import pipeline
import json
import spacy
import spacy_streamlit
from spacy_streamlit import visualize_parser
import nltk
from nltk.tokenize import word_tokenize
from deep_translator import GoogleTranslator
from textblob import TextBlob
from streamlit_lottie import st_lottie
import requests


@st.cache_resource()
def load_summarizer():
    model = pipeline("summarization", device=-1)
    return model

def generate_chunks(inp_str):
    max_chunk = 500
    inp_str = inp_str.replace('.', '.<eos>')
    inp_str = inp_str.replace('?', '?<eos>')
    inp_str = inp_str.replace('!', '!<eos>')

    sentences = inp_str.split('<eos>')
    current_chunk = 0
    chunks = []
    for sentence in sentences:
        if len(chunks) == current_chunk + 1:
            if len(chunks[current_chunk]) + len(sentence.split(' ')) <= max_chunk:
                chunks[current_chunk].extend(sentence.split(' '))
            else:
                current_chunk += 1
                chunks.append(sentence.split(' '))
        else:
            chunks.append(sentence.split(' '))

    for chunk_id in range(len(chunks)):
        chunks[chunk_id] = ' '.join(chunks[chunk_id])

    return chunks

def draw_all(
        key,
        plot=False,
):
    st.write(
        """
        # NLP Web App

        This Natural Language Processing Based Web App can do anything u can imagine with Text. 😱 

        This App is built using pretrained transformers which are capable of doing wonders with the Textual data.

        ```python
        # Key Features of this App.
        1. Advanced Text Summarizer
        2. Named Entity Recognition
        3. Part-of-speech tagging
        4. Sentiment Analysis
        5. Language Translator
        6. Question Answering
        7. Text Completion
        ```
        """
    )


with st.sidebar:
    draw_all("sidebar")


def main():
    st.title("NLP Web App")
    menu = ["--Select--", "Summarizer", "Named Entity Recognition", "Part-of-speech tagging", "Sentiment Analysis", "Language Translator", "Question Answering",
            "Text Completion"]
    choice = st.sidebar.selectbox("Choose What u wanna do !!", menu)

    if choice == "--Select--":

        st.write("""

                 This is a Natural Language Processing Based Web App that can do anything u can imagine with the Text.
        """)

        st.write("""

                 Natural Language Processing (NLP) is a computational technique to understand the human language in the way they spoke and write.
        """)

        st.write("""

                 NLP is a sub field of Artificial Intelligence (AI) to understand the context of text just like humans.
        """)

        # Fetch the Lottie animation JSON
        url = "https://lottie.host/c3cffc4d-fe40-4eae-9e79-37ed762b70e2/rYqa1jlen1.json"
        response = requests.get(url)
        animation = response.json()

        # Display the Lottie animation in Streamlit
        st_lottie(animation,height=250,width=250)

    elif choice == "Summarizer":
        summarizer = load_summarizer()
        st.title("Summarize Text")
        sentence = st.text_area('Please paste your article :', height=30)
        button = st.button("Summarize")

        max = st.sidebar.slider('Select max', 50, 500, step=10, value=150)
        min = st.sidebar.slider('Select min', 10, 450, step=10, value=50)
        do_sample = st.sidebar.checkbox("Do sample", value=False)
        with st.spinner("Generating Summary.."):
            if button and sentence:
                chunks = generate_chunks(sentence)
                res = summarizer(chunks,
                                 max_length=max,
                                 min_length=min,
                                 do_sample=do_sample)
                text = ' '.join([summ['summary_text'] for summ in res])
                # st.write(result[0]['summary_text'])
                st.write(text)


    elif choice == "Named Entity Recognition":
        nlp = spacy.load("en_core_web_sm")
        st.subheader("Text Based Named Entity Recognition")

        text = st.text_area("Enter the Text below To extract Named Entities !", "")
        if st.button("Analyze"):
            docx = nlp(text)
            for _ in stqdm(range(50), desc="Please wait a bit. The model is fetching the results !!"):
                sleep(0.1)
            spacy_streamlit.visualize_ner(docx, labels=nlp.get_pipe('ner').labels, title="List of Entities")

    elif choice == "Part-of-speech tagging":
        nlp = spacy.load("en_core_web_sm")
        st.subheader("Visualizing the dependency parse and part-of-speech tags")
        text = st.text_area("Enter the Text below To extract PoS !", "")
        docx = nlp(text)
        if st.button("Analyze"):
            for _ in stqdm(range(50), desc="Please wait a bit. The model is fetching the results !!"):
                sleep(0.1)
            visualize_parser(docx, title="")
            annotations = [(str(token), token.pos_) for token in docx]
            annotated_text(*annotations)
            #spacy_streamlit.visualize_similarity(nlp, ("pizza", "fries"))

    elif choice == "Sentiment Analysis":
        st.subheader("Sentiment Analysis")
        raw_text = st.text_area("Enter the Text below To find out its Sentiment !", "")
        if st.button("Analyze"):
            edu = TextBlob(raw_text)
            score = edu.sentiment.polarity
            for _ in stqdm(range(50), desc="Please wait a bit. The model is fetching the results !!"):
                sleep(0.1)
            if score == 0:
                st.info("This text seems Neutral ... 😐")
            elif score < 0:
                st.info("This text has Negative Sentiment😤")
            elif score > 0:
                st.info("This text seems Neutral ... 😐")

    elif choice == "Language Translator":
        st.markdown(
            "<h1 style='text-align: center; color: voilet;font-family: Blippo,fantasy'>Language Translator</h1>",
            unsafe_allow_html=True)
        st.write("****")

        text = st.text_area("Enter text:", height=None, max_chars=None, key=None, help="Enter your text here -")
        st.write("****")

        option1 = st.selectbox('Input language', (
        'english', 'hindi', 'afrikaans', 'albanian', 'amharic', 'arabic', 'armenian', 'azerbaijani', 'basque',
        'belarusian', 'bengali', 'bosnian', 'bulgarian', 'catalan', 'cebuano', 'chichewa', 'chinese',
        'chinese (simplified)', 'chinese (traditional)', 'corsican', 'croatian', 'czech', 'danish', 'dutch',
        'esperanto', 'estonian', 'filipino', 'finnish', 'french', 'frisian', 'galician', 'georgian', 'german', 'greek',
        'gujarati', 'haitian creole', 'hausa', 'hawaiian', 'hebrew', 'hmong', 'hungarian', 'icelandic', 'igbo',
        'indonesian', 'irish', 'italian', 'japanese', 'javanese', 'kannada', 'kazakh', 'khmer', 'korean',
        'kurdish (kurmanji)', 'kyrgyz', 'lao', 'latin', 'latvian', 'lithuanian', 'luxembourgish', 'macedonian',
        'malagasy', 'malay', 'malayalam', 'maltese', 'maori', 'marathi', 'mongolian', 'myanmar (burmese)', 'nepali',
        'norwegian', 'pashto', 'persian', 'polish', 'portuguese', 'punjabi', 'romanian', 'russian', 'samoan',
        'scots gaelic', 'serbian', 'sesotho', 'shona', 'sindhi', 'sinhala', 'slovak', 'slovenian', 'somali', 'spanish',
        'sundanese', 'swahili', 'swedish', 'tajik', 'tamil', 'telugu', 'thai', 'turkish', 'ukrainian', 'urdu', 'uzbek',
        'vietnamese', 'welsh', 'xhosa', 'yiddish', 'yoruba', 'zulu', 'Filipino', 'Hebrew'))
        option2 = st.selectbox('Output language', (
        'english', 'hindi', 'afrikaans', 'albanian', 'amharic', 'arabic', 'armenian', 'azerbaijani', 'basque',
        'belarusian', 'bengali', 'bosnian', 'bulgarian', 'catalan', 'cebuano', 'chichewa', 'chinese',
        'chinese (simplified)', 'chinese (traditional)', 'corsican', 'croatian', 'czech', 'danish', 'dutch',
        'esperanto', 'estonian', 'filipino', 'finnish', 'french', 'frisian', 'galician', 'georgian', 'german', 'greek',
        'gujarati', 'haitian creole', 'hausa', 'hawaiian', 'hebrew', 'hmong', 'hungarian', 'icelandic', 'igbo',
        'indonesian', 'irish', 'italian', 'japanese', 'javanese', 'kannada', 'kazakh', 'khmer', 'korean',
        'kurdish (kurmanji)', 'kyrgyz', 'lao', 'latin', 'latvian', 'lithuanian', 'luxembourgish', 'macedonian',
        'malagasy', 'malay', 'malayalam', 'maltese', 'maori', 'marathi', 'mongolian', 'myanmar (burmese)', 'nepali',
        'norwegian', 'pashto', 'persian', 'polish', 'portuguese', 'punjabi', 'romanian', 'russian', 'samoan',
        'scots gaelic', 'serbian', 'sesotho', 'shona', 'sindhi', 'sinhala', 'slovak', 'slovenian', 'somali', 'spanish',
        'sundanese', 'swahili', 'swedish', 'tajik', 'tamil', 'telugu', 'thai', 'turkish', 'ukrainian', 'urdu', 'uzbek',
        'vietnamese', 'welsh', 'xhosa', 'yiddish', 'yoruba', 'zulu', 'Filipino', 'Hebrew'))
        st.write("****")

        if st.button('Translate Sentence'):
            st.write(" ")
            st.write(" ")
            if text == "":
                st.warning('Please **enter text** for translation')

            else:
                if option1 == option2:
                    st.error("source and target language can't be the same")
                else:
                    translated = GoogleTranslator(source=option1, target=option2).translate(text=text)
                    st.write("Translated text -")
                    st.info(str(translated))

    elif choice == "Question Answering":
        st.subheader("Question Answering")
        st.write(" Enter the Context and ask the Question to find out the Answer !")
        question_answering = pipeline("question-answering")

        context = st.text_area("Context", "Enter the Context Here")

        question = st.text_area("Your Question", "Enter your Question Here")

        if context != "Enter Text Here" and question != "Enter your Question Here":
            result = question_answering(question=question, context=context)
            s1 = json.dumps(result)
            d2 = json.loads(s1)
            generated_text = d2['answer']
            generated_text = '. '.join(list(map(lambda x: x.strip().capitalize(), generated_text.split('.'))))
            st.write(f" Here's your Answer :\n {generated_text}")

    elif choice == "Text Completion":
        st.subheader("Text Completion")
        st.write(" Enter the uncomplete Text to complete it automatically using AI !")
        text_generation = pipeline("text-generation")
        message = st.text_area("Your Text", "Enter the Text to complete")

        if message != "Enter the Text to complete":
            generator = text_generation(message)
            s1 = json.dumps(generator[0])
            d2 = json.loads(s1)
            generated_text = d2['generated_text']
            generated_text = '. '.join(list(map(lambda x: x.strip().capitalize(), generated_text.split('.'))))
            st.write(f" Here's your Generated Text :\n   {generated_text}")


if __name__ == '__main__':
    main()