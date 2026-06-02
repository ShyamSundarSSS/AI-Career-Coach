from flask import Flask, request, render_template, redirect, url_for
from dotenv import load_dotenv
import os
from werkzeug.utils import secure_filename
import PyPDF2

from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_text_splitters import CharacterTextSplitter

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

# =====================================================
# Flask App
# =====================================================

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


# =====================================================
# OpenAI Model
# =====================================================

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)


# =====================================================
# Embeddings + Text Splitter
# =====================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=2000,
    chunk_overlap=200,
    length_function=len,
)


# =====================================================
# Resume Summary Prompt
# =====================================================

resume_summary_template = """
Role: You are an AI Career Coach.

Task:
Given the candidate's resume, provide a comprehensive summary that includes:

- Career Objective
- Skills and Expertise
- Professional Experience
- Educational Background
- Notable Achievements

Instructions:
Provide a concise and professional summary of the resume.
Highlight strengths, technical skills, experience,
and career progression.

Resume:
{resume}
"""

resume_prompt = PromptTemplate(
    input_variables=["resume"],
    template=resume_summary_template,
)

# LCEL Chain (replacement for LLMChain)
resume_analysis_chain = resume_prompt | llm


# =====================================================
# PDF Text Extraction
# =====================================================

def extract_text_from_pdf(pdf_path):
    text = ""

    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)

        for page in reader.pages:
            extracted = page.extract_text()

            if extracted:
                text += extracted

    return text


# =====================================================
# Question Answering
# =====================================================

def perform_qa(query):

    db = FAISS.load_local(
        "vector_index",
        embeddings,
        allow_dangerous_deserialization=True,
    )

    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )

    qa_prompt = PromptTemplate.from_template(
        """
You are an AI Career Coach.

Answer the question only using the resume context.

Context:
{context}

Question:
{input}
"""
    )

    document_chain = create_stuff_documents_chain(
        llm,
        qa_prompt,
    )

    retrieval_chain = create_retrieval_chain(
        retriever,
        document_chain,
    )

    result = retrieval_chain.invoke(
        {"input": query}
    )

    return result["answer"]


# =====================================================
# Routes
# =====================================================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():

    if "file" not in request.files:
        return redirect(url_for("index"))

    file = request.files["file"]

    if file.filename == "":
        return redirect(url_for("index"))

    filename = secure_filename(file.filename)

    file_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename,
    )

    file.save(file_path)

    # Extract resume text
    resume_text = extract_text_from_pdf(file_path)

    # Split text
    splitted_text = text_splitter.split_text(
        resume_text
    )

    # Create vector store
    vectorstore = FAISS.from_texts(
        splitted_text,
        embeddings,
    )

    vectorstore.save_local("vector_index")

    # Resume analysis
    response = resume_analysis_chain.invoke(
        {"resume": resume_text}
    )

    resume_analysis = response.content

    return render_template(
        "results.html",
        resume_analysis=resume_analysis,
    )


@app.route("/ask", methods=["GET", "POST"])
def ask_query():

    if request.method == "POST":

        query = request.form["query"]

        result = perform_qa(query)

        return render_template(
            "qa_results.html",
            query=query,
            result=result,
        )

    return render_template("ask.html")


# =====================================================
# Run App
# =====================================================

if __name__ == "__main__":
    app.run(debug=True)
