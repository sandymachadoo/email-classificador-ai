from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from transformers import pipeline
import os

app = FastAPI()

# ⚠️ MODELO CARREGADO UMA ÚNICA VEZ (ESSENCIAL)
classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=-1
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

def classificar_email(texto: str) -> str:
    texto_lower = texto.lower()

    palavras_produtivas = [
        "erro", "problema", "falha", "prazo", "chamado",
        "suporte", "status", "bug", "não funciona",
        "instabilidade", "ajuda", "urgente"
    ]

    for palavra in palavras_produtivas:
        if palavra in texto_lower:
            return "Produtivo"

    if len(texto.split()) < 5:
        return "Improdutivo"

    resultado = classifier(texto, truncation=True)[0]["label"]

    if resultado == "NEGATIVE":
        return "Produtivo"

    return "Improdutivo"

def gerar_resposta(categoria: str) -> str:
    if categoria == "Produtivo":
        return (
            "Olá!\n\n"
            "Recebemos sua mensagem e ela já foi encaminhada para o time responsável. "
            "Nossa equipe está analisando a situação e retornará em breve.\n\n"
            "Agradecemos o contato."
        )
    else:
        return (
            "Olá!\n\n"
            "Agradecemos sua mensagem!\n"
            "Ficamos felizes pelo seu contato e esperamos que você tenha um ótimo dia.\n\n"
            "Atenciosamente,\n"
            "Equipe AutoU"
        )

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "resultado": ""}
    )

@app.post("/analisar", response_class=HTMLResponse)
def analisar_email(request: Request, email_texto: str = Form(...)):
    categoria = classificar_email(email_texto)
    resposta = gerar_resposta(categoria)

    resultado = f"Categoria: {categoria}\n\nResposta automática:\n{resposta}"

    return templates.TemplateResponse(
        "index.html",
        {"request": request, "resultado": resultado}
    )
