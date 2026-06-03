# %%
!pip install rank-bm25

# %%
# =====================================================
# TALENTCLEF TASK A - PIPELINE BGE-M3 + QUERY ENRICHMENT
# Dense Retrieval + BM25 + Top-K alto + Cross-Encoder validado
# =====================================================

import os
import re
import zipfile
import pandas as pd
import numpy as np
import torch

from sentence_transformers import SentenceTransformer, CrossEncoder, util
from ranx import Qrels, Run, evaluate
from rank_bm25 import BM25Okapi


# =====================================================
# 1. CONFIGURACIÓN GENERAL
# =====================================================

base_path = r"c:\Users\moren\Downloads\cursor\TalenCLEF"

dev_en_path = os.path.join(base_path, "development", "en")
dev_es_path = os.path.join(base_path, "development", "es")

test_en_path = os.path.join(base_path, "en")
test_es_path = os.path.join(base_path, "es")

out_dir = os.path.join(base_path, "submission")
os.makedirs(out_dir, exist_ok=True)

zip_path = os.path.join(base_path, "melissamoreno_taskA_test.zip")

RUN_TAG = "VerbaNexAI"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Dispositivo detectado:", device)

if device == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# =====================================================
# 2. PARÁMETROS
# =====================================================

BI_ENCODER_MODEL = "BAAI/bge-m3"

CROSS_ENCODER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

TOP_K_RETRIEVAL = 700
TOP_K_FINAL = 100

# Según tus resultados previos, alpha bajo funcionó mejor
BEST_ALPHA_EN = 0.60
BEST_ALPHA_ES = 0.60
BEST_ALPHA_AVG = 0.60

# El código evalúa CON y SIN cross-encoder en development
USE_CROSS_ENCODER_FOR_TEST = True

LAMBDA_CE = 0.95

BATCH_SIZE_BI_ENCODER = 8
BATCH_SIZE_CROSS_ENCODER = 8


# =====================================================
# 3. CARGA DE DATOS
# =====================================================

def load_text_folder(folder_path):
    data = []

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if os.path.isfile(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read().strip()

            data.append({
                "id": str(os.path.splitext(filename)[0]),
                "text": text
            })

    df = pd.DataFrame(data)
    df["id"] = df["id"].astype(str)

    return df


def load_qrels(path):
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["query_id", "iter", "doc_id", "relevance"]
    )

    df["query_id"] = df["query_id"].astype(str)
    df["doc_id"] = df["doc_id"].astype(str)
    df["relevance"] = pd.to_numeric(df["relevance"], errors="coerce").fillna(0)

    return df


# =====================================================
# 4. LIMPIEZA
# =====================================================

def clean_text(text):
    text = str(text)

    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\+?\d[\d\s\-]{7,}", " ", text)

    text = text.lower()

    # Conserva señales técnicas: c++, c#, .net, node.js
    text = re.sub(r"[^a-záéíóúñü0-9\+\#\. ]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def tokenize_bm25(text):
    return text.split()


# =====================================================
# 5. QUERY ENRICHMENT TIPO LLM / FINE-GRAINED JD
# =====================================================

def enrich_job_description(text, lang="en"):
    """
    Versión ligera inspirada en LLM-based fine-grained job description.
    No llama a una API externa. Reestructura la query para que BGE/BM25
    lean la vacante como requisitos explícitos.
    """

    text = str(text).strip()

    if lang == "es":
        enriched = f"""
        Descripción del cargo:
        {text}

        Requisitos principales del cargo:
        Habilidades esenciales, habilidades técnicas, herramientas, tecnologías, lenguajes, certificaciones.

        Preferencias de experiencia:
        años de experiencia, funciones realizadas, responsabilidades previas, sector o industria relacionada.

        Formación esperada:
        nivel educativo, área profesional, carrera, especialización o conocimiento requerido.

        Preferencias implícitas:
        tipo de perfil buscado, competencias transferibles, experiencia similar, dominio profesional, ajuste entre cargo y hoja de vida.
        """
    else:
        enriched = f"""
        Job description:
        {text}

        Core skill requirements:
        essential skills, technical skills, tools, technologies, programming languages, certifications.

        Experience preferences:
        years of experience, previous responsibilities, role-related tasks, industry background.

        Educational background requirements:
        degree level, academic field, professional training, specialization.

        Implicit preferences:
        candidate profile, transferable skills, related experience, domain expertise, job-resume alignment.
        """

    return clean_text(enriched)


# =====================================================
# 6. PREPARAR DATAFRAMES
# =====================================================

def prepare_dataframes(queries_df, corpus_df, lang="en"):
    queries_df = queries_df.copy()
    corpus_df = corpus_df.copy()

    queries_df["text_clean"] = queries_df["text"].apply(clean_text)
    corpus_df["text_clean"] = corpus_df["text"].apply(clean_text)

    # Query enriquecida
    queries_df["query_enriched"] = queries_df["text"].apply(
        lambda x: enrich_job_description(x, lang=lang)
    )

    # Para BGE-M3 no se usa "query:" / "passage:"
    queries_df["final_dense"] = queries_df["query_enriched"]
    corpus_df["final_dense"] = corpus_df["text_clean"]

    return queries_df, corpus_df


# =====================================================
# 7. NORMALIZACIÓN
# =====================================================

def minmax_normalize(scores):
    scores = np.array(scores, dtype=float)

    min_s = scores.min()
    max_s = scores.max()

    if max_s - min_s == 0:
        return np.zeros_like(scores)

    return (scores - min_s) / (max_s - min_s)


# =====================================================
# 8. CARGAR DEVELOPMENT
# =====================================================

print("\nCargando development...")

corpus_en = load_text_folder(os.path.join(dev_en_path, "corpus"))
queries_en = load_text_folder(os.path.join(dev_en_path, "queries"))
qrels_en = load_qrels(os.path.join(dev_en_path, "qrels.tsv"))

corpus_es = load_text_folder(os.path.join(dev_es_path, "corpus"))
queries_es = load_text_folder(os.path.join(dev_es_path, "queries"))
qrels_es = load_qrels(os.path.join(dev_es_path, "qrels.tsv"))

queries_en, corpus_en = prepare_dataframes(queries_en, corpus_en, lang="en")
queries_es, corpus_es = prepare_dataframes(queries_es, corpus_es, lang="es")

print("EN corpus:", corpus_en.shape)
print("EN queries:", queries_en.shape)
print("ES corpus:", corpus_es.shape)
print("ES queries:", queries_es.shape)


# =====================================================
# 9. CARGAR MODELOS
# =====================================================

print("\nCargando bi-encoder:", BI_ENCODER_MODEL)
bi_encoder = SentenceTransformer(BI_ENCODER_MODEL, device=device)

print("\nCargando cross-encoder:", CROSS_ENCODER_MODEL)
cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL, device=device)

print("Bi-encoder device:", bi_encoder.device)
print("Cross-encoder device:", cross_encoder.model.device)


# =====================================================
# 10. EMBEDDINGS
# =====================================================

def encode_texts(texts, model):
    return model.encode(
        texts.tolist(),
        batch_size=BATCH_SIZE_BI_ENCODER,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )


print("\nEmbeddings development EN...")
corpus_en_emb = encode_texts(corpus_en["final_dense"], bi_encoder)
queries_en_emb = encode_texts(queries_en["final_dense"], bi_encoder)

print("\nEmbeddings development ES...")
corpus_es_emb = encode_texts(corpus_es["final_dense"], bi_encoder)
queries_es_emb = encode_texts(queries_es["final_dense"], bi_encoder)


# =====================================================
# 11. BM25 SOBRE QUERY ENRIQUECIDA
# =====================================================

def build_bm25(corpus_df):
    tokenized_corpus = corpus_df["text_clean"].apply(tokenize_bm25).tolist()
    return BM25Okapi(tokenized_corpus)


print("\nConstruyendo BM25 EN...")
bm25_en = build_bm25(corpus_en)

print("Construyendo BM25 ES...")
bm25_es = build_bm25(corpus_es)


# =====================================================
# 12. HYBRID RETRIEVAL: BGE-M3 + BM25
# =====================================================

def hybrid_retrieve(
    queries_df,
    corpus_df,
    queries_emb,
    corpus_emb,
    bm25,
    alpha=0.60,
    top_k=700
):
    rows = []

    for i, qid in enumerate(queries_df["id"]):

        dense_scores = util.cos_sim(
            queries_emb[i],
            corpus_emb
        )[0].cpu().numpy()

        query_tokens = tokenize_bm25(queries_df.iloc[i]["query_enriched"])
        bm25_scores = bm25.get_scores(query_tokens)

        dense_norm = minmax_normalize(dense_scores)
        bm25_norm = minmax_normalize(bm25_scores)

        hybrid_scores = alpha * dense_norm + (1 - alpha) * bm25_norm

        top_k_real = min(top_k, len(hybrid_scores))
        idxs = hybrid_scores.argsort()[::-1][:top_k_real]

        for rank_position, idx in enumerate(idxs, 1):
            rows.append({
                "query_id": qid,
                "doc_id": corpus_df.iloc[idx]["id"],
                "rank_initial": rank_position,
                "score_initial": float(hybrid_scores[idx]),
                "score_dense": float(dense_norm[idx]),
                "score_bm25": float(bm25_norm[idx]),
                "query_text": queries_df.iloc[i]["query_enriched"],
                "doc_text": corpus_df.iloc[idx]["text_clean"]
            })

    return pd.DataFrame(rows)


# =====================================================
# 13. CROSS-ENCODER RERANKING
# =====================================================

def rerank_with_cross_encoder(
    run_df,
    cross_encoder=None,
    top_k_final=100,
    lambda_ce=0.95
):
    final_df = run_df.copy()

    if cross_encoder is not None:
        pairs = list(zip(
            final_df["query_text"].tolist(),
            final_df["doc_text"].tolist()
        ))

        ce_scores = cross_encoder.predict(
            pairs,
            batch_size=BATCH_SIZE_CROSS_ENCODER,
            show_progress_bar=True
        )

        final_df["score_cross"] = ce_scores

        final_rows = []

        for qid, group in final_df.groupby("query_id"):
            group = group.copy()

            group["score_cross_norm"] = minmax_normalize(
                group["score_cross"].values
            )

            group["score"] = (
                lambda_ce * group["score_cross_norm"]
                + (1 - lambda_ce) * group["score_initial"]
            )

            group = group.sort_values("score", ascending=False).head(top_k_final)
            group["rank"] = range(1, len(group) + 1)

            final_rows.append(group)

        final_df = pd.concat(final_rows, ignore_index=True)

    else:
        final_rows = []

        for qid, group in final_df.groupby("query_id"):
            group = group.copy()

            group["score"] = group["score_initial"]
            group = group.sort_values("score", ascending=False).head(top_k_final)
            group["rank"] = range(1, len(group) + 1)

            final_rows.append(group)

        final_df = pd.concat(final_rows, ignore_index=True)

    return final_df[["query_id", "doc_id", "rank", "score"]]


# =====================================================
# 14. EVALUACIÓN
# =====================================================

def eval_run(qrels_df, run_df):
    qrels_dict = {}

    for _, row in qrels_df.iterrows():
        qrels_dict.setdefault(row["query_id"], {})[row["doc_id"]] = int(row["relevance"])

    run_dict = {}

    for _, row in run_df.iterrows():
        run_dict.setdefault(row["query_id"], {})[row["doc_id"]] = float(row["score"])

    return evaluate(
        Qrels(qrels_dict),
        Run(run_dict),
        ["map", "mrr", "ndcg@10", "precision@5", "precision@10"]
    )


# =====================================================
# 15. VALIDAR CON Y SIN CROSS-ENCODER EN DEVELOPMENT
# =====================================================

print("\n====================================")
print("VALIDACIÓN DEVELOPMENT CON Y SIN CROSS-ENCODER")
print("====================================")

# EN initial
dev_initial_en = hybrid_retrieve(
    queries_en,
    corpus_en,
    queries_en_emb,
    corpus_en_emb,
    bm25_en,
    alpha=BEST_ALPHA_EN,
    top_k=TOP_K_RETRIEVAL
)

# ES initial
dev_initial_es = hybrid_retrieve(
    queries_es,
    corpus_es,
    queries_es_emb,
    corpus_es_emb,
    bm25_es,
    alpha=BEST_ALPHA_ES,
    top_k=TOP_K_RETRIEVAL
)

# Sin cross
dev_run_en_no_cross = rerank_with_cross_encoder(
    dev_initial_en,
    cross_encoder=None,
    top_k_final=TOP_K_FINAL
)

dev_run_es_no_cross = rerank_with_cross_encoder(
    dev_initial_es,
    cross_encoder=None,
    top_k_final=TOP_K_FINAL
)

metrics_en_no_cross = eval_run(qrels_en, dev_run_en_no_cross)
metrics_es_no_cross = eval_run(qrels_es, dev_run_es_no_cross)

avg_no_cross = (
    metrics_en_no_cross["map"] + metrics_es_no_cross["map"]
) / 2

# Con cross
dev_run_en_cross = rerank_with_cross_encoder(
    dev_initial_en,
    cross_encoder=cross_encoder,
    top_k_final=TOP_K_FINAL,
    lambda_ce=LAMBDA_CE
)

dev_run_es_cross = rerank_with_cross_encoder(
    dev_initial_es,
    cross_encoder=cross_encoder,
    top_k_final=TOP_K_FINAL,
    lambda_ce=LAMBDA_CE
)

metrics_en_cross = eval_run(qrels_en, dev_run_en_cross)
metrics_es_cross = eval_run(qrels_es, dev_run_es_cross)

avg_cross = (
    metrics_en_cross["map"] + metrics_es_cross["map"]
) / 2

print("\n--- SIN CROSS-ENCODER ---")
print("MAP EN:", metrics_en_no_cross["map"])
print("MAP ES:", metrics_es_no_cross["map"])
print("AVG MAP:", avg_no_cross)

print("\n--- CON CROSS-ENCODER ---")
print("MAP EN:", metrics_en_cross["map"])
print("MAP ES:", metrics_es_cross["map"])
print("AVG MAP:", avg_cross)

if avg_cross > avg_no_cross:
    print("\nDecisión: usar CROSS-ENCODER en test.")
    USE_CROSS_ENCODER_FOR_TEST = True
else:
    print("\nDecisión: NO usar CROSS-ENCODER en test porque baja MAP.")
    USE_CROSS_ENCODER_FOR_TEST = False


# =====================================================
# 16. FUNCIONES TREC
# =====================================================

def save_trec(run_df, path, run_tag):
    with open(path, "w", encoding="utf-8") as f:
        for _, row in run_df.iterrows():
            f.write(
                f"{row['query_id']} Q0 {row['doc_id']} "
                f"{int(row['rank'])} {float(row['score'])} {run_tag}\n"
            )


def validate_trec_file(path):
    errors = []

    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for i, line in enumerate(lines, 1):
        parts = line.strip().split()

        if len(parts) != 6:
            errors.append((i, "Número incorrecto de columnas", line))
            continue

        qid, q0, doc_id, rank, score, tag = parts

        if q0 != "Q0":
            errors.append((i, "La segunda columna no es Q0", line))

        try:
            int(rank)
        except:
            errors.append((i, "Rank no es entero", line))

        try:
            float(score)
        except:
            errors.append((i, "Score no es numérico", line))

    return errors


# =====================================================
# 17. CARGAR TEST
# =====================================================

print("\nCargando test...")

test_corpus_en = load_text_folder(os.path.join(test_en_path, "corpus"))
test_queries_en = load_text_folder(os.path.join(test_en_path, "queries"))

test_corpus_es = load_text_folder(os.path.join(test_es_path, "corpus"))
test_queries_es = load_text_folder(os.path.join(test_es_path, "queries"))

test_queries_en, test_corpus_en = prepare_dataframes(
    test_queries_en,
    test_corpus_en,
    lang="en"
)

test_queries_es, test_corpus_es = prepare_dataframes(
    test_queries_es,
    test_corpus_es,
    lang="es"
)

print("TEST EN corpus:", test_corpus_en.shape)
print("TEST EN queries:", test_queries_en.shape)
print("TEST ES corpus:", test_corpus_es.shape)
print("TEST ES queries:", test_queries_es.shape)


# =====================================================
# 18. EMBEDDINGS TEST
# =====================================================

print("\nEmbeddings TEST EN...")
test_corpus_en_emb = encode_texts(test_corpus_en["final_dense"], bi_encoder)
test_queries_en_emb = encode_texts(test_queries_en["final_dense"], bi_encoder)

print("\nEmbeddings TEST ES...")
test_corpus_es_emb = encode_texts(test_corpus_es["final_dense"], bi_encoder)
test_queries_es_emb = encode_texts(test_queries_es["final_dense"], bi_encoder)


# =====================================================
# 19. BM25 TEST
# =====================================================

print("\nConstruyendo BM25 TEST EN...")
bm25_test_en = build_bm25(test_corpus_en)

print("Construyendo BM25 TEST ES...")
bm25_test_es = build_bm25(test_corpus_es)


# =====================================================
# 20. RANKING TEST EN-EN
# =====================================================

print("\nRanking TEST en-en...")

initial_en_en = hybrid_retrieve(
    test_queries_en,
    test_corpus_en,
    test_queries_en_emb,
    test_corpus_en_emb,
    bm25_test_en,
    alpha=BEST_ALPHA_EN,
    top_k=TOP_K_RETRIEVAL
)

run_en_en = rerank_with_cross_encoder(
    initial_en_en,
    cross_encoder=cross_encoder if USE_CROSS_ENCODER_FOR_TEST else None,
    top_k_final=TOP_K_FINAL,
    lambda_ce=LAMBDA_CE
)


# =====================================================
# 21. RANKING TEST ES-ES
# =====================================================

print("\nRanking TEST es-es...")

initial_es_es = hybrid_retrieve(
    test_queries_es,
    test_corpus_es,
    test_queries_es_emb,
    test_corpus_es_emb,
    bm25_test_es,
    alpha=BEST_ALPHA_ES,
    top_k=TOP_K_RETRIEVAL
)

run_es_es = rerank_with_cross_encoder(
    initial_es_es,
    cross_encoder=cross_encoder if USE_CROSS_ENCODER_FOR_TEST else None,
    top_k_final=TOP_K_FINAL,
    lambda_ce=LAMBDA_CE
)


# =====================================================
# 22. RANKING TEST EN-ES
# =====================================================

print("\nRanking TEST en-es...")

initial_en_es = hybrid_retrieve(
    test_queries_en,
    test_corpus_es,
    test_queries_en_emb,
    test_corpus_es_emb,
    bm25_test_es,
    alpha=BEST_ALPHA_AVG,
    top_k=TOP_K_RETRIEVAL
)

run_en_es = rerank_with_cross_encoder(
    initial_en_es,
    cross_encoder=cross_encoder if USE_CROSS_ENCODER_FOR_TEST else None,
    top_k_final=TOP_K_FINAL,
    lambda_ce=LAMBDA_CE
)


# =====================================================
# 23. GUARDAR ARCHIVOS TREC
# =====================================================

f_en_en = os.path.join(out_dir, "run_en-en.trec")
f_es_es = os.path.join(out_dir, "run_es-es.trec")
f_en_es = os.path.join(out_dir, "run_en-es.trec")

save_trec(run_en_en, f_en_en, RUN_TAG)
save_trec(run_es_es, f_es_es, RUN_TAG)
save_trec(run_en_es, f_en_es, RUN_TAG)

print("\nArchivos TREC guardados:")
print(f_en_en)
print(f_es_es)
print(f_en_es)


# =====================================================
# 24. VALIDAR FORMATO TREC
# =====================================================

for file_path in [f_en_en, f_es_es, f_en_es]:
    errors = validate_trec_file(file_path)

    if len(errors) == 0:
        print("Formato correcto:", os.path.basename(file_path))
    else:
        print("Errores en:", os.path.basename(file_path))
        print(errors[:10])


# =====================================================
# 25. CREAR ZIP FINAL
# =====================================================

if os.path.exists(zip_path):
    os.remove(zip_path)

with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    z.write(f_en_en, os.path.basename(f_en_en))
    z.write(f_es_es, os.path.basename(f_es_es))
    z.write(f_en_es, os.path.basename(f_en_es))

print("\n====================================")
print("ZIP FINAL CREADO")
print("====================================")
print(zip_path)

with zipfile.ZipFile(zip_path, "r") as z:
    print("\nContenido del ZIP:")
    print(z.namelist())


# =====================================================
# 26. RESUMEN FINAL
# =====================================================

print("\n====================================")
print("RESUMEN FINAL")
print("====================================")
print("Bi-encoder:", BI_ENCODER_MODEL)
print("Cross-encoder:", CROSS_ENCODER_MODEL)
print("Usar cross-encoder en test:", USE_CROSS_ENCODER_FOR_TEST)
print("Alpha EN:", BEST_ALPHA_EN)
print("Alpha ES:", BEST_ALPHA_ES)
print("Alpha EN-ES:", BEST_ALPHA_AVG)
print("Top-K retrieval:", TOP_K_RETRIEVAL)
print("Top-K final:", TOP_K_FINAL)
print("MAP DEV sin cross:", avg_no_cross)
print("MAP DEV con cross:", avg_cross)
print("ZIP:", zip_path)


