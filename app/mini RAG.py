import openai
import numpy as np
import os
import requests
from typing import List, Dict
import json


class YandexRAG:
    """
    RAG система с использованием Yandex Foundation Models
    через OpenAI Compatible API
    """

    def __init__(self):
        """Инициализация RAG системы"""
        self.api_key = "your_yandex_api_key"
        self.folder_id = "your_yandex_folder_id"
        self.base_url = "https://llm.api.cloud.yandex.net/foundationModels/v1/"

        if not self.api_key:
            raise ValueError("Установите переменную окружения YANDEX_CLOUD_API_KEY")

        # Настройка OpenAI клиента для Yandex
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

        self.embedding_model = f"emb://{self.folder_id}/text-search-doc/latest"
        self.generation_model = f"gpt://{self.folder_id}/yandexgpt/latest"
        self.documents = []

        print(f"✅ YandexRAG инициализирован с моделью: yandexgpt")

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Получение эмбеддингов через Yandex API

        Args:
            texts: Список текстов для векторизации

        Returns:
            Список векторов эмбеддингов
        """
        embeddings = []
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        for text in texts:
            payload = {
                "modelUri": self.embedding_model,
                "text": text
            }

            try:
                response = requests.post(
                    f"{self.base_url}textEmbedding",
                    headers=headers,
                    json=payload,
                    timeout=30
                )
                response.raise_for_status()

                embedding = response.json()["embedding"]
                embeddings.append(embedding)

            except requests.exceptions.RequestException as e:
                print(f"❌ Ошибка получения эмбеддинга: {e}")
                # Возвращаем случайный вектор как fallback
                embeddings.append(np.random.rand(256).tolist())

        return embeddings

    def split_documents(self, docs: List[str], chunk_size: int = 1000) -> List[str]:
        """
        Разбиение документов на чанки

        Args:
            docs: Список документов
            chunk_size: Размер чанка в словах

        Returns:
            Список чанков
        """
        chunks = []
        for doc in docs:
            words = doc.split()
            for i in range(0, len(words), chunk_size):
                chunk = " ".join(words[i:i + chunk_size])
                if chunk.strip():  # Избегаем пустых чанков
                    chunks.append(chunk)

        print(f"📄 Создано {len(chunks)} чанков из {len(docs)} документов")
        return chunks

    def add_documents(self, docs: List[str]) -> None:
        """
        Добавление документов в векторную базу

        Args:
            docs: Список текстов документов
        """
        print(f"🔄 Обработка {len(docs)} документов...")

        # Разбиение на чанки
        chunks = self.split_documents(docs)

        # Получение эмбеддингов
        embeddings = self.get_embeddings(chunks)

        # Сохранение в векторную БД
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            self.documents.append({
                "id": i,
                "text": chunk,
                "embedding": np.array(embedding)
            })

        print(f"✅ Добавлено {len(chunks)} чанков в векторную базу")

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Семантический поиск по запросу

        Args:
            query: Поисковый запрос
            top_k: Количество результатов

        Returns:
            Список наиболее релевантных документов
        """
        if not self.documents:
            print("⚠️ Векторная база пустая!")
            return []

        print(f"🔍 Поиск по запросу: '{query}'")

        # Получение эмбеддинга запроса
        query_embeddings = self.get_embeddings([query])
        query_vector = np.array(query_embeddings[0])

        # Вычисление сходства
        similarities = []
        for doc in self.documents:
            similarity = np.dot(query_vector, doc["embedding"]) / (
                    np.linalg.norm(query_vector) * np.linalg.norm(doc["embedding"])
            )
            similarities.append({
                "document": doc,
                "similarity": float(similarity)
            })

        # Сортировка по убыванию сходства
        similarities.sort(key=lambda x: x["similarity"], reverse=True)

        results = similarities[:top_k]
        print(f"📊 Найдено {len(results)} релевантных документов")

        return results

    def generate_answer(self, query: str, context_docs: List[Dict]) -> str:
        """
        Генерация ответа на основе найденного контекста

        Args:
            query: Вопрос пользователя
            context_docs: Найденные документы

        Returns:
            Сгенерированный ответ
        """
        # Формирование контекста
        context_texts = []
        for i, doc_info in enumerate(context_docs, 1):
            doc = doc_info["document"]
            similarity = doc_info["similarity"]
            context_texts.append(f"Документ {i} (релевантность: {similarity:.3f}): {doc['text']}")

            context = """

                      """.join(context_texts)

            # Создание промпта
            messages = [
                {
                    "role": "system",
                    "content": "Ты - помощник по поиску информации. Отвечай на вопросы пользователя, используя только предоставленный контекст. Если в контексте нет ответа, честно скажи об этом."
                },
                {
                    "role": "user",
                    "content": f"""Контекст:
{context}

Вопрос: {query}

Ответь на вопрос, основываясь только на предоставленном контексте:"""
                }
            ]

        try:
            print("🤖 Генерация ответа...")
            response = self.client.chat.completions.create(
                model=self.generation_model,
                messages=messages,
                max_tokens=1500,
                temperature=0.2
            )

            answer = response.choices[0].message.content
            return answer

        except Exception as e:
            error_msg = f"❌ Ошибка генерации ответа: {e}"
            print(error_msg)
            return error_msg

    def ask(self, query: str) -> Dict:
        """
        Полный RAG запрос: поиск + генерация

        Args:
            query: Вопрос пользователя

        Returns:
            Результат с ответом и метаданными
        """
        # Поиск релевантных документов
        search_results = self.search(query)

        if not search_results:
            return {
                "answer": "Извините, не удалось найти релевантные документы.",
                "sources": [],
                "query": query
            }

        # Генерация ответа
        answer = self.generate_answer(query, search_results)

        # Формирование источников
        sources = []
        for result in search_results:
            doc = result["document"]
            sources.append({
                "text": doc["text"][:200] + "..." if len(doc["text"]) > 200 else doc["text"],
                "similarity": result["similarity"],
                "id": doc["id"]
            })

        return {
            "answer": answer,
            "sources": sources,
            "query": query,
            "model": "yandexgpt"
        }


# Пример использования
def demo_yandex_rag():
    """Демонстрация работы YandexRAG"""

    # Инициализация
    rag = YandexRAG()

    # Тестовые документы
    documents = [
        "Yandex Foundation Models - это семейство больших языковых моделей от Яндекса для различных задач обработки естественного языка.",
        "YandexGPT Pro поддерживает контекст до 32000 токенов и обеспечивает высокое качество генерации на русском и английском языках.",
        "RAG (Retrieval-Augmented Generation) позволяет языковым моделям использовать внешние источники знаний для генерации более точных ответов.",
        "Векторные эмбеддинги преобразуют текст в числовые векторы, что позволяет выполнять семантический поиск по документам.",
        "Семантический поиск находит документы не по ключевым словам, а по смыслу и контексту запроса."
    ]

    # Добавление документов
    rag.add_documents(documents)

    # Тестовые запросы
    queries = [
        "Что такое YandexGPT Pro?",
        "Как работает RAG?",
        "Что такое семантический поиск?"
    ]

    print("""
          " + " = """*80)
    print("🎯 ДЕМОНСТРАЦИЯ YANDEX RAG СИСТЕМЫ")
    print("=" * 80)

    for query in queries:
        print(f"❓ Запрос: {query}")
    print("-" * 60)

    result = rag.ask(query)

    print(f"🤖 Ответ: {result['answer']}")
    print(f"📚 Источники({len(result['sources'])}): ")

    for i, source in enumerate(result['sources'], 1):
        print(f"   {i}. Сходство: {source['similarity']:.3f}")
    print(f"      {source['text']}")

    print(" " + " - " * 60)


demo_yandex_rag()
