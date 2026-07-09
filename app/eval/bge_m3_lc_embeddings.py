"""
BGE-M3 LangChain Embeddings 适配器

将项目已有的 BGE-M3 嵌入模型（通过 pymilvus-model 的 BGEM3EmbeddingFunction）
包装为 LangChain 标准的 Embeddings 接口，供 RAGAS 评估框架使用。

RAGAS 的 LangchainEmbeddingsWrapper 需要一个实现了
embed_documents / embed_query 方法的 LangChain Embeddings 对象，
而项目的 generate_embeddings() 返回 {dense, sparse} 字典格式，
此适配器负责格式转换。
"""

from langchain_core.embeddings import Embeddings

from app.lm.embedding_utils import generate_embeddings
from app.core.logger import logger


class BGEM3LangchainEmbeddings(Embeddings):
    """
    BGE-M3 的 LangChain Embeddings 适配器

    仅暴露 BGE-M3 的稠密向量（dense），忽略稀疏向量（sparse），
    因为 RAGAS 评估指标只需要稠密向量做语义相似度计算。
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        批量生成文档的稠密向量（LangChain Embeddings 标准接口）

        :param texts: 文本列表
        :return: 稠密向量列表，每个向量是 1024 维的 float 列表
        """
        if not texts:
            return []
        logger.debug(f"[BGE-M3适配器] 批量生成 {len(texts)} 条文档向量")
        result = generate_embeddings(texts)
        return result["dense"]

    def embed_query(self, text: str) -> list[float]:
        """
        生成单条查询文本的稠密向量（LangChain Embeddings 标准接口）

        :param text: 查询文本
        :return: 1024 维的稠密向量
        """
        logger.debug(f"[BGE-M3适配器] 生成查询向量: {text[:50]}...")
        result = generate_embeddings([text])
        return result["dense"][0]
