import os
import re
from typing import Any, Dict, List

from dotenv import load_dotenv
from neo4j import GraphDatabase

from app.core.logger import logger

load_dotenv()

_neo4j_driver = None    
def get_neo4j_driver() -> GraphDatabase:
    """
    获取 Neo4j 驱动实例
    """
    global _neo4j_driver
    if _neo4j_driver is None:
        uri = os.getenv("NEO4J_URI")
        username = os.getenv("NEO4J_USERNAME")
        password = os.getenv("NEO4J_PASSWORD")
        if not uri or not username or not password:
            raise ValueError("请在.env中配置 NEO4J_URI、NEO4J_USERNAME、NEO4J_PASSWORD")
        _neo4j_driver = GraphDatabase.driver(uri, auth=(username, password))
    return _neo4j_driver


ALLOWED_RELATION_TYPES = {
    "HAS_PART",
    "HAS_SPEC",
    "HAS_OPERATION",
    "HAS_WARNING",
    "HAS_RISK",
    "HAS_FAULT",
    "POSSIBLE_CAUSE",
    "SOLUTION",
    "REQUIRES",
    "RELATED_TO",
}


def is_neo4j_configured() -> bool:
    """
    判断是否已配置 Neo4j。未配置时导入/查询节点会跳过 KG，不影响原 RAG 主链路。
    """
    return all([
        os.getenv("NEO4J_URI"),
        os.getenv("NEO4J_USERNAME"),
        os.getenv("NEO4J_PASSWORD"),
    ])


def normalize_relation_type(relation_type: str) -> str:
    """
    将模型输出的关系类型收敛到白名单，避免动态 Cypher 注入和关系类型发散。
    """
    value = (relation_type or "").strip().upper()
    value = re.sub(r"[^A-Z0-9_]", "_", value)
    if value not in ALLOWED_RELATION_TYPES:
        return "RELATED_TO"
    return value


def init_kg_schema() -> None:
    """
    初始化 KG 需要的唯一约束。Neo4j 5 支持 IF NOT EXISTS，重复执行安全。
    """
    driver = get_neo4j_driver()
    statements = [
        "CREATE CONSTRAINT kg_item_name IF NOT EXISTS FOR (n:KGItem) REQUIRE n.name IS UNIQUE",
        "CREATE CONSTRAINT kg_entity_key IF NOT EXISTS FOR (n:KGEntity) REQUIRE n.key IS UNIQUE",
        "CREATE CONSTRAINT kg_chunk_id IF NOT EXISTS FOR (n:KGChunk) REQUIRE n.chunk_id IS UNIQUE",
    ]
    with driver.session() as session:
        for statement in statements:
            session.run(statement)


def delete_item_graph(item_name: str) -> None:
    """
    删除某个 item_name 对应的旧图谱，保证重复导入同一文档时图谱幂等更新。
    """
    if not item_name:
        return
    driver = get_neo4j_driver()
    with driver.session() as session:
        session.run(
            """
            MATCH (n)
            WHERE n.item_name = $item_name OR (n:KGItem AND n.name = $item_name)
            DETACH DELETE n
            """,
            item_name=item_name,
        )


def upsert_kg_triples(item_name: str, triples: List[Dict[str, Any]]) -> int:
    """
    批量写入图谱三元组。每条关系会保留证据 chunk、标题、原文证据和置信度。
    """
    if not triples:
        return 0

    init_kg_schema()
    driver = get_neo4j_driver()
    written_count = 0

    with driver.session() as session:
        for triple in triples:
            subject = (triple.get("subject") or item_name or "").strip()
            predicate = normalize_relation_type(triple.get("predicate", "RELATED_TO"))
            obj = (triple.get("object") or "").strip()
            if not subject or not obj:
                continue

            params = {
                "item_name": item_name,
                "subject": subject,
                "subject_key": f"{item_name}::{subject}",
                "subject_type": triple.get("subject_type") or "概念",
                "object": obj,
                "object_key": f"{item_name}::{obj}",
                "object_type": triple.get("object_type") or "概念",
                "chunk_id": str(triple.get("chunk_id") or ""),
                "file_title": triple.get("file_title") or "",
                "title": triple.get("title") or "",
                "evidence": triple.get("evidence") or "",
                "confidence": float(triple.get("confidence") or 0.0),
            }
            session.run(
                f"""
                MERGE (item:KGItem {{name: $item_name}})
                SET item.item_name = $item_name
                MERGE (s:KGEntity {{key: $subject_key}})
                SET s.name = $subject,
                    s.type = $subject_type,
                    s.item_name = $item_name
                MERGE (o:KGEntity {{key: $object_key}})
                SET o.name = $object,
                    o.type = $object_type,
                    o.item_name = $item_name
                MERGE (c:KGChunk {{chunk_id: $chunk_id}})
                SET c.item_name = $item_name,
                    c.file_title = $file_title,
                    c.title = $title
                MERGE (item)-[:HAS_ENTITY]->(s)
                MERGE (s)-[r:{predicate}]->(o)
                SET r.item_name = $item_name,
                    r.chunk_id = $chunk_id,
                    r.file_title = $file_title,
                    r.title = $title,
                    r.evidence = $evidence,
                    r.confidence = $confidence
                MERGE (s)-[:EVIDENCED_BY]->(c)
                MERGE (o)-[:EVIDENCED_BY]->(c)
                """,
                **params,
            )
            written_count += 1

    logger.info(f"Neo4j 图谱写入完成，item_name={item_name}，数量={written_count}")
    return written_count


def search_kg_facts(item_names: List[str], query: str, limit: int = 8) -> List[Dict[str, Any]]:
    """
    按 item_name 召回图谱事实，并用问题关键词做轻量本地排序。
    返回值已经转换成 query/rerank 可直接使用的文档格式。
    """
    if not item_names:
        return []

    driver = get_neo4j_driver()
    facts: List[Dict[str, Any]] = []
    with driver.session() as session:
        result = session.run(
            """
            MATCH (s:KGEntity)-[r]->(o:KGEntity)
            WHERE r.item_name IN $item_names
              AND type(r) <> 'EVIDENCED_BY'
            RETURN s.name AS subject,
                   s.type AS subject_type,
                   type(r) AS predicate,
                   o.name AS object,
                   o.type AS object_type,
                   r.chunk_id AS chunk_id,
                   r.file_title AS file_title,
                   r.title AS title,
                   r.evidence AS evidence,
                   r.confidence AS confidence
            LIMIT 100
            """,
            item_names=item_names,
        )
        for record in result:
            facts.append(dict(record))

    query_text = (query or "").lower()
    relation_keywords = {
        "HAS_WARNING": ["安全", "注意", "警告", "危险", "防护", "禁忌"],
        "HAS_RISK": ["风险", "危险", "烫伤", "损坏", "危害"],
        "HAS_FAULT": ["故障", "问题", "异常", "无法", "不能", "失效"],
        "POSSIBLE_CAUSE": ["原因", "为什么", "可能", "排查", "导致"],
        "SOLUTION": ["解决", "处理", "怎么办", "排除", "修复"],
        "HAS_OPERATION": ["操作", "步骤", "使用", "设置", "开启"],
        "HAS_SPEC": ["参数", "规格", "温度", "电压", "尺寸", "参数"],
        "HAS_PART": ["部件", "组成", "结构", "零件", "组件"],
        "REQUIRES": ["需要", "必须", "前提", "条件"],
    }

    query_tokens = [t.strip() for t in query_text if t.strip()]

    def score_fact(fact: Dict[str, Any]) -> float:
        haystack = " ".join([
            str(fact.get("subject", "")),
            str(fact.get("predicate", "")),
            str(fact.get("object", "")),
            str(fact.get("evidence", "")),
            str(fact.get("title", "")),
        ]).lower()
        score = float(fact.get("confidence") or 0.0)

        matched_chars = 0
        for char in query_tokens:
            if char in haystack:
                matched_chars += 1
        if query_tokens:
            score += (matched_chars / len(query_tokens)) * 0.3

        predicate = fact.get("predicate", "")
        for keyword in relation_keywords.get(predicate, []):
            if keyword in query_text:
                score += 0.4

        subject = str(fact.get("subject", "")).lower()
        obj = str(fact.get("object", "")).lower()
        for token in query_tokens:
            if token in subject or token in obj:
                score += 0.1

        return score

    facts.sort(key=score_fact, reverse=True)
    docs = []
    for fact in facts[:limit]:
        chunk_id = fact.get("chunk_id") or ""
        predicate = fact.get("predicate") or "RELATED_TO"
        subject = fact.get("subject") or ""
        obj = fact.get("object") or ""
        evidence = fact.get("evidence") or ""
        docs.append({
            "chunk_id": f"kg:{chunk_id}:{predicate}:{subject}:{obj}",
            "text": f"知识图谱事实：{subject} - {predicate} - {obj}。证据：{evidence}",
            "title": fact.get("title") or fact.get("file_title") or "知识图谱事实",
            "source": "kg",
            "url": "",
            "kg": fact,
        })
    return docs
