"""
Local Neo4j-backed replacement for Zep Cloud client.
Mirrors the Zep SDK interface so existing services work with minimal changes.
"""

import json
import uuid as uuid_lib
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional

from neo4j import GraphDatabase

from ..config import Config
from ..utils.logger import get_logger
from .voyage_embedding import VoyageEmbedding
from .llm_entity_extractor import LLMEntityExtractor
from ..utils.llm_client import LLMClient

logger = get_logger(__name__)


# ============== Data Models (match Zep SDK attribute names) ==============

@dataclass
class NodeObject:
    uuid_: str
    name: str
    labels: List[str] = field(default_factory=list)
    summary: str = ""
    attributes: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""

    @property
    def uuid(self):
        return self.uuid_


@dataclass
class EdgeObject:
    uuid_: str
    name: str
    fact: str = ""
    fact_type: str = ""
    source_node_uuid: str = ""
    target_node_uuid: str = ""
    source_node_name: str = ""
    target_node_name: str = ""
    attributes: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    valid_at: str = ""
    invalid_at: str = ""
    expired_at: str = ""
    episodes: List[str] = field(default_factory=list)
    episode_ids: List[str] = field(default_factory=list)

    @property
    def uuid(self):
        return self.uuid_


@dataclass
class EpisodeObject:
    uuid_: str
    processed: bool = False

    @property
    def uuid(self):
        return self.uuid_


@dataclass
class SearchResponse:
    edges: List[EdgeObject] = field(default_factory=list)
    nodes: List[NodeObject] = field(default_factory=list)


# ============== Helper Functions ==============

def _neo4j_node_to_object(node) -> NodeObject:
    attrs_raw = node.get("attributes", "{}")
    attrs = json.loads(attrs_raw) if isinstance(attrs_raw, str) else attrs_raw
    labels_raw = node.get("labels", "[]")
    labels = json.loads(labels_raw) if isinstance(labels_raw, str) else labels_raw
    created = node.get("created_at", "")
    if hasattr(created, "isoformat"):
        created = created.isoformat()
    return NodeObject(
        uuid_=node["uuid"],
        name=node.get("name", ""),
        labels=labels,
        summary=node.get("summary", ""),
        attributes=attrs,
        created_at=str(created),
    )


def _neo4j_fact_to_object(fact) -> EdgeObject:
    attrs_raw = fact.get("attributes", "{}")
    attrs = json.loads(attrs_raw) if isinstance(attrs_raw, str) else attrs_raw
    episodes_raw = fact.get("episodes", "[]")
    episodes = json.loads(episodes_raw) if isinstance(episodes_raw, str) else episodes_raw

    def _ts(val):
        if val is None:
            return ""
        if hasattr(val, "isoformat"):
            return val.isoformat()
        return str(val)

    return EdgeObject(
        uuid_=fact["uuid"],
        name=fact.get("name", ""),
        fact=fact.get("fact", ""),
        fact_type=fact.get("fact_type", ""),
        source_node_uuid=fact.get("source_node_uuid", ""),
        target_node_uuid=fact.get("target_node_uuid", ""),
        source_node_name=fact.get("source_node_name", ""),
        target_node_name=fact.get("target_node_name", ""),
        attributes=attrs,
        created_at=_ts(fact.get("created_at")),
        valid_at=_ts(fact.get("valid_at")),
        invalid_at=_ts(fact.get("invalid_at")),
        expired_at=_ts(fact.get("expired_at")),
        episodes=episodes,
        episode_ids=episodes,
    )


def _reciprocal_rank_fusion(vector_ranked, bm25_ranked, k=60):
    """Merge two ranked lists using RRF."""
    scores = {}
    for rank, uuid in enumerate(vector_ranked):
        scores[uuid] = scores.get(uuid, 0) + 1.0 / (k + rank + 1)
    for rank, uuid in enumerate(bm25_ranked):
        scores[uuid] = scores.get(uuid, 0) + 1.0 / (k + rank + 1)
    return sorted(scores.keys(), key=lambda u: scores[u], reverse=True)


# ============== Namespace Classes ==============

class EpisodeNamespace:
    def __init__(self, driver):
        self._driver = driver

    def get(self, uuid_: str) -> EpisodeObject:
        with self._driver.session() as session:
            result = session.run(
                "MATCH (ep:Episode {uuid: $uuid}) RETURN ep.processed AS processed",
                uuid=uuid_,
            )
            record = result.single()
            if record:
                return EpisodeObject(uuid_=uuid_, processed=record["processed"])
            return EpisodeObject(uuid_=uuid_, processed=False)


class NodeNamespace:
    def __init__(self, driver):
        self._driver = driver

    def get(self, uuid_: str) -> NodeObject:
        with self._driver.session() as session:
            result = session.run(
                "MATCH (n:Entity {uuid: $uuid}) RETURN n",
                uuid=uuid_,
            )
            record = result.single()
            if not record:
                raise ValueError(f"Node {uuid_} not found")
            node = record["n"]
            return _neo4j_node_to_object(node)

    def get_by_graph_id(self, graph_id: str, limit: int = 100, uuid_cursor: str = None) -> List[NodeObject]:
        with self._driver.session() as session:
            if uuid_cursor:
                result = session.run(
                    """MATCH (n:Entity {graph_id: $graph_id})
                    WHERE n.uuid > $cursor
                    RETURN n ORDER BY n.uuid LIMIT $limit""",
                    graph_id=graph_id, cursor=uuid_cursor, limit=limit,
                )
            else:
                result = session.run(
                    """MATCH (n:Entity {graph_id: $graph_id})
                    RETURN n ORDER BY n.uuid LIMIT $limit""",
                    graph_id=graph_id, limit=limit,
                )
            return [_neo4j_node_to_object(r["n"]) for r in result]

    def get_entity_edges(self, node_uuid: str) -> List[EdgeObject]:
        with self._driver.session() as session:
            result = session.run(
                """MATCH (f:Fact)
                WHERE f.source_node_uuid = $uuid OR f.target_node_uuid = $uuid
                RETURN f""",
                uuid=node_uuid,
            )
            return [_neo4j_fact_to_object(r["f"]) for r in result]


class EdgeNamespace:
    def __init__(self, driver):
        self._driver = driver

    def get_by_graph_id(self, graph_id: str, limit: int = 100, uuid_cursor: str = None) -> List[EdgeObject]:
        with self._driver.session() as session:
            if uuid_cursor:
                result = session.run(
                    """MATCH (f:Fact {graph_id: $graph_id})
                    WHERE f.uuid > $cursor
                    RETURN f ORDER BY f.uuid LIMIT $limit""",
                    graph_id=graph_id, cursor=uuid_cursor, limit=limit,
                )
            else:
                result = session.run(
                    """MATCH (f:Fact {graph_id: $graph_id})
                    RETURN f ORDER BY f.uuid LIMIT $limit""",
                    graph_id=graph_id, limit=limit,
                )
            return [_neo4j_fact_to_object(r["f"]) for r in result]


# ============== GraphNamespace ==============

class GraphNamespace:
    def __init__(self, driver, voyage: VoyageEmbedding, extractor: LLMEntityExtractor):
        self._driver = driver
        self._voyage = voyage
        self._extractor = extractor
        self.node = NodeNamespace(driver)
        self.edge = EdgeNamespace(driver)
        self.episode = EpisodeNamespace(driver)

    def create(self, graph_id: str, name: str, description: str = ""):
        with self._driver.session() as session:
            session.run(
                """CREATE (g:Graph {
                    graph_id: $graph_id, name: $name, description: $description,
                    created_at: datetime(), ontology: '{}'
                })""",
                graph_id=graph_id, name=name, description=description,
            )
            for stmt in [
                "CREATE INDEX entity_uuid IF NOT EXISTS FOR (n:Entity) ON (n.uuid)",
                "CREATE INDEX entity_graph IF NOT EXISTS FOR (n:Entity) ON (n.graph_id)",
                "CREATE INDEX fact_uuid IF NOT EXISTS FOR (f:Fact) ON (f.uuid)",
                "CREATE INDEX fact_graph IF NOT EXISTS FOR (f:Fact) ON (f.graph_id)",
                "CREATE INDEX episode_uuid IF NOT EXISTS FOR (ep:Episode) ON (ep.uuid)",
                "CREATE FULLTEXT INDEX entity_fulltext IF NOT EXISTS FOR (n:Entity) ON EACH [n.name, n.summary]",
                "CREATE FULLTEXT INDEX fact_fulltext IF NOT EXISTS FOR (e:Fact) ON EACH [e.name, e.fact]",
            ]:
                try:
                    session.run(stmt)
                except Exception:
                    pass
            for stmt in [
                """CREATE VECTOR INDEX entity_embedding IF NOT EXISTS
                   FOR (n:Entity) ON (n.embedding)
                   OPTIONS {indexConfig: {`vector.dimensions`: 1024, `vector.similarity_function`: 'cosine'}}""",
                """CREATE VECTOR INDEX fact_embedding IF NOT EXISTS
                   FOR (e:Fact) ON (e.embedding)
                   OPTIONS {indexConfig: {`vector.dimensions`: 1024, `vector.similarity_function`: 'cosine'}}""",
            ]:
                try:
                    session.run(stmt)
                except Exception:
                    pass
        logger.info(f"Created graph {graph_id}")


    def delete(self, graph_id: str):
        with self._driver.session() as session:
            session.run("MATCH (n {graph_id: $gid}) DETACH DELETE n", gid=graph_id)
            session.run("MATCH (g:Graph {graph_id: $gid}) DELETE g", gid=graph_id)
        logger.info(f"Deleted graph {graph_id}")

    def set_ontology(self, graph_ids: List[str], entities=None, edges=None):
        ontology_data = {"entities": {}, "edges": {}}
        if entities:
            for name, val in entities.items():
                ontology_data["entities"][name] = val if isinstance(val, dict) else {"name": name}
        if edges:
            for name, val in edges.items():
                ontology_data["edges"][name] = val if isinstance(val, dict) else {"name": name}
        ontology_json = json.dumps(ontology_data, ensure_ascii=False)
        with self._driver.session() as session:
            for gid in graph_ids:
                session.run(
                    "MATCH (g:Graph {graph_id: $gid}) SET g.ontology = $ont",
                    gid=gid, ont=ontology_json,
                )
        logger.info(f"Set ontology for graphs {graph_ids}")

    def _get_ontology(self, graph_id: str) -> Dict[str, Any]:
        with self._driver.session() as session:
            result = session.run(
                "MATCH (g:Graph {graph_id: $gid}) RETURN g.ontology AS ont",
                gid=graph_id,
            )
            record = result.single()
            if record and record["ont"]:
                return json.loads(record["ont"])
        return {}

    def _ontology_to_llm_format(self, ontology_raw: Dict) -> Dict:
        entity_types = []
        for name, val in ontology_raw.get("entities", {}).items():
            et = {"name": name}
            if isinstance(val, dict):
                et.update(val)
            entity_types.append(et)
        edge_types = []
        for name, val in ontology_raw.get("edges", {}).items():
            et = {"name": name}
            if isinstance(val, dict):
                et.update(val)
            edge_types.append(et)
        return {"entity_types": entity_types, "edge_types": edge_types}


    def add_batch(self, graph_id: str, episodes) -> list:
        """Store episodes immediately, process extraction in background thread."""
        ontology_raw = self._get_ontology(graph_id)
        ontology_for_llm = self._ontology_to_llm_format(ontology_raw)
        episode_objects = []
        for ep in episodes:
            ep_uuid = str(uuid_lib.uuid4())
            text = ep.data if hasattr(ep, "data") else ep.get("data", "")
            with self._driver.session() as session:
                session.run(
                    """CREATE (ep:Episode {
                        uuid: $uuid, graph_id: $gid, data: $data,
                        type: 'text', processed: false, created_at: datetime()
                    })""",
                    uuid=ep_uuid, gid=graph_id, data=text,
                )
            # Launch background processing
            thr = threading.Thread(
                target=self._process_episode_async,
                args=(graph_id, ep_uuid, text, ontology_for_llm),
                daemon=True,
            )
            thr.start()
            episode_objects.append(EpisodeObject(uuid_=ep_uuid, processed=False))
        return episode_objects

    def _process_episode_async(self, graph_id: str, ep_uuid: str, text: str, ontology: Dict):
        """Background: extract entities/relations via LLM, embed, store in Neo4j."""
        try:
            entities, relations = self._extractor.extract(text, ontology)
            self._store_entities(graph_id, entities, ep_uuid)
            self._store_relations(graph_id, relations, ep_uuid)
            with self._driver.session() as session:
                session.run(
                    "MATCH (ep:Episode {uuid: $uuid}) SET ep.processed = true",
                    uuid=ep_uuid,
                )
            logger.info(f"Episode {ep_uuid[:8]} processed: {len(entities)} entities, {len(relations)} relations")
        except Exception as e:
            logger.error(f"Episode {ep_uuid[:8]} processing failed: {e}")
            with self._driver.session() as session:
                session.run(
                    "MATCH (ep:Episode {uuid: $uuid}) SET ep.processed = true",
                    uuid=ep_uuid,
                )

    def add(self, graph_id: str, type: str = "text", data: str = ""):
        self.add_batch(graph_id, [{"data": data}])


    def _store_entities(self, graph_id: str, entities: List[Dict], episode_uuid: str):
        if not entities:
            return
        texts = [f"{e.get('name', '')}: {e.get('summary', '')}" for e in entities]
        try:
            embeddings = self._voyage.embed(texts)
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            embeddings = [[] for _ in entities]

        with self._driver.session() as session:
            for ent, emb in zip(entities, embeddings):
                name = ent.get("name", "Unknown")
                etype = ent.get("type", "Entity")
                summary = ent.get("summary", "")
                attrs = ent.get("attributes", {})

                match_uuid = self._find_duplicate_entity(session, graph_id, name, emb)

                if match_uuid:
                    session.run(
                        """MATCH (n:Entity {uuid: $uuid})
                        SET n.summary = CASE
                            WHEN $summary <> '' AND NOT n.summary CONTAINS $summary
                            THEN n.summary + ' ' + $summary ELSE n.summary END,
                        n.embedding = $emb, n.attributes = $attrs,
                        n.aliases_list = CASE
                            WHEN NOT $name IN coalesce(n.aliases_list, [])
                            THEN coalesce(n.aliases_list, []) + $name
                            ELSE n.aliases_list END""",
                        uuid=match_uuid, summary=summary, name=name,
                        emb=emb, attrs=json.dumps(attrs, ensure_ascii=False),
                    )
                else:
                    labels = json.dumps([etype, "Entity"], ensure_ascii=False)
                    session.run(
                        """MERGE (n:Entity {graph_id: $gid, name: $name})
                        ON CREATE SET n.uuid = $uuid, n.labels = $labels,
                            n.labels_list = $labels_list, n.aliases_list = [],
                            n.summary = $summary, n.attributes = $attrs,
                            n.embedding = $emb, n.created_at = datetime()
                        ON MATCH SET
                            n.summary = CASE WHEN $summary <> '' AND NOT n.summary CONTAINS $summary
                                THEN n.summary + ' ' + $summary ELSE n.summary END,
                            n.embedding = $emb, n.attributes = $attrs""",
                        gid=graph_id, name=name, uuid=str(uuid_lib.uuid4()),
                        labels=labels, labels_list=[etype, "Entity"],
                        summary=summary,
                        attrs=json.dumps(attrs, ensure_ascii=False),
                        emb=emb,
                    )

    def _find_duplicate_entity(self, session, graph_id: str, name: str,
                                embedding: List[float]) -> Optional[str]:
        """Three-stage entity dedup: exact match -> embedding similarity -> LLM confirm."""
        # Stage 1: Exact normalized name match
        norm = name.strip().lower()
        result = session.run(
            """MATCH (n:Entity {graph_id: $gid})
            WHERE toLower(trim(n.name)) = $norm
            RETURN n.uuid AS uuid LIMIT 1""",
            gid=graph_id, norm=norm,
        )
        record = result.single()
        if record:
            return record["uuid"]

        # Also check aliases
        result = session.run(
            """MATCH (n:Entity {graph_id: $gid})
            WHERE $norm IN [a IN coalesce(n.aliases_list, []) | toLower(trim(a))]
            RETURN n.uuid AS uuid LIMIT 1""",
            gid=graph_id, norm=norm,
        )
        record = result.single()
        if record:
            return record["uuid"]

        # Stage 2: Embedding semantic match via vector index
        if not embedding:
            return None

        auto_threshold = Config.DEDUP_AUTO_MERGE_THRESHOLD
        candidate_threshold = Config.DEDUP_CANDIDATE_THRESHOLD

        try:
            result = session.run(
                """CALL db.index.vector.queryNodes('entity_embedding', $top_k, $emb)
                YIELD node, score
                WITH node, score WHERE node.graph_id = $gid AND score > $threshold
                RETURN node.uuid AS uuid, node.name AS name,
                       node.summary AS summary, score
                ORDER BY score DESC""",
                top_k=10, emb=embedding, gid=graph_id,
                threshold=candidate_threshold,
            )
            candidates = [dict(r) for r in result]
        except Exception as e:
            logger.warning(f"Vector search for dedup failed: {e}")
            return None

        if not candidates:
            return None

        # Single high-confidence match -> auto-merge
        if len(candidates) == 1 and candidates[0]["score"] > auto_threshold:
            logger.info(f"Auto-merge: '{name}' -> '{candidates[0]['name']}' (score={candidates[0]['score']:.3f})")
            return candidates[0]["uuid"]

        # Stage 3: LLM confirmation for ambiguous candidates
        return self._llm_confirm_dedup(name, candidates)

    def _llm_confirm_dedup(self, name: str, candidates: List[Dict]) -> Optional[str]:
        """Stage 3: LLM confirms whether new entity matches any candidate."""
        cands_for_prompt = []
        for i, c in enumerate(candidates):
            cands_for_prompt.append({
                "candidate_id": i,
                "name": c.get("name", ""),
                "summary": c.get("summary", "")[:200],
                "similarity": round(c.get("score", 0), 3),
            })

        prompt = f"""判断新实体是否与候选实体中的某一个是同一个真实世界的实体。

<新实体>
名称: {name}
</新实体>

<候选实体（按语义相似度排序）>
{json.dumps(cands_for_prompt, ensure_ascii=False, indent=1)}
</候选实体>

规则：
- 只有非常确信是同一实体时才匹配
- 注意各种称呼：姓名/字号/绰号/职业称呼/简称/描述性称呼
- 关注描述中的身份、职业、特征线索
- 不同但相关的实体不要合并

返回 JSON：{{"duplicate_candidate_id": <id或-1>, "reason": "原因"}}"""

        try:
            result = self._extractor.llm_client.chat_json(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=256,
            )
            cid = result.get("duplicate_candidate_id", -1)
            reason = result.get("reason", "")
            if 0 <= cid < len(candidates):
                matched = candidates[cid]
                logger.info(f"LLM confirm: '{name}' -> '{matched['name']}' ({reason})")
                return matched["uuid"]
        except Exception as e:
            logger.warning(f"LLM confirm failed for '{name}': {e}")

        return None

    def _store_relations(self, graph_id: str, relations: List[Dict], episode_uuid: str):
        if not relations:
            return
        facts = [r.get("fact", r.get("type", "")) for r in relations]
        try:
            embeddings = self._voyage.embed(facts)
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            embeddings = [[] for _ in relations]
        with self._driver.session() as session:
            for rel, emb in zip(relations, embeddings):
                source_name = rel.get("source", "")
                target_name = rel.get("target", "")
                rel_type = rel.get("type", "RELATES")
                fact = rel.get("fact", "")
                src = session.run(
                    """MATCH (n:Entity {graph_id: $gid})
                    WHERE n.name = $name OR $name IN coalesce(n.aliases_list, [])
                    RETURN n.uuid AS uuid LIMIT 1""",
                    gid=graph_id, name=source_name,
                ).single()
                tgt = session.run(
                    """MATCH (n:Entity {graph_id: $gid})
                    WHERE n.name = $name OR $name IN coalesce(n.aliases_list, [])
                    RETURN n.uuid AS uuid LIMIT 1""",
                    gid=graph_id, name=target_name,
                ).single()
                src_uuid = src["uuid"] if src else ""
                tgt_uuid = tgt["uuid"] if tgt else ""
                fact_uuid = str(uuid_lib.uuid4())
                episodes_json = json.dumps([episode_uuid], ensure_ascii=False)
                session.run(
                    """CREATE (f:Fact {
                        uuid: $uuid, graph_id: $gid, name: $name,
                        fact: $fact, fact_type: $fact_type,
                        source_node_uuid: $src_uuid, target_node_uuid: $tgt_uuid,
                        source_node_name: $src_name, target_node_name: $tgt_name,
                        attributes: '{}', embedding: $emb,
                        episodes: $episodes,
                        created_at: datetime(), valid_at: datetime(),
                        invalid_at: null, expired_at: null
                    })""",
                    uuid=fact_uuid, gid=graph_id, name=rel_type,
                    fact=fact, fact_type="relation",
                    src_uuid=src_uuid, tgt_uuid=tgt_uuid,
                    src_name=source_name, tgt_name=target_name,
                    emb=emb, episodes=episodes_json,
                )
                if src_uuid and tgt_uuid:
                    session.run(
                        """MATCH (a:Entity {uuid: $src}), (b:Entity {uuid: $tgt})
                        CREATE (a)-[:RELATES {fact_uuid: $fuuid}]->(b)""",
                        src=src_uuid, tgt=tgt_uuid, fuuid=fact_uuid,
                    )

    def search(self, graph_id: str, query: str, limit: int = 10,
               scope: str = "edges", reranker: str = "rrf") -> SearchResponse:
        try:
            query_embedding = self._voyage.embed_single(query)
        except Exception as e:
            logger.error(f"Query embedding failed: {e}")
            return SearchResponse()
        top_k = limit * 3
        if scope == "edges":
            vector_uuids = self._vector_search_facts(graph_id, query_embedding, top_k)
            bm25_uuids = self._bm25_search_facts(graph_id, query, top_k)
            fused_uuids = _reciprocal_rank_fusion(vector_uuids, bm25_uuids)[:limit]
            edges = self._fetch_facts_by_uuids(fused_uuids)
            return SearchResponse(edges=edges)
        else:
            vector_uuids = self._vector_search_entities(graph_id, query_embedding, top_k)
            bm25_uuids = self._bm25_search_entities(graph_id, query, top_k)
            fused_uuids = _reciprocal_rank_fusion(vector_uuids, bm25_uuids)[:limit]
            nodes = self._fetch_entities_by_uuids(fused_uuids)
            return SearchResponse(nodes=nodes)

    def _vector_search_facts(self, graph_id, embedding, top_k) -> List[str]:
        with self._driver.session() as session:
            try:
                result = session.run(
                    """CALL db.index.vector.queryNodes('fact_embedding', $top_k, $emb)
                    YIELD node, score WHERE node.graph_id = $gid
                    RETURN node.uuid AS uuid""",
                    top_k=top_k, emb=embedding, gid=graph_id,
                )
                return [r["uuid"] for r in result]
            except Exception as e:
                logger.warning(f"Vector search failed: {e}")
                return []

    def _vector_search_entities(self, graph_id, embedding, top_k) -> List[str]:
        with self._driver.session() as session:
            try:
                result = session.run(
                    """CALL db.index.vector.queryNodes('entity_embedding', $top_k, $emb)
                    YIELD node, score WHERE node.graph_id = $gid
                    RETURN node.uuid AS uuid""",
                    top_k=top_k, emb=embedding, gid=graph_id,
                )
                return [r["uuid"] for r in result]
            except Exception as e:
                logger.warning(f"Vector search failed: {e}")
                return []


    def _bm25_search_facts(self, graph_id, query, top_k) -> List[str]:
        with self._driver.session() as session:
            try:
                result = session.run(
                    """CALL db.index.fulltext.queryNodes('fact_fulltext', $query)
                    YIELD node, score WHERE node.graph_id = $gid
                    RETURN node.uuid AS uuid LIMIT $top_k""",
                    query=query, gid=graph_id, top_k=top_k,
                )
                return [r["uuid"] for r in result]
            except Exception as e:
                logger.warning(f"BM25 search failed: {e}")
                return []

    def _bm25_search_entities(self, graph_id, query, top_k) -> List[str]:
        with self._driver.session() as session:
            try:
                result = session.run(
                    """CALL db.index.fulltext.queryNodes('entity_fulltext', $query)
                    YIELD node, score WHERE node.graph_id = $gid
                    RETURN node.uuid AS uuid LIMIT $top_k""",
                    query=query, gid=graph_id, top_k=top_k,
                )
                return [r["uuid"] for r in result]
            except Exception as e:
                logger.warning(f"BM25 search failed: {e}")
                return []

    def _fetch_facts_by_uuids(self, uuids: List[str]) -> List[EdgeObject]:
        if not uuids:
            return []
        with self._driver.session() as session:
            result = session.run(
                "MATCH (f:Fact) WHERE f.uuid IN $uuids RETURN f",
                uuids=uuids,
            )
            facts_map = {r["f"]["uuid"]: _neo4j_fact_to_object(r["f"]) for r in result}
        return [facts_map[u] for u in uuids if u in facts_map]

    def _fetch_entities_by_uuids(self, uuids: List[str]) -> List[NodeObject]:
        if not uuids:
            return []
        with self._driver.session() as session:
            result = session.run(
                "MATCH (n:Entity) WHERE n.uuid IN $uuids RETURN n",
                uuids=uuids,
            )
            nodes_map = {r["n"]["uuid"]: _neo4j_node_to_object(r["n"]) for r in result}
        return [nodes_map[u] for u in uuids if u in nodes_map]


# ============== Main Client ==============

class LocalGraphClient:
    """Drop-in replacement for zep_cloud.client.Zep"""

    def __init__(self, api_key: str = None, neo4j_uri: str = None,
                 neo4j_user: str = None, neo4j_password: str = None,
                 voyage_api_key: str = None, llm_client: LLMClient = None):
        uri = neo4j_uri or Config.NEO4J_URI
        user = neo4j_user or Config.NEO4J_USER
        password = neo4j_password or Config.NEO4J_PASSWORD
        v_key = voyage_api_key or Config.VOYAGE_API_KEY
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        voyage = VoyageEmbedding(api_key=v_key)
        extractor = LLMEntityExtractor(llm_client=llm_client)
        self.graph = GraphNamespace(self._driver, voyage, extractor)

    def close(self):
        self._driver.close()
