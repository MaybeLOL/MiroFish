"""
LLM-based entity and relation extraction from text.
Uses the ontology schema to guide extraction.
"""

import json
import re
from typing import Dict, Any, List, Tuple, Optional
from ..utils.llm_client import LLMClient
from ..utils.logger import get_logger

logger = get_logger(__name__)

EXTRACTION_PROMPT = """你是一个专业的实体与关系提取引擎，擅长从各类文本（包括中文古典文学、新闻、学术论文等）中精确识别实体和关系。

## 本体定义

### 实体类型：
{entity_types}

### 关系类型：
{edge_types}

## 待分析文本：
{text}

## 提取规则（严格遵守）：

### 实体提取规则：
1. 人名必须使用完整姓名，不得截断。例如"宋集薪"不能写成"宋集"或"集薪"
2. 如果文本中出现别名、字号、绰号，在 summary 中注明，但 name 使用最常用的全名
3. 如果实体是某人的从属（如婢女、仆人），仍然作为独立实体提取，在 summary 中说明从属关系
4. 地名、组织名同样使用完整名称
5. 不要合并不同的实体，即使它们关系密切
6. 每个实体的 summary 应包含文本中提到的关键信息（身份、特征、行为等）

### 关系提取规则：
1. source 和 target 必须使用与 entities 中完全一致的 name
2. fact 应该是一句完整的陈述，描述两个实体之间的具体关系或事件
3. 不要编造文本中没有提到的关系
4. 如果同一对实体之间有多种关系，分别提取为多条记录

### 输出格式：
返回严格合法的 JSON，不要包含任何其他文字：
{{
  "entities": [
    {{"name": "完整实体名", "type": "实体类型", "summary": "基于文本的简要描述", "attributes": {{}}}}
  ],
  "relations": [
    {{"source": "源实体完整名", "target": "目标实体完整名", "type": "关系类型", "fact": "关系的具体描述"}}
  ]
}}

如果文本中没有可提取的实体或关系，返回 {{"entities": [], "relations": []}}。"""


class LLMEntityExtractor:

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client or LLMClient()

    def extract(self, text: str, ontology: Dict[str, Any]) -> Tuple[List[Dict], List[Dict]]:
        """Extract entities and relations, then resolve duplicates."""
        entity_types_str = self._format_entity_types(ontology)
        edge_types_str = self._format_edge_types(ontology)

        prompt = EXTRACTION_PROMPT.format(
            entity_types=entity_types_str,
            edge_types=edge_types_str,
            text=text,
        )

        try:
            result = self.llm_client.chat_json(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=4096,
            )
            entities = result.get("entities", [])
            relations = result.get("relations", [])
            logger.info(f"Extracted {len(entities)} entities, {len(relations)} relations")

            return entities, relations
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return [], []

    def _format_entity_types(self, ontology: Dict[str, Any]) -> str:
        lines = []
        for et in ontology.get("entity_types", []):
            name = et.get("name", et.get("type", "Unknown"))
            desc = et.get("description", "")
            attrs = et.get("attributes", [])
            attr_str = ", ".join(
                a.get("name", "") for a in attrs
            ) if attrs else "none"
            lines.append(f"- {name}: {desc} (attributes: {attr_str})")
        return "\n".join(lines) if lines else "No entity types defined"

    def _format_edge_types(self, ontology: Dict[str, Any]) -> str:
        lines = []
        for et in ontology.get("edge_types", []):
            name = et.get("name", et.get("type", "Unknown"))
            desc = et.get("description", "")
            src = et.get("source_entity_type", "Any")
            tgt = et.get("target_entity_type", "Any")
            lines.append(f"- {name}: {desc} ({src} -> {tgt})")
        return "\n".join(lines) if lines else "No relation types defined"

