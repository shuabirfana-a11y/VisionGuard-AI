from __future__ import annotations

from collections import Counter
import json
from math import log, sqrt
from pathlib import Path
import re

from app.schemas import KnowledgeEvidence


CATEGORY_CONTEXT = {
    "fire": "火灾 火焰 明火 燃烧 报警 疏散 应急",
    "smoke": "烟雾 烟气 蒸汽 粉尘 雾气 温升 排查",
    "no_detection": "未检出 漏检 遮挡 画质 安全 复核",
}


def _tokens(text: str) -> list[str]:
    normalized = re.sub(r"\s+", "", text.lower())
    han_runs = re.findall(r"[\u3400-\u9fff]+", normalized)
    chinese = [
        run[index:index + 2]
        for run in han_runs
        for index in range(max(0, len(run) - 1))
    ]
    latin = re.findall(r"[a-z0-9][a-z0-9_-]+", normalized)
    return chinese + latin


def _normalize(vector: dict[str, float]) -> dict[str, float]:
    magnitude = sqrt(sum(value * value for value in vector.values()))
    return (
        {term: value / magnitude for term, value in vector.items()}
        if magnitude
        else {}
    )


class SafetyKnowledgeService:
    """Category-constrained hybrid RAG over a small, versioned safety corpus."""

    retrieval_method = "category-constrained-tfidf-rag-v1"

    def __init__(self, knowledge_path: Path | None = None) -> None:
        path = knowledge_path or Path(__file__).parents[1] / "knowledge" / "safety_rules.json"
        self.rules = json.loads(path.read_text(encoding="utf-8"))
        tokenized = [_tokens(self._searchable(rule)) for rule in self.rules]
        document_frequency = Counter(
            term for terms in tokenized for term in set(terms)
        )
        count = len(self.rules)
        self.idf = {
            term: log((1 + count) / (1 + frequency)) + 1
            for term, frequency in document_frequency.items()
        }
        self.rule_vectors = [self._vector(terms) for terms in tokenized]

    async def retrieve(
        self, categories: list[str], task: str, top_k: int = 3
    ) -> list[KnowledgeEvidence]:
        if top_k < 1:
            raise ValueError("top_k必须大于0")
        wanted = set(categories) or {"no_detection"}
        query = task + " " + " ".join(CATEGORY_CONTEXT.get(item, item) for item in wanted)
        query_vector = self._vector(_tokens(query))
        results = []
        for index, rule in enumerate(self.rules):
            if rule["category"] not in wanted:
                continue
            semantic_score = sum(
                value * self.rule_vectors[index].get(term, 0.0)
                for term, value in query_vector.items()
            )
            keyword_hits = [
                keyword for keyword in rule.get("keywords", []) if keyword in task
            ]
            authority_score = (
                0.08
                if rule.get("authority_level") in {"law", "department_rule"}
                else 0.0
            )
            score = round(
                min(
                    1.0,
                    0.48
                    + semantic_score * 0.32
                    + min(0.12, len(keyword_hits) * 0.04)
                    + authority_score,
                ),
                3,
            )
            results.append(KnowledgeEvidence(
                rule_id=rule["id"],
                title=rule["title"],
                matched_category=rule["category"],
                basis=rule["basis"],
                recommended_actions=rule["recommended_actions"],
                source=rule["source"],
                source_title=rule["source_title"],
                source_section=rule["source_section"],
                source_version=rule["source_version"],
                source_url=rule.get("source_url"),
                authority_level=rule.get("authority_level", "internal_method"),
                applicability=rule.get("applicability", ""),
                citation_id=rule["id"],
                retrieval_score=score,
                retrieval_method=self.retrieval_method,
                matched_terms=keyword_hits,
            ))
        return sorted(
            results,
            key=lambda item: (
                item.retrieval_score,
                item.authority_level != "internal_method",
                item.rule_id,
            ),
            reverse=True,
        )[:top_k]

    @staticmethod
    def _searchable(rule: dict) -> str:
        return " ".join((
            CATEGORY_CONTEXT.get(rule["category"], rule["category"]),
            rule["title"],
            rule["basis"],
            " ".join(rule["recommended_actions"]),
            " ".join(rule.get("keywords", [])),
            rule.get("applicability", ""),
        ))

    def _vector(self, terms: list[str]) -> dict[str, float]:
        counts = Counter(term for term in terms if term in self.idf)
        return _normalize({
            term: (1 + log(frequency)) * self.idf[term]
            for term, frequency in counts.items()
        })
