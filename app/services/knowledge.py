import json
from pathlib import Path

from app.schemas import KnowledgeEvidence


class SafetyKnowledgeService:
    def __init__(self, knowledge_path: Path | None = None) -> None:
        path = knowledge_path or Path(__file__).parents[1] / "knowledge" / "safety_rules.json"
        self.rules = json.loads(path.read_text(encoding="utf-8"))

    async def retrieve(self, categories: list[str], task: str) -> list[KnowledgeEvidence]:
        wanted = set(categories)
        if not wanted:
            wanted.add("no_detection")
        results = []
        for rule in self.rules:
            if rule["category"] in wanted:
                score = self._score(rule, task, rule["category"] in set(categories))
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
                    citation_id=rule["id"],
                    retrieval_score=score,
                ))
        return sorted(results, key=lambda item: item.retrieval_score, reverse=True)

    @staticmethod
    def _score(rule: dict, task: str, category_match: bool) -> float:
        searchable = "".join((rule["title"], rule["basis"], "".join(rule["recommended_actions"])))
        task_chars = {char for char in task if char.strip()}
        overlap = len(task_chars.intersection(searchable)) / max(1, len(task_chars))
        return round(min(1.0, (0.75 if category_match else 0.45) + overlap * 0.25), 3)
