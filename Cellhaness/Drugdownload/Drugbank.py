import argparse
import csv
import json
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import requests


@dataclass(frozen=True)
class DrugRecord:
    name: str
    concept_id: str
    approved: Optional[bool]
    anti_neoplastic: Optional[bool]
    immunotherapy: Optional[bool]
    attributes: Tuple[Tuple[str, str], ...]


@dataclass(frozen=True)
class DrugConfig:
    drug_name: str
    drug_concept_id: str
    feed_time_h: float


class DGIdbClient:
    def __init__(self, *, endpoint: str = "https://dgidb.org/api/graphql", timeout_s: int = 60, user_agent: str = "celldrug/0.1") -> None:
        self.endpoint = str(endpoint)
        self.timeout_s = int(timeout_s)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": str(user_agent)})

    def query(self, *, query: str, variables: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        r = self.session.post(self.endpoint, json={"query": str(query), "variables": variables or {}}, timeout=float(self.timeout_s))
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict) and data.get("errors"):
            raise RuntimeError(f"DGIdb GraphQL error: {data['errors']}")
        if not isinstance(data, dict) or "data" not in data:
            raise RuntimeError("DGIdb GraphQL response missing data")
        return dict(data["data"])

    def list_drugs(self, *, first: int = 200, after: Optional[str] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        q = """
        query($first:Int, $after:String) {
          drugs(first:$first, after:$after) {
            pageInfo { hasNextPage endCursor }
            edges {
              node {
                name
                conceptId
                approved
                antiNeoplastic
                immunotherapy
                drugAttributes { name value }
              }
            }
          }
        }
        """
        out = self.query(query=q, variables={"first": int(first), "after": after})
        conn = out.get("drugs") or {}
        edges = list(conn.get("edges") or [])
        page = dict(conn.get("pageInfo") or {})
        nodes = [dict(e.get("node") or {}) for e in edges]
        return nodes, page


def _is_small_molecule(drug_node: Dict[str, Any]) -> bool:
    attrs = drug_node.get("drugAttributes")
    if not isinstance(attrs, list):
        return False
    for a in attrs:
        if not isinstance(a, dict):
            continue
        v = str(a.get("value") or "").strip().lower()
        if v == "small molecule" or "small molecule" in v:
            return True
    return False


def _to_drug_record(drug_node: Dict[str, Any]) -> DrugRecord:
    attrs = []
    for a in list(drug_node.get("drugAttributes") or []):
        if isinstance(a, dict):
            n = str(a.get("name") or "").strip()
            v = str(a.get("value") or "").strip()
            if n and v:
                attrs.append((n, v))
    return DrugRecord(
        name=str(drug_node.get("name") or "").strip(),
        concept_id=str(drug_node.get("conceptId") or "").strip(),
        approved=drug_node.get("approved"),
        anti_neoplastic=drug_node.get("antiNeoplastic"),
        immunotherapy=drug_node.get("immunotherapy"),
        attributes=tuple(attrs),
    )


def download_small_molecules(
    *,
    out_dir: str,
    limit: int = 0,
    page_size: int = 200,
    seed: int = 0,
    sleep_s: float = 0.0,
    cache_jsonl: str = "dgidb_small_molecules.jsonl",
    cache_csv: str = "dgidb_small_molecules.csv",
) -> List[DrugRecord]:
    os.makedirs(os.path.abspath(str(out_dir)), exist_ok=True)
    client = DGIdbClient()
    rs = random.Random(int(seed))

    acc: List[DrugRecord] = []
    after: Optional[str] = None
    while True:
        nodes, page = client.list_drugs(first=int(page_size), after=after)
        rs.shuffle(nodes)
        for n in nodes:
            if not _is_small_molecule(n):
                continue
            dr = _to_drug_record(n)
            if not dr.name or not dr.concept_id:
                continue
            acc.append(dr)
            if int(limit) > 0 and int(len(acc)) >= int(limit):
                break
        if int(limit) > 0 and int(len(acc)) >= int(limit):
            break
        if not bool(page.get("hasNextPage")):
            break
        after = page.get("endCursor")
        if float(sleep_s) > 0:
            time.sleep(float(sleep_s))

    jsonl_path = os.path.join(os.path.abspath(str(out_dir)), str(cache_jsonl))
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for d in acc:
            f.write(
                json.dumps(
                    {
                        "name": d.name,
                        "concept_id": d.concept_id,
                        "approved": d.approved,
                        "anti_neoplastic": d.anti_neoplastic,
                        "immunotherapy": d.immunotherapy,
                        "attributes": list(d.attributes),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    csv_path = os.path.join(os.path.abspath(str(out_dir)), str(cache_csv))
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "name",
                "concept_id",
                "approved",
                "anti_neoplastic",
                "immunotherapy",
                "drug_class",
                "indication",
            ],
        )
        w.writeheader()
        for d in acc:
            classes = [v for (k, v) in d.attributes if k == "Drug Class"]
            inds = [v for (k, v) in d.attributes if k == "Indication"]
            w.writerow(
                {
                    "name": d.name,
                    "concept_id": d.concept_id,
                    "approved": "" if d.approved is None else int(bool(d.approved)),
                    "anti_neoplastic": "" if d.anti_neoplastic is None else int(bool(d.anti_neoplastic)),
                    "immunotherapy": "" if d.immunotherapy is None else int(bool(d.immunotherapy)),
                    "drug_class": "|".join(classes),
                    "indication": "|".join(inds),
                }
            )

    return acc


def build_drugbuffer_config(*, drugs: Sequence[DrugRecord], feed_times_h: Sequence[float], out_csv: str) -> str:
    out_csv = os.path.abspath(str(out_csv))
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    rows: List[DrugConfig] = []
    for d in drugs:
        for t in feed_times_h:
            rows.append(DrugConfig(drug_name=str(d.name), drug_concept_id=str(d.concept_id), feed_time_h=float(t)))
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["drug_name", "drug_concept_id", "feed_time_h"])
        w.writeheader()
        for r in rows:
            w.writerow({"drug_name": r.drug_name, "drug_concept_id": r.drug_concept_id, "feed_time_h": float(r.feed_time_h)})
    return out_csv


class DrugBuffer:
    def __init__(self, cfg_csv: str, *, seed: int = 0) -> None:
        self.cfg_csv = os.path.abspath(str(cfg_csv))
        self.rng = random.Random(int(seed))
        self.rows: List[DrugConfig] = []
        with open(self.cfg_csv, "r", encoding="utf-8", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                self.rows.append(
                    DrugConfig(
                        drug_name=str(row.get("drug_name") or ""),
                        drug_concept_id=str(row.get("drug_concept_id") or ""),
                        feed_time_h=float(row.get("feed_time_h") or 0.0),
                    )
                )

    def __len__(self) -> int:
        return int(len(self.rows))

    def sample(self, k: int = 1) -> List[DrugConfig]:
        if not self.rows:
            return []
        if int(k) <= 1:
            return [self.rng.choice(self.rows)]
        return [self.rng.choice(self.rows) for _ in range(int(k))]


def _parse_floats(spec: str) -> List[float]:
    s = str(spec or "").strip()
    if not s:
        return []
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=str, default=os.path.join(os.path.dirname(__file__), "outputs", "celldrug"))
    p.add_argument("--limit", type=int, default=2000)
    p.add_argument("--page-size", type=int, default=200)
    p.add_argument("--feed-times-h", type=str, default="24")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sleep-s", type=float, default=0.0)
    args = p.parse_args()

    out_dir = os.path.abspath(str(args.out_dir))
    drugs = download_small_molecules(out_dir=out_dir, limit=int(args.limit), page_size=int(args.page_size), seed=int(args.seed), sleep_s=float(args.sleep_s))
    times = _parse_floats(str(args.feed_times_h)) or [24.0]
    cfg = build_drugbuffer_config(drugs=drugs, feed_times_h=times, out_csv=os.path.join(out_dir, "drugbuffer_config.csv"))
    print(json.dumps({"out_dir": out_dir, "n_drugs": int(len(drugs)), "config_csv": cfg}, ensure_ascii=False))


if __name__ == "__main__":
    main()
