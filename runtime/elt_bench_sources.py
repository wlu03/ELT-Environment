"""Stage the source data and task files from the original ELT-Bench checkout."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
import re
import shutil
from urllib.parse import parse_qs, urlsplit
from zipfile import ZipFile

import numpy as np
import pandas as pd
import yaml


@dataclass(frozen=True)
class BenchmarkTask:
    name: str
    bench: Path
    assets: Path
    config: dict
    rest: dict | None

    @property
    def original(self):
        return self.bench / "elt-bench" / "snowflake" / self.name

    @property
    def gold(self):
        return self.assets / "gt_snowflake" / self.name

    @property
    def postgres_script(self):
        return self.bench / "setup" / "sources" / self.name / "postgres.sh"

    @property
    def dbt_utils(self) -> Path | None:
        specification = (self.original / "data_model.yaml").read_text()
        if re.search(r"\bdbt[ _-]utils\b", specification, re.IGNORECASE):
            return self.assets / "dbt_utils"
        return None

    def files(self):
        for entry in self.config.get("flat_files", []):
            table, extension = entry["table"], entry["format"]
            url = urlsplit(entry["path"])
            key, = parse_qs(url.query)["id"]
            if (url.hostname != "drive.google.com" or not re.fullmatch(r"[\w-]+", key)
                    or extension not in {"csv", "jsonl", "parquet"}):
                raise ValueError(f"{self.name}.{table}: unsupported flat-file link")
            yield entry, self.assets / "flat_files" / f"{key}.{extension}"

    def database_files(self):
        for backend in ("postgres", "mongodb"):
            for table in self.config.get(backend, {}).get("config", {}).get("tables", []):
                yield backend, table, f"{table}.csv"
        for entry in self.config.get("aws_s3", {}).get("data", []):
            url = urlsplit(entry["path"])
            yield "s3", entry["table"], url.path.lstrip("/")


def load_task(bench: Path, assets: Path, name: str, upstream_rest: dict, hosts: dict) -> BenchmarkTask:
    original = bench / "elt-bench" / "snowflake" / name
    config = yaml.safe_load((original / "config.yaml").read_text())
    rest = None
    if "custom_api" in config:
        names = config["custom_api"]["config"]["tables"]
        rest = {key: deepcopy(upstream_rest[key]) for key in ("version", "type", "spec")}
        rest["check"] = {"type": "CheckStream", "stream_names": names[:1]}
        rest["definitions"] = {
            "base_requester": {
                "type": "HttpRequester", "url_base": f"http://{hosts['elt-api']}:5005",
            },
            "streams": {table: upstream_rest["definitions"]["streams"][table] for table in names},
        }
        rest["streams"] = [{"$ref": f"#/definitions/streams/{table}"} for table in names]
        rest["schemas"] = {table: upstream_rest["schemas"][table] for table in names}
    return BenchmarkTask(name, bench, assets, config, rest)


def check_task(task: BenchmarkTask, counts: dict, database_names: dict, api: ZipFile) -> None:
    specification = yaml.safe_load((task.original / "data_model.yaml").read_text())
    models = {model["name"].casefold() for model in specification["models"]}
    gold = {path.stem.casefold() for path in task.gold.glob("*.csv") if path.stat().st_size}
    if not models or models != gold:
        raise ValueError(f"{task.name}: missing or mismatched v5 ground truth")
    sources = []
    for _, table, filename in task.database_files():
        sources.append(table)
        if f"data/{task.name}/{filename}".casefold() not in database_names:
            raise FileNotFoundError(f"{task.name}: database fixture {filename} is missing")
    if "postgres" in task.config and not task.postgres_script.is_file():
        raise FileNotFoundError(f"{task.name}: PostgreSQL setup script is missing")
    for entry, file in task.files():
        sources.append(entry["table"])
        if not file.is_file() or not file.stat().st_size:
            raise FileNotFoundError(f"{task.name}: flat-file fixture {file.name} is missing")
    if task.rest:
        for table, stream in task.rest["definitions"]["streams"].items():
            sources.append(table)
            route = stream["retriever"]["requester"]["path"]
            api.getinfo(f"data{route}.csv")
    if sorted(s.casefold() for s in sources) != sorted(s.casefold() for s in counts):
        raise ValueError(f"{task.name}: configured source streams do not match Stage 1")
    if task.dbt_utils and not (task.dbt_utils / "dbt_project.yml").is_file():
        raise FileNotFoundError(f"{task.name}: the offline dbt_utils package is missing")


def copy_member(archive: ZipFile, member: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with archive.open(member) as source, output.open("wb") as handle:
        shutil.copyfileobj(source, handle, length=1024 * 1024)


def write_records(csv_file, output: Path, *, for_mongo: bool = False, stringify_loyalty=False):
    """Preserve the upstream pandas CSV interpretation for REST and MongoDB."""

    frame = pd.read_csv(csv_file).replace({np.nan: None})
    output.parent.mkdir(parents=True, exist_ok=True)
    if for_mongo:
        with output.open("w") as handle:
            for row in frame.to_dict(orient="records"):
                document = {key: value for key, value in row.items() if value is not None}
                if stringify_loyalty and isinstance(document.get("loyalty"), (int, float)):
                    document["loyalty"] = str(document["loyalty"])
                handle.write(json.dumps(document, default=str) + "\n")
        return
    with output.open("w") as handle:
        json.dump({"data": frame.to_dict(orient="records")}, handle, default=str)
    (output.parent / "index.json").write_text(json.dumps({
        "pages": [output.name], "row_count": len(frame),
    }))


def stage_sources(task: BenchmarkTask, database: ZipFile, database_names: dict, api: ZipFile,
                  root: Path, hosts: dict) -> tuple[dict, dict]:
    config = deepcopy(task.config)
    config.pop("snowflake")
    config["Airbyte"]["config"].pop("snowflake_definition_id", None)
    tables = {}
    postgres = config.get("postgres", {}).get("config")
    seed_script = root / "postgres.sh"
    if postgres:
        script = task.postgres_script.read_text()
        database_name = re.search(r'^DB_NAME="([\w]+)"', script, re.MULTILINE)
        if not database_name or "DB_PORT=5433" not in script:
            raise ValueError(f"{task.name}: unsupported PostgreSQL setup layout")
        pg_database = database_name.group(1)
        postgres.update(host=hosts["elt-postgres"], database=pg_database)
        seed_script.write_text(re.sub(r"^DB_PORT=5433\b", "DB_PORT=5432", script, flags=re.M))

    mongo = config.get("mongodb", {}).get("config")
    if mongo:
        mongo["connection_string"] = mongo["connection_string"].replace(
            "elt-mongodb", hosts["elt-mongodb"],
        )
    s3 = config.get("aws_s3")
    if s3:
        s3["AWS_ENDPOINT_URL"] = f"http://{hosts['elt-localstack']}:4566"
    object_paths = {entry["table"]: urlsplit(entry["path"]) for entry in (s3 or {}).get("data", [])}

    rendered = root / "rendered"
    for backend, table, filename in task.database_files():
        member = database_names[f"data/{task.name}/{filename}".casefold()]
        if backend == "postgres":
            copy_member(database, member, root / "data" / pg_database / Path(member).name)
        elif backend == "mongodb":
            relative = Path("mongodb") / f"{table}.jsonl"
            with database.open(member) as data:
                write_records(data, rendered / relative, for_mongo=True,
                              stringify_loyalty=(task.name == "card_games" and table == "cards"))
            tables[table] = {"backend": backend, "collection": table, "rendered_file": str(relative)}
        else:
            relative = Path("s3") / table
            object_path = object_paths[table]
            copy_member(database, member, rendered / relative / Path(member).name)
            tables[table] = {
                "backend": "s3", "bucket": object_path.hostname, "object_key": filename,
                "rendered_dir": str(relative), "rendered_parts_glob": Path(member).name,
            }

    if task.rest:
        for table, stream in task.rest["definitions"]["streams"].items():
            route = stream["retriever"]["requester"]["path"]
            relative = Path("rest") / table
            with api.open(f"data{route}.csv") as data:
                write_records(data, rendered / relative / "page.json")
            tables[table] = {"backend": "rest", "route": route, "rendered_dir": str(relative)}

    for entry, (_, original) in zip(config.get("flat_files", []), task.files(), strict=True):
        table = entry["table"]
        relative = Path("files") / f"{table}.{entry['format']}"
        (rendered / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, rendered / relative)
        entry["path"] = f"https://{hosts['elt-files']}:8443/{relative.as_posix()}"
        tables[table] = {"backend": "files", "url": entry["path"], "rendered_file": str(relative)}

    rendered.mkdir(exist_ok=True)
    return config, {"database": mongo["database"] if mongo else task.name, "tables": tables}
