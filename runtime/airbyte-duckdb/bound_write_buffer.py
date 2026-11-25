"""Bound the pinned DuckDB connector's memory use between Airbyte checkpoints."""

from pathlib import Path

import destination_duckdb.destination


path = Path(destination_duckdb.destination.__file__)
source = path.read_text()
source = source.replace("import uuid\n", "import uuid\nfrom pathlib import Path\n", 1)
connect = "        con = duckdb.connect(database=path, read_only=False, config=duckdb_config)"
assert source.count(connect + "\n\n        con.execute") == 1
source = source.replace(connect, '''        job_id = os.getenv("WORKER_JOB_ID")
        local_job = bool(job_id) and path.startswith("/local/")
        if path.startswith("/local/"):
            duckdb_config.update(memory_limit="768MB" if local_job else "1024MB", threads=1)
        # The connector generates unique IDs. An ART index on those IDs grows
        # with every batch and cannot be evicted by the DuckDB buffer manager.
        identifier_constraint = "NOT NULL" if local_job else "PRIMARY KEY"
''' + connect, 1)
identifier = "_airbyte_ab_id TEXT PRIMARY KEY"
assert source.count(identifier) == 1
source = source.replace(identifier, "_airbyte_ab_id TEXT {identifier_constraint}", 1)

start = source.index("        buffer = defaultdict(lambda: defaultdict(list))\n")
end = source.index("    @staticmethod\n    def _safe_write", start)
source = source[:start] + '''        pending = Path(f"{path}.airbyte-pending.json") if local_job else None
        if pending is not None and pending.is_file():
            previous = json.loads(pending.read_text())
            if previous["job_id"] == job_id:
                # The retry can replay rows written after the last checkpoint,
                # including rows flushed when a failed source closed its pipe.
                old_prefix = str(uuid.UUID(previous["prefix"] + "000000000000"))[:24]
                end = previous.get("end", 0xffffffffffff)
                batch_size = 500_000 if "end" in previous else end
                for stream_name in streams:
                    for start in range(0, end, batch_size):
                        lower = f"{old_prefix}{start:012x}"
                        upper = f"{old_prefix}{min(start + batch_size, end):012x}"
                        con.execute(
                            f"DELETE FROM {schema_name}._airbyte_raw_{stream_name} "
                            "WHERE _airbyte_ab_id >= ? AND _airbyte_ab_id < ?",
                            [lower, upper],
                        )
            pending.unlink()

        prefix = str(uuid.uuid4())[:24]
        offset = 0
        buffer = defaultdict(lambda: defaultdict(list))
        rows = size = 0

        def flush():
            nonlocal rows, size
            if pending is not None and buffer:
                temporary = pending.with_suffix(".tmp")
                temporary.write_text(json.dumps({"job_id": job_id, "prefix": prefix, "end": offset}))
                temporary.replace(pending)
            for stream_name in buffer:
                self._safe_write(con=con, buffer=buffer, schema_name=schema_name, stream_name=stream_name)
            buffer.clear()
            rows = size = 0

        try:
            # Outside an Airbyte job, preserve the checkpoint transaction.
            if pending is None:
                con.begin()
            for message in input_messages:
                if message.type == Type.STATE:
                    flush()
                    if pending is None:
                        con.commit()
                    else:
                        pending.unlink(missing_ok=True)
                        prefix = str(uuid.uuid4())[:24]
                        offset = 0
                    yield message
                    if pending is None:
                        con.begin()
                elif message.type == Type.RECORD:
                    stream_name = message.record.stream
                    if stream_name not in streams:
                        logger.debug(f"Stream {stream_name} was not present in configured streams, skipping")
                        continue
                    payload = json.dumps(message.record.data)
                    record_id = str(uuid.uuid4()) if pending is None else f"{prefix}{offset:012x}"
                    offset += 1
                    buffer[stream_name]["_airbyte_ab_id"].append(record_id)
                    buffer[stream_name]["_airbyte_emitted_at"].append(datetime.datetime.now().isoformat())
                    buffer[stream_name]["_airbyte_data"].append(payload)
                    rows += 1
                    size += len(payload)
                    if rows >= 8192 or size >= 8 * 1024 * 1024:
                        flush()
                else:
                    logger.info(f"Message type {message.type} not supported, skipping")
            flush()
            if pending is None:
                con.commit()
        finally:
            con.close()

''' + source[end:]
path.write_text(source)
