FROM airbyte/destination-duckdb:0.6.0@sha256:0cc88166429a818a24f01606a57dd4c4fc856bffd40b9f896026738276b351f1

USER root
RUN python -c 'from pathlib import Path; import inspect; from airbyte_cdk.destinations import Destination; p = Path(inspect.getfile(Destination)); text = p.read_text(); old = "print(message.json(exclude_unset=True))"; assert text.count(old) == 1; p.write_text(text.replace(old, "print(message.json(exclude_unset=True, by_alias=True))"))'
COPY bound_write_buffer.py /tmp/bound_write_buffer.py
RUN python /tmp/bound_write_buffer.py && rm /tmp/bound_write_buffer.py
USER airbyte
