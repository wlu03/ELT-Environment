ARG BASE_IMAGE=elt-swe
FROM ${BASE_IMAGE}

RUN pip install --no-cache-dir dbt-duckdb==1.10.0 duckdb==1.4.2 swe-rex==1.4.0 flake8==7.3.0
RUN mkdir -p /tmp/elt-provider /opt/elt-terraform/providers \
    && printf 'terraform {\n required_providers {\n airbyte = { source = "airbytehq/airbyte", version = "0.6.5" }\n }\n}\n' > /tmp/elt-provider/main.tf \
    && terraform -chdir=/tmp/elt-provider providers mirror /opt/elt-terraform/providers \
    && printf 'provider_installation {\n filesystem_mirror { path = "/opt/elt-terraform/providers" }\n}\n' > /opt/elt-terraform/terraform.rc \
    && rm -r /tmp/elt-provider
ENV TF_CLI_CONFIG_FILE=/opt/elt-terraform/terraform.rc CHECKPOINT_DISABLE=1 PIP_DISABLE_PIP_VERSION_CHECK=1
