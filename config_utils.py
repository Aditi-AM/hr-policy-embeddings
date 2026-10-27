"""
Central config loader — pulls all runtime configuration from AWS Secrets Manager.

Only env var required:
    HR_BOT_CONFIG_SECRET_NAME  (default: "hr-bot-config-dev")
    ASM_REGION                 (default: "ap-south-1")

Secret schema (hr-bot-config-dev):
{
    "opensearch_endpoint":          "https://search-xxx.ap-south-1.es.amazonaws.com",
    "opensearch_index":             "hr-docs-embeddings",
    "embedding_dimension":          1536,
    "ai_gateway_url":               "https://ai.titan.in/gateway",
    "embed_model":                  "gemini-multimodalembedding",
    "ai_gateway_token_secret_name": "titan-ess-entrapp-token-dev",
    "asm_region":                   "ap-south-1"
}
"""

import os
import json
import logging
import traceback
import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

_CONFIG_SECRET_NAME = os.environ.get("HR_BOT_CONFIG_SECRET_NAME", "hr-bot-config-dev")
_ASM_REGION         = os.environ.get("ASM_REGION", "ap-south-1")

# Module-level cache — fetched once per cold start
_config: dict = None


def get_config() -> dict:
    """Return the config dict, fetching from ASM on the first call."""
    global _config
    if _config is not None:
        logger.info("[config] get_config: returning cached config")
        return _config

    logger.info(
        f"[config] get_config: loading from ASM "
        f"secret='{_CONFIG_SECRET_NAME}' region='{_ASM_REGION}'"
    )
    try:
        _config = _load_from_asm(_CONFIG_SECRET_NAME)
    except Exception as e:
        logger.error(
            f"[config] get_config: FAILED to load config — "
            f"type={type(e).__name__} error={e}\n{traceback.format_exc()}"
        )
        raise

    logger.info(
        f"[config] get_config: loaded OK — "
        f"opensearch_index={_config.get('opensearch_index')} "
        f"embed_model={_config.get('embed_model')} "
        f"ai_gateway_url={_config.get('ai_gateway_url')} "
        f"token_secret={_config.get('ai_gateway_token_secret_name')}"
    )
    return _config


def _load_from_asm(secret_name: str) -> dict:
    """Fetch and parse a JSON secret from AWS Secrets Manager."""
    logger.info(f"[config] _load_from_asm: fetching secret='{secret_name}' region='{_ASM_REGION}'")

    try:
        client = boto3.client("secretsmanager", region_name=_ASM_REGION)
    except Exception as e:
        logger.error(
            f"[config] _load_from_asm: failed to create ASM client — "
            f"type={type(e).__name__} region='{_ASM_REGION}' error={e}\n"
            f"{traceback.format_exc()}"
        )
        raise RuntimeError(
            f"Cannot create Secrets Manager client (region={_ASM_REGION}): {e}"
        ) from e

    try:
        response = client.get_secret_value(SecretId=secret_name)
        logger.info(f"[config] _load_from_asm: fetched secret '{secret_name}' OK")
    except ClientError as e:
        code = e.response["Error"]["Code"]
        msg  = e.response["Error"].get("Message", "")
        logger.error(
            f"[config] _load_from_asm: ClientError code='{code}' "
            f"secret='{secret_name}' message='{msg}'\n{traceback.format_exc()}"
        )
        if code == "ResourceNotFoundException":
            raise RuntimeError(
                f"Config secret '{secret_name}' not found in ASM (region={_ASM_REGION}). "
                "Create it or set HR_BOT_CONFIG_SECRET_NAME to the correct name."
            ) from e
        elif code in ("AccessDeniedException", "AccessDenied"):
            raise RuntimeError(
                f"Access denied to config secret '{secret_name}'. "
                "Ensure the Lambda execution role has secretsmanager:GetSecretValue on this secret."
            ) from e
        else:
            raise RuntimeError(
                f"ASM ClientError for config secret '{secret_name}': code={code} msg={msg}"
            ) from e
    except Exception as e:
        logger.error(
            f"[config] _load_from_asm: unexpected error fetching secret — "
            f"type={type(e).__name__} error={e}\n{traceback.format_exc()}"
        )
        raise RuntimeError(f"Failed to fetch config secret '{secret_name}': {e}") from e

    try:
        secret_string = response.get("SecretString")
        if not secret_string:
            raise RuntimeError(
                f"Config secret '{secret_name}' has no SecretString. "
                "Binary secrets are not supported."
            )

        config = json.loads(secret_string)
    except json.JSONDecodeError as e:
        logger.error(
            f"[config] _load_from_asm: secret '{secret_name}' is not valid JSON — "
            f"error={e}\n{traceback.format_exc()}"
        )
        raise RuntimeError(
            f"Config secret '{secret_name}' is not valid JSON: {e}"
        ) from e
    except RuntimeError:
        raise
    except Exception as e:
        logger.error(
            f"[config] _load_from_asm: failed to parse secret response — "
            f"type={type(e).__name__} error={e}\n{traceback.format_exc()}"
        )
        raise RuntimeError(
            f"Failed to parse config secret '{secret_name}': {e}"
        ) from e

    try:
        _validate_config(config, secret_name)
    except Exception as e:
        logger.error(
            f"[config] _load_from_asm: config validation failed — "
            f"type={type(e).__name__} error={e}\n{traceback.format_exc()}"
        )
        raise

    return config


def _validate_config(config: dict, secret_name: str) -> None:
    """Raise RuntimeError if embedding-required keys are missing or blank."""
    try:
        embedding_required = [
            "ai_gateway_url",
            "embed_model",
            "ai_gateway_token_secret_name",
        ]

        missing = [k for k in embedding_required if not config.get(k)]
        if missing:
            raise RuntimeError(
                f"Config secret '{secret_name}' is missing required keys: {missing}. "
                "Update the secret with all required fields."
            )

        logger.info(
            f"[config] _validate_config: all required keys present in '{secret_name}'"
        )
    except RuntimeError:
        raise
    except Exception as e:
        logger.error(
            f"[config] _validate_config: unexpected error — "
            f"type={type(e).__name__} error={e}\n{traceback.format_exc()}"
        )
        raise RuntimeError(f"Config validation failed unexpectedly: {e}") from e
