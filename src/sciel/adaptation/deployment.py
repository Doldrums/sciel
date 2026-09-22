"""Production verifier for an owned, explicitly pinned managed DeltaServe instance.

The signing key comes from the trusted launch channel, never from an untrusted
receipt. Live status is evidence from that owned loopback transport, not remote
hardware attestation or an independently signed freshness challenge.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import PurePosixPath
import re
import threading
from urllib.request import Request, build_opener, ProxyHandler, HTTPRedirectHandler

from .types import InferenceSpec, canonical_bytes, content_hash, require_hash

CONFIG = 'sciel-managed-deployment-client/1'
FROZEN = 'sciel-inference-deployment/1'
RECEIPT = 'sciel-managed-deployment/1'
STATUS = 'sciel-managed-deployment-status/1'
_HEADERS = ('x-sciel-server-instance', 'x-sciel-deployment')
_BASE = ('base_model', 'base_revision', 'tokenizer_revision', 'chat_template_sha256')


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _generation(value):
    _need(type(value) is str and re.fullmatch(r'[0-9a-f]{32}', value),
          'managed server/worker generation must be a 128-bit hexadecimal ID')
    return value


def configuration(llm):
    if 'managed_deployment' not in llm:
        return None
    value = llm['managed_deployment']
    fields = {'public_key', 'server_instance_id', 'bootstrap_sha256'}
    _need(llm.get('backend') == 'local_openai' and type(value) is dict and set(value) == fields,
          'managed_deployment requires local_openai and exact trusted launch pins')
    require_hash(value['public_key'], 'Ed25519 public key')
    _generation(value['server_instance_id'])
    require_hash(value['bootstrap_sha256'], 'deployment bootstrap hash')
    return dict(schema=CONFIG, **deepcopy(value))


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *_args, **_kwargs):
        raise ValueError('managed deployment endpoint cannot redirect')


class SignedDeploymentReceipts:
    """Check pinned signatures and recorded inventories without claiming freshness.

    Historical gate replay uses this same verifier as live admission. A recorded
    inventory is trusted-controller evidence of an observation, not a signed
    freshness challenge or proof that the server remains available now.
    """
    def __init__(self, pins, *, api_base, base_spec):
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        from ..engine.llm.local import validate_loopback_url
        _need(type(pins) is dict and pins.get('schema') == CONFIG,
              'managed verifier requires typed trusted launch pins')
        raw = {key: value for key, value in pins.items() if key != 'schema'}
        _need(configuration({'backend': 'local_openai', 'managed_deployment': raw}) == pins,
              'managed launch pin schema differs')
        self.pins = deepcopy(pins)
        self.api_base = validate_loopback_url(api_base)
        _need(api_base.endswith('/v1') and not api_base.endswith('//v1'),
              'managed api_base must exactly name the signed /v1 endpoint')
        self.base = InferenceSpec.from_dict(base_spec)
        _need(self.base.owner == 'base' and self.base.role == 'base' and self.base.adapter_id is None,
              'managed bootstrap requires the canonical base specification')
        self._key = Ed25519PublicKey.from_public_bytes(bytes.fromhex(pins['public_key']))

    def _projection(self, spec):
        _need(all(getattr(spec, name) == getattr(self.base, name) for name in _BASE),
              'managed request changes the pinned base identity')
        if spec.adapter_id is None:
            _need(spec.adapter_sha256 is None and spec.dataset_sha256 is None
                  and spec.serving_alias == self.base.serving_alias,
                  'base deployment projection requires all adapter fields null')
            return self.base
        _need(spec.role in {'G', 'repair'}, 'managed adapters can only serve G/repair')
        return replace(spec, role='G')

    def _receipt(self, receipt):
        from cryptography.exceptions import InvalidSignature
        _need(type(receipt) is dict and set(receipt) == {'body', 'sha256', 'public_key', 'signature'},
              'invalid managed signed receipt envelope')
        _need(receipt['public_key'] == self.pins['public_key'], 'receipt is not from the pinned signing key')
        require_hash(receipt['sha256'], 'signed deployment body hash')
        _need(receipt['sha256'] == content_hash(receipt['body']), 'signed deployment body hash differs')
        signature = receipt['signature']
        _need(type(signature) is str and re.fullmatch(r'[0-9a-f]{128}', signature), 'invalid Ed25519 signature')
        try:
            self._key.verify(bytes.fromhex(signature), canonical_bytes(receipt['body']))
        except InvalidSignature as exc:
            raise ValueError('deployment signature verification failed') from exc
        body = receipt['body']
        _need(type(body) is dict, 'invalid signed deployment body')
        common = {'schema', 'server_instance_id', 'worker_boot_id', 'bootstrap_sha256', 'api_base',
                  'inference_spec', 'immutable_alias', 'inplace_updates', 'previous_versions_retained',
                  'retention', 'kind', 'worker_ack'}
        extra = {'registry_manifest_sha256', 'files'} if body.get('kind') == 'adapter' else set()
        _need(type(body) is dict and set(body) == common | extra and body['schema'] == RECEIPT,
              'invalid signed deployment body schema')
        _need(body['server_instance_id'] == self.pins['server_instance_id']
              and body['bootstrap_sha256'] == self.pins['bootstrap_sha256']
              and body['api_base'] == self.api_base, 'deployment differs from trusted launch/endpoint')
        _generation(body['worker_boot_id'])
        _need(body['immutable_alias'] is True and body['inplace_updates'] is False
              and body['previous_versions_retained'] is True, 'deployment lacks immutable retention')
        retention = body['retention']
        _need(type(retention) is dict and set(retention) == {'kind', 'max_public_versions', 'private_lora_ids'}
              and retention['kind'] == 'bounded_pinned_cpu_gpu'
              and type(retention['max_public_versions']) is int and retention['max_public_versions'] > 0
              and retention['private_lora_ids'] in ([], [1000]), 'invalid bounded deployment retention')
        spec = InferenceSpec.from_dict(body['inference_spec'])
        _need(all(getattr(spec, key) == getattr(self.base, key) for key in _BASE),
              'deployed base identity differs')
        if body['kind'] == 'base':
            _need(spec == self.base and body['worker_ack'] is None, 'invalid canonical base deployment')
        else:
            _need(body['kind'] == 'adapter' and spec.adapter_id is not None and spec.role == 'G'
                  and spec.owner != 'base' and spec.serving_alias != self.base.serving_alias,
                  'invalid adapted deployment specification')
            _need(spec.serving_alias == 'lora-' + content_hash({'owner': spec.owner, 'adapter_id': spec.adapter_id}),
                  'deployed alias differs from owner/revision')
            files = body['files']
            _need(type(files) is dict and {'adapter_config.json', 'adapter_model.safetensors', 'training_receipt.json'} <= set(files),
                  'deployment lacks the registered bundle file commitment')
            for path, value in files.items():
                _need(type(path) is str and path and not PurePosixPath(path).is_absolute()
                      and '..' not in PurePosixPath(path).parts, 'invalid deployed bundle member')
                require_hash(value, 'deployed file hash')
            _need(content_hash(files) == spec.adapter_sha256, 'deployed bundle hash differs from exact adapter')
            require_hash(body['registry_manifest_sha256'], 'deployed registry manifest hash')
            ack = body['worker_ack']
            _need(type(ack) is dict and set(ack) == {'alias', 'path', 'lora_int_id', 'manifest_sha256', 'adapter_sha256'}
                  and ack['alias'] == spec.serving_alias and ack['adapter_sha256'] == spec.adapter_sha256
                  and ack['manifest_sha256'] == body['registry_manifest_sha256']
                  and type(ack['lora_int_id']) is int and ack['lora_int_id'] > 0 and ack['lora_int_id'] != 1000,
                  'worker ACK differs from immutable deployed revision')
            path = ack['path']
            _need(type(path) is str and PurePosixPath(path).is_absolute() and '..' not in PurePosixPath(path).parts
                  and tuple(PurePosixPath(path).parts[-2:]) == (body['server_instance_id'], spec.adapter_sha256),
                  'worker snapshot path differs from owned server generation/artifact')
        return spec

    def _matches(self, receipt, requested):
        _need(self._receipt(receipt) == self._projection(requested),
              'signed deployment does not match the exact requested inference specification')

    def binding(self, receipt, spec):
        """Validate a signed binding; this alone grants no live route authority."""
        spec = spec if isinstance(spec, InferenceSpec) else InferenceSpec.from_dict(spec)
        self._matches(receipt, spec)
        return dict(schema=FROZEN, inference_spec=spec.to_dict(), receipt=deepcopy(receipt))

    def inventory(self, status, *, expected=None, worker_boot_id=None):
        _need(type(status) is dict and set(status) == {'schema', 'server_instance_id', 'public_key', 'receipts'}
              and status['schema'] == STATUS and status['server_instance_id'] == self.pins['server_instance_id']
              and status['public_key'] == self.pins['public_key'] and type(status['receipts']) is dict,
              'live managed status differs from the trusted server')
        aliases = {}
        for token, receipt in status['receipts'].items():
            spec = self._receipt(receipt)
            _need(token == receipt['sha256'] and spec.serving_alias not in aliases,
                  'live deployment token/alias is ambiguous')
            aliases[spec.serving_alias] = deepcopy(receipt)
        base = aliases.get(self.base.serving_alias)
        _need(base is not None, 'live server lacks its signed base deployment')
        boot = base['body']['worker_boot_id']
        retention = base['body']['retention']
        _need(all(row['body']['worker_boot_id'] == boot and row['body']['retention'] == retention
                  for row in aliases.values()) and len(aliases) - 1 <= retention['max_public_versions'],
              'live receipts disagree on worker generation/capacity')
        _need(worker_boot_id is None or boot == worker_boot_id,
              'managed worker generation changed; explicit rebind required')
        for alias, receipt in (expected or {}).items():
            _need(aliases.get(alias) == receipt, 'accepted old deployment is no longer retained')
        return aliases


class ManagedDeploymentVerifier(SignedDeploymentReceipts):
    """Verify live admission; native freezing only reads a sealed view.

    The cache is a verifier result, not route authorization: LocalOpenAILLM must
    separately validate static or accepted-history ownership before using it.
    """
    def __init__(self, pins, *, api_base, base_spec, initial_specs=(), timeout=10.0):
        super().__init__(pins, api_base=api_base, base_spec=base_spec)
        self._timeout = timeout
        self._mutex = threading.Lock()
        self._receipts = {}
        self._worker_boot_id = None
        live = self._live()
        specs = [self.base, *(InferenceSpec.from_dict(value) for value in initial_specs)]
        accepted = {}
        for spec in specs:
            alias = self._projection(spec).serving_alias
            receipt = live.get(alias)
            _need(receipt is not None, 'managed bootstrap lacks an initial alias')
            self._matches(receipt, spec)
            accepted[alias] = deepcopy(receipt)
        self._receipts = accepted
        self.identity = dict(schema=CONFIG, api_base=api_base, pins=deepcopy(pins),
                             initial_receipts=deepcopy(accepted))

    def _get_status(self):
        # Disable proxies/redirects: the launch pins refer to this exact loopback
        # endpoint. No API keys, environment credentials or response secrets leak.
        opener = build_opener(ProxyHandler({}), _NoRedirect())
        request = Request(self.api_base + '/sciel/deployment/status', headers={'Accept': 'application/json'})
        with opener.open(request, timeout=self._timeout) as response:
            data = response.read(4 * 1024 * 1024 + 1)
        _need(len(data) <= 4 * 1024 * 1024, 'managed status exceeds bounded receipt size')
        def unique(pairs):
            out = {}
            for key, value in pairs:
                _need(key not in out, 'duplicate managed status field')
                out[key] = value
            return out
        value = json.loads(data, object_pairs_hook=unique)
        canonical_bytes(value)
        return value

    def _live(self):
        status = self._get_status()
        aliases = self.inventory(status, expected=self._receipts, worker_boot_id=self._worker_boot_id)
        self._worker_boot_id = aliases[self.base.serving_alias]['body']['worker_boot_id']
        self._last_status = deepcopy(status)
        return aliases

    def observe(self):
        """Return the checked live inventory for durable controller evidence."""
        self._live()
        return deepcopy(self._last_status)

    def verify(self, receipt, *, inference_spec, api_base):
        _need(api_base == self.api_base, 'deployment verifier endpoint differs')
        spec = InferenceSpec.from_dict(inference_spec)
        self._matches(receipt, spec)
        live = self._live()
        alias = self._projection(spec).serving_alias
        _need(live.get(alias) == receipt, 'exact signed deployment is absent from live worker inventory')
        with self._mutex:
            old = self._receipts.get(alias)
            _need(old is None or old == receipt, 'accepted immutable alias was rewritten')
            self._receipts = {**self._receipts, alias: deepcopy(receipt)}
        return dict(schema='sciel-verified-deployment/1', receipt_sha256=content_hash(receipt),
            server_instance_id=self.pins['server_instance_id'], deployment_id=receipt['sha256'],
            api_base=api_base, inference_spec=spec.to_dict(), immutable_alias=True,
            inplace_updates=False, previous_versions_retained=True)

    def freeze(self, spec):
        spec = spec if isinstance(spec, InferenceSpec) else InferenceSpec.from_dict(spec)
        alias = self._projection(spec).serving_alias
        receipt = self._receipts.get(alias)
        _need(receipt is not None, 'inference has no previously verified deployment')
        self._matches(receipt, spec)
        return dict(schema=FROZEN, inference_spec=spec.to_dict(), receipt=deepcopy(receipt))

    def validate_frozen(self, value, spec):
        _need(type(value) is dict and value == self.freeze(spec),
              'request deployment differs from its verified immutable binding')
        return deepcopy(value)


def request_headers(binding):
    body = binding['receipt']['body']
    return dict(zip(_HEADERS, (body['server_instance_id'], binding['receipt']['sha256'])))
