"""Pinned local OpenAI transport; adaptation is explicit and phase scoped.

An alias is a serving contract, not proof of the loaded weights. The campaign
must register immutable aliases against its verified adapter manifests. The
transport additionally rejects a server response reporting a different alias.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import ipaddress
from types import MappingProxyType
from urllib.parse import urlsplit

from ...adaptation.types import InferenceSpec
from ...campaign.budget import RequestNotDispatched
from .client import COST_PROVENANCE_SCHEMA, EnsembleLLM, validate_generation_seed

ADAPTED_ROLES = frozenset({'G', 'repair'})


def _text(value, label):
    if type(value) is not str or not value.strip():
        raise ValueError(label + ' must be a nonempty string')
    return value


def validate_loopback_url(value):
    parsed = urlsplit(_text(value, 'local api_base'))
    host = parsed.hostname
    try:
        loopback = host == 'localhost' or ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback = False
    if (parsed.scheme not in {'http', 'https'} or not loopback or parsed.username
            or parsed.password or parsed.query or parsed.fragment):
        raise ValueError('local_openai requires a loopback HTTP(S) endpoint without credentials')
    return value


class LocalOpenAILLM(EnsembleLLM):
    local_inference = True
    chat_template_kwargs = MappingProxyType({'enable_thinking': False})

    def __init__(self, *, local_inference, models, weights, api_base,
                 generation_seed=None, continuous_routing=None, deployment_verifier=None,
                 managed_deployment=None, **kwargs):
        # The upstream port currently admits static base inference only. Native
        # adoption and its source-closure verifier must move together later.
        if continuous_routing is not None or deployment_verifier is not None:
            raise ValueError('upstream v0.2 continuous adaptation port is not enabled')
        if type(local_inference) is dict and local_inference.get('adapters'):
            raise ValueError('upstream v0.2 adapter inference port is not enabled')
        if managed_deployment is not None and deployment_verifier is not None:
            raise ValueError('managed deployment cannot substitute an injected verifier')
        if continuous_routing is not None:
            from ...adaptation import corpus_policy as C
            policy = C.selected(continuous_routing) if type(continuous_routing) is dict else None
            from ..route_updates import SCHEMA
            if (type(continuous_routing) is not dict or set(continuous_routing) != ({'schema', 'registry'} | ({'corpus_policy'} if policy is not None else set()))
                    or continuous_routing['schema'] != SCHEMA
                    or (managed_deployment is None and not callable(getattr(deployment_verifier, 'verify', None)))):
                raise ValueError('continuous routing requires explicit configuration and an injected deployment verifier')
        elif deployment_verifier is not None:
            raise ValueError('deployment verifier requires continuous routing')
        self._continuous_routing_identity = deepcopy(continuous_routing)
        self._deployment_verifier = deployment_verifier
        self._route_history = None
        self._generation_seed = (None if generation_seed is None
                                 else validate_generation_seed(generation_seed))
        config = deepcopy(local_inference)
        required = {'base_model', 'base_revision', 'tokenizer_revision',
                    'chat_template_sha256', 'base_alias', 'learner_by_island',
                    'adapters', 'node_ledger_id'}
        if type(config) is not dict or set(config) != required:
            raise ValueError('local_inference requires exactly ' + ', '.join(sorted(required)))
        self.base_alias = _text(config['base_alias'], 'base_alias')
        if models != [self.base_alias]:
            raise ValueError('local_openai models must contain only the frozen base_alias')
        self.node_ledger_id = _text(config['node_ledger_id'], 'node_ledger_id')
        self._base = InferenceSpec(owner='base', role='base',
            base_model=config['base_model'], base_revision=config['base_revision'],
            tokenizer_revision=config['tokenizer_revision'],
            chat_template_sha256=config['chat_template_sha256'], serving_alias=self.base_alias)
        learners = config['learner_by_island']
        if (type(learners) is not dict or not learners
                or any(type(k) is not str or not k.isdecimal() or str(int(k)) != k
                       for k in learners)
                or any(type(v) is not str or not v.strip() or v == 'base' for v in learners.values())
                or len(set(learners.values())) != len(learners)):
            raise ValueError('learner_by_island must map canonical island IDs 1:1 to learners')
        self._learners = MappingProxyType(dict(learners))
        for owner in learners.values():
            replace(self._base, owner=owner)  # Validate owners before any dispatch.
        adapters = config['adapters']
        if type(adapters) is not dict or not set(adapters) <= set(learners.values()):
            raise ValueError('adapters must refer to configured learners')
        specs, aliases = {}, {self.base_alias}
        for owner, entry in adapters.items():
            if type(entry) is not dict or set(entry) != {
                    'serving_alias', 'adapter_id', 'adapter_sha256', 'dataset_sha256'}:
                raise ValueError('adapter requires alias, ID, weights hash and dataset hash')
            _text(entry['adapter_id'], 'adapter_id')
            spec = replace(self._base, owner=owner, role='G', **entry)
            if spec.serving_alias in aliases:
                raise ValueError('each adapter requires a distinct immutable serving_alias')
            aliases.add(spec.serving_alias)
            specs[owner] = spec
        self._adapters = MappingProxyType(specs)
        self._routing_config = config
        self._managed_deployment = None
        if managed_deployment is not None:
            from ...adaptation.deployment import ManagedDeploymentVerifier
            self._managed_deployment = ManagedDeploymentVerifier(managed_deployment,
                api_base=api_base, base_spec=self._base.to_dict(),
                initial_specs=[spec.to_dict() for spec in self._adapters.values()])
            if continuous_routing is not None:
                self._deployment_verifier = self._managed_deployment
        super().__init__(models=models, weights=weights,
                         api_base=validate_loopback_url(api_base),
                         api_key='sciel-local-no-external-key', **kwargs)

    @property
    def generation_seed(self):
        """One fixed seed per provider/config; repair and tracing consume no RNG."""
        return self._generation_seed

    def _prompt_counter(self):
        if not hasattr(self, '_pinned_prompt_counter'):
            from .token_count import PinnedChatTokenCounter
            self._pinned_prompt_counter = PinnedChatTokenCounter(self._routing_config)
        return self._pinned_prompt_counter

    @property
    def prompt_tokenizer_identity(self):
        return self._prompt_counter().identity

    def count_prompt_tokens(self, prompt):
        return self._prompt_counter().count(prompt)

    @property
    def inference_identity(self):
        return {'schema': 'sciel-local-inference/1', 'api_base': self.api_base,
                'config': deepcopy(self._routing_config),
                'temperature': self.temperature, 'max_tokens': self.max_tokens,
                'model_capabilities': deepcopy(self.model_capabilities),
                'chat_template_kwargs': dict(self.chat_template_kwargs),
                **({'generation_seed': self.generation_seed} if self.generation_seed is not None else {})}

    @property
    def continuous_routing_identity(self):
        return deepcopy(self._continuous_routing_identity)

    @property
    def managed_deployment_identity(self):
        return deepcopy(self._managed_deployment.identity) if self._managed_deployment is not None else None

    @property
    def managed_deployment_pins(self):
        return deepcopy(self._managed_deployment.pins) if self._managed_deployment is not None else None

    def bind_deployment(self, spec, value=None):
        if self._managed_deployment is None:
            if value is not None:
                raise RequestNotDispatched('unmanaged inference cannot accept deployment metadata')
            return None
        try:
            if value is None:
                return self._managed_deployment.freeze(spec)
            return self._managed_deployment.validate_frozen(value, spec)
        except (ValueError, TypeError, KeyError) as exc:
            raise RequestNotDispatched('invalid managed inference deployment') from exc

    def _dispatch_completion(self, request, transport_context=None):
        if transport_context is None:
            return super()._dispatch_completion(request)
        from ...adaptation.deployment import request_headers
        raw = self._client.chat.completions.with_raw_response.create(
            **request, extra_headers=request_headers(transport_context))
        echoed = {name: raw.headers.get(name) for name in request_headers(transport_context)}
        return raw.parse(), echoed

    def freeze_route(self, *, island=None, role='base'):
        history = self._route_history
        if self._continuous_routing_identity is None or history is None:
            raise RequestNotDispatched('continuous route history is not initialized for this native run')
        return history.freeze(island=island, role=role)

    def resolve_inference(self, *, island=None, role='base'):
        if self._continuous_routing_identity is not None:
            return InferenceSpec.from_dict(self.freeze_route(island=island, role=role)['inference_spec'])
        role = _text(role, 'inference role')
        owner = 'base'
        if island is not None:
            if type(island) is not int or str(island) not in self._learners:
                raise RequestNotDispatched('island has no configured learner owner')
            owner = self._learners[str(island)]
        if role in ADAPTED_ROLES and owner == 'base':
            raise RequestNotDispatched('generation requires an explicit learner owner')
        spec = self._adapters.get(owner, self._base) if role in ADAPTED_ROLES else self._base
        return replace(spec, owner=owner, role=role)

    def validate_inference(self, model, value, *, inference_route=None):
        if model != self.base_alias:
            raise RequestNotDispatched('local requests must use the configured logical base model')
        try:
            spec = value if isinstance(value, InferenceSpec) else InferenceSpec.from_dict(value)
        except (TypeError, ValueError) as exc:
            raise RequestNotDispatched('invalid frozen inference spec') from exc
        if self._continuous_routing_identity is not None:
            history = self._route_history
            try:
                if history is None or history.validate_frozen(inference_route) != spec:
                    raise ValueError('frozen route differs from request spec')
            except (ValueError, TypeError) as exc:
                raise RequestNotDispatched('inference lacks its exact accepted route prefix') from exc
            return spec
        if inference_route is not None:
            raise RequestNotDispatched('phase routing cannot accept continuous route metadata')
        if spec.owner == 'base':
            expected = self.resolve_inference(role=spec.role)
        else:
            island = next((int(k) for k, v in self._learners.items() if v == spec.owner), None)
            if island is None:
                raise RequestNotDispatched('unknown learner in frozen inference spec')
            expected = self.resolve_inference(island=island, role=spec.role)
        if spec != expected:
            raise RequestNotDispatched('frozen inference spec differs from phase routing')
        return spec

    def _usage_cost(self, model, usage, pt, ct):
        # This zero is ONLY the external API charge. Node rent, training and
        # inference wall time belong to the referenced campaign node ledger.
        return 0.0, {'schema': COST_PROVENANCE_SCHEMA, 'source': 'local_node_accounted',
                     'node_ledger_id': self.node_ledger_id, 'external_api_usd': 0.0,
                     'scope': 'external_api_only'}

    def _wire_options(self, options):
        # Match response-only training tokenization for every local role. This
        # flag must not depend on which learner/adapter happened to be selected.
        return {**options, 'extra_body': {'chat_template_kwargs': dict(self.chat_template_kwargs)},
                **({'seed': self.generation_seed} if self.generation_seed is not None else {})}

    def generate(self, prompt, model, temperature=None, *, inference_spec=None,
                 adaptation_visibility=None, inference_route=None, inference_deployment=None, **options):
        if '_transport_context' in options or 'extra_headers' in options:
            raise RequestNotDispatched('local transport headers are derived only from verified deployment')
        if inference_spec is None:
            raise RequestNotDispatched('local inference requires an explicitly frozen spec')
        spec = self.validate_inference(model, inference_spec, inference_route=inference_route)
        if self._managed_deployment is not None and inference_deployment is None:
            raise RequestNotDispatched('managed request omitted its pre-dispatch frozen deployment')
        binding = self.bind_deployment(spec, inference_deployment)
        if adaptation_visibility is not None:
            from ...adaptation.visibility import validate_capsule
            try:
                if spec.role not in ADAPTED_ROLES:
                    raise ValueError('visibility requires a generation role')
                adaptation_visibility = validate_capsule(adaptation_visibility,
                    prompt=prompt, inference_spec=spec.to_dict())
            except (ValueError, TypeError) as exc:
                raise RequestNotDispatched('Cannot bind adaptation visibility') from exc
        response = super().generate(prompt, spec.serving_alias, temperature,
            **({'_transport_context': binding} if binding is not None else {}), **options)
        response.inference_spec = spec.to_dict()
        response.adaptation_visibility = adaptation_visibility
        if inference_route is not None:
            response.inference_route = deepcopy(inference_route)
        if binding is not None:
            from ...adaptation.deployment import request_headers
            response.inference_deployment = deepcopy(binding)
            if response.deployment_response != request_headers(binding):
                response.response_error = {'stage': 'response', 'reason': 'deployment_identity_mismatch',
                                           'cause_type': 'OtherError'}
        if response.served_model != spec.serving_alias:
            response.response_error = {'stage': 'response', 'reason': 'inference_identity_mismatch',
                                       'cause_type': 'OtherError'}
        return response


def freeze_snapshot_inference(provider, snapshot):
    """Called before saving a job intent and again defensively by the worker."""
    if getattr(provider, 'local_inference', False) is not True:
        return
    role = 'repair' if snapshot.op == 'error_repair' else 'G'
    if getattr(provider, 'continuous_routing_identity', None) is not None:
        if snapshot.inference_spec is None:
            if snapshot.inference_route is not None:
                raise RequestNotDispatched('snapshot has a route prefix without its frozen spec')
            snapshot.inference_route = provider.freeze_route(island=snapshot.island, role=role)
            snapshot.inference_spec = deepcopy(snapshot.inference_route['inference_spec'])
        provider.validate_inference(snapshot.model, snapshot.inference_spec,
                                    inference_route=snapshot.inference_route)
        if (snapshot.inference_spec['owner'] != provider._learners.get(str(snapshot.island))
                or snapshot.inference_spec['role'] != role):
            raise RequestNotDispatched('snapshot route owner/role differs from its island/operator')
        snapshot.inference_deployment = provider.bind_deployment(snapshot.inference_spec,
            getattr(snapshot, 'inference_deployment', None))
        return
    if getattr(snapshot, 'inference_route', None) is not None:
        raise RequestNotDispatched('static snapshot unexpectedly carries a continuous route')
    expected = provider.resolve_inference(island=snapshot.island, role=role).to_dict()
    if snapshot.inference_spec is None:
        snapshot.inference_spec = expected
    elif snapshot.inference_spec != expected:
        raise RequestNotDispatched('snapshot inference differs from the configured phase')
    snapshot.inference_deployment = provider.bind_deployment(snapshot.inference_spec,
        getattr(snapshot, 'inference_deployment', None))


def generation_options(provider, snapshot, *, repair=False, prompt=None, op=None,
                       previous_response=None, previous_call_id=None):
    if getattr(provider, 'local_inference', False) is not True:
        return {}
    freeze_snapshot_inference(provider, snapshot)
    spec = InferenceSpec.from_dict(snapshot.inference_spec)
    if repair:
        spec = replace(spec, role='repair')
    options = {'inference_spec': spec.to_dict()}
    if getattr(snapshot, 'inference_deployment', None) is not None:
        binding = deepcopy(snapshot.inference_deployment)
        binding['inference_spec'] = spec.to_dict()  # Repair keeps the same deployed weights.
        options['inference_deployment'] = provider.bind_deployment(spec, binding)
    if getattr(snapshot, 'inference_route', None) is not None:
        route = deepcopy(snapshot.inference_route)
        route['inference_spec'] = spec.to_dict()  # Repair keeps the same version/prefix.
        provider.validate_inference(snapshot.model, spec, inference_route=route)
        options['inference_route'] = route
    if prompt is not None:
        from ...adaptation.visibility import build_capsule
        try:
            options['adaptation_visibility'] = build_capsule(snapshot, prompt, spec.to_dict(),
                op=op or snapshot.op, previous_response=previous_response,
                previous_call_id=previous_call_id)
        except (ValueError, TypeError) as exc:
            raise RequestNotDispatched('Cannot prepare adaptation visibility') from exc
    return options


def bind_inference_options(provider, model, options, context):
    """Resolve every non-generation role to base before a durable call hash."""
    if getattr(provider, 'local_inference', False) is not True:
        return options
    options = dict(options)
    value = options.get('inference_spec')
    if value is None:
        role = context.get('role') or 'unknown'
        if role in ADAPTED_ROLES:
            raise RequestNotDispatched('generation call omitted its frozen learner spec')
        island = context.get('island', context.get('island_id'))
        value = provider.resolve_inference(island=island, role=role)
    if getattr(provider, 'continuous_routing_identity', None) is not None:
        spec = value if isinstance(value, InferenceSpec) else InferenceSpec.from_dict(value)
        route = options.get('inference_route')
        if route is None:
            island = next((int(k) for k, v in provider._learners.items() if v == spec.owner), None)
            route = provider.freeze_route(island=island, role=spec.role)
        spec = provider.validate_inference(model, spec, inference_route=route)
        options['inference_route'] = route
    else:
        if options.get('inference_route') is not None:
            raise RequestNotDispatched('static request unexpectedly carries a continuous route')
        spec = provider.validate_inference(model, value)
    if (context.get('role') in {'H', 'C', 'R', 'idea', 'research_tool'}
            and spec.role in ADAPTED_ROLES):
        raise RequestNotDispatched('generation adapter cannot serve an auxiliary role')
    options['inference_spec'] = spec.to_dict()
    binding = provider.bind_deployment(spec, options.get('inference_deployment'))
    if binding is not None:
        options['inference_deployment'] = binding
    return options
