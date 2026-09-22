# Upstream v0.2: first local-inference port

This isolated branch starts at upstream `bc8108bfcce446e7ea159ef2fb485e7a6c297a18`.
The release source is `a6c0d215b61ee52dae608a6390c9f2c6d2f2554d`; the three later
commits change only README. The LoRA reference is
`3c7a9bdc5fee590772b9883b27e8ec5a67e1cc03`. These repositories have unrelated Git
roots, so this is an explicit port onto the new engine, not a merge that claims
historical checkpoint compatibility. The existing comparison uses its original
frozen sources.

## Implemented slice

The engine remains upstream `espalier-3.64`, including its split lifecycle,
candidate evaluation backend, Codex Responses provider, resolved-model fields,
subscription accounting, cooperative IO and research-tool implementation.
The local OpenAI provider carries a frozen base identity, explicit sampling seed,
non-thinking chat-template control, signed deployment and exact response-header
check. Global auxiliary roles still use the canonical base deployment. Its zero
API charge is explicitly `external_api_only`, referenced to a node ledger; it is
not an assertion of zero GPU, wall-time or rental cost.

A normal generation snapshot freezes its learner/base specification before the
native step is queued. The worker, transcript and parallel call journal retain
that specification and signed binding. A returned response retains both the new
upstream `resolved_model` and local served-model/header evidence. Replaying the
same durable call does not generate again. Changing its seed/binding changes its
request hash.

Local and managed deployment identities are checked on checkpoint restore even
when force is requested. Presence of old adaptation fields in checkpoint state,
identity or saved configuration is rejected, including an empty field. There is
no migration that clears outstanding requests, route history or paid outcomes.
The static port supports a fresh run and exact same-identity restart, not a
3.44-to-3.64 LoRA checkpoint resume.

The original LoRA types, signed verifier, token counter and visibility helper
are copied byte-for-byte. The local provider starts from the original module,
with an explicit temporary refusal of configured adapters/continuous routing and
without importing the not-yet-ported corpus policy for its static identity.
Unknown generation visibility remains explicitly unknown. This port does not
claim new training-row eligibility or C1/C3 qualification.

## Local validation

The new CPU fixture uses a real Ed25519 signer, loopback HTTP status/generation
protocol, SDK, new upstream Engine, native checkpoint save/restore, and parallel
CallLedger. Model completions and evaluator results are synthetic. It proves one
ordinary G call, base auxiliary binding, strict signature/header checks, exact
WAL replay, and refusal of unsupported config/forced checkpoint rebinding.
It does not load model weights, run a GPU, inspect private banks or measure
scientific quality.

The first combined selection passed 312 tests in 17.21 seconds: the new fixture,
upstream structured transport, cost provenance, output limits, model capability,
resolved-model, transcript, subscription accounting, error metadata, checkpoint,
Codex and parallel-call suites. A subsequent narrow saved-config/identity refusal
addition passed all 38 local-port/checkpoint tests in 0.91 seconds. Final logs and
source hashes are retained under `outputs/upstream-v020-local-review-001`.
No full test suite was run.

## Required next integration

- Move route, producer, owned execution and two-update continuation hooks into
  the new initialization/lifecycle/step/checkpoint mixins. Keep new evaluator and
  accounting hooks; never replace them with the old monolithic Engine. Adapted
  route visibility changes only after a durable committed checkpoint. Restore
  validates detached state before installing any live state.
- Restore the complete adaptation source closure and fixed-panel task. Port
  final prompt/card delivery and validation-trace joins. Prove an actual native
  full-tools own TRAIN export before claiming C1/C3 compatibility.
- Add an explicit current tool-protocol `/34` ancestry reader that verifies the
  full delegated renderer/package source identity. The old current-controller
  hash shortcut is insufficient for the new split renderer. Keep bounded legacy
  `/24`–`/26` replay separate; never execute retained source as a renderer.
- Integrate exact generation/auxiliary token caps and all generation operator
  paths. The current guard refuses circle-generation/intervention and two-stage
  idea local modes; the first fixture covers ordinary G/repair, not every search
  mode or all-tools budget behavior. Local dynamic islands require a configured
  owner for every generated island.
- Only then qualify a fresh source/release and exercise complete adoption and
  continuation. Neither existing live packets nor old results acquire a new
  source identity through this port.

The source-only lifecycle map is retained in the reference checkout at
`outputs/upstream-engine364-lora-map-001/note.md`. No merge, push, deployment,
model download or upstream-branch mutation is part of this slice.
