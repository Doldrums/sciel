"""Trusted historical tool prompt renderers, never loaded from evidence.

The historical renderer bodies and guidance constants preserve source601
(controller016ea...) semantics. The older named native004 profile changes only
its known Python/JSON hint on this frozen renderer. Other helper source bytes are
pinned at runtime; changing them requires a separately reviewed compatibility
revision. Exact reconstructed-prompt comparison remains in ToolAncestry.

The finite /24 and optional-output profiles also retain their exact field hints
on the explicitly pinned /25 or /26 renderer logic. The /25 profile preserves
its argument guidance; /26 changes only dispatch budget guards, not prompt text. No stored controller code is loaded.
"""
from copy import deepcopy
from pathlib import Path
from types import FunctionType, MethodType
import ast
import hashlib
import json
from typing import TYPE_CHECKING

# Renderer code is rebound only by bind(), after the shared-byte checks.
# Declare those explicit globals for static checking without binding them early.
if TYPE_CHECKING:
    from ..research_tools.controller import (
        PROMPT_MARKER,
        TARGET_VIEW_MARKER,
        TARGET_VIEW_SCHEMA,
        _ACTION_GUIDANCE,
        _DEPENDENCY_GUIDANCE,
        _QUEUE_GUIDANCE,
        _YIELD_GUIDANCE,
        _clip,
        _clip_json,
        _feedback_card,
        _research_results_prompt,
        _sha,
        add_canonical_diagnostics,
        canonical_bytes,
        experiment_protocol,
        shared_evidence,
        state_prompt,
        validate_targets,
        artifact_manifest,
    )

CONTROLLER_601 = '016ea142cdf2876cfc88f782c08334a8a116828eaa17b4517921185e2b0cd5ec'
CONTROLLER_004 = '2d3690af546d256e820d721cc2f1d60dd05cc6b519bd4bea7663b34c712839fb'
# /24 at b796fde/07ff813; the reviewed optional-output change touches only
# dispatch kwargs. This exact source pair preserves its current prompt renderer.
CONTROLLER_24 = 'c827497a2379479eda8f5a888306c04ea3c2fee769d5ce4760fcf5dd5ad7a2b1'
_OUTPUT_CAP_CONTROLLER = 'fd1dd2ccfcc0905a8300a536731d84ea92b39891fdaddd7af16b2233d73fd643'
_OUTPUT_CAP_CONFIG = '85c5c0b5f1eea766aff696a254c1b1e9ec3ce59f567da5718295d3501d443761'
CONTROLLER_OUTPUT_CAP = _OUTPUT_CAP_CONTROLLER
_ARGUMENT_GUIDANCE_CONTROLLER = 'b7486dfce411f6c73f1a274ce30817eea001937a0883cea7f273600a591d875a'
CONTROLLER_ARGUMENT_GUIDANCE = _ARGUMENT_GUIDANCE_CONTROLLER
_TOKEN_BUDGET_CONTROLLER = '92b73df7b5cd1b5802011736099da682dcafa0f74a55010f253ed50c51c41f84'
_TOKEN_BUDGET_CONFIG = 'ed801e57f3825addbebed9286ac13320e568101eb2af443659a8706c21dffda9'
PROFILES = frozenset((CONTROLLER_601, CONTROLLER_004, CONTROLLER_24, CONTROLLER_OUTPUT_CAP, CONTROLLER_ARGUMENT_GUIDANCE))

_V24_PYTHON_CODE_GUIDANCE = (
    'JSON code example: source property only, not task quality; choose a useful check.\n'
    + json.dumps({'code': "def run(payload):\n"
        "    t = payload['research_targets']['selected']\n"
        "    o = {} if t is None else dict(source_sha256=t['source_sha256'],source_lines=len(t['source'].splitlines()))\n"
        "    return dict(schema='sciel-tool-result/1',status='tool_error' if t is None else 'success',observations=o,diagnostics={},error='source unavailable' if t is None else None)\n"}, separators=(',', ':'))
    + '\nPreserve newlines/indentation; Python None, not null.\n')


# source601 and reviewed /24 (adds stdlib ast for the future subject view).
_IMPORTS_SHA256 = frozenset(('16ab9188b306a0dfe32dee00ee263be2663222540ab5ce1c0ae3d81190b96605',
                             '39243c5dbd8462320a38d9d5dd7fb1d415fa967ffed074135f05698903090376'))
_CLASS_HEADER_SHA256 = 'f872731c056b5534d1156b302b2c553fb4ed612861f065ed592ad11994b0fb2d'

_SHARED_NODES = {'ARTIFACT_SCHEMA': 'bfd0f24112e142551cffe0d0552372775b313f0609fef4fce2535cd720e1f50e',
 'PROMPT_MARKER': '077f8c56d7144e0ab142f75a1880cd18c8dedc6642adeaf4537c68fe545e34e8',
 'RESULTS_VIEW_MARKER': '78f2a41ece5ef12cab22b00dca9f7662ac6361ff5adda102ac5510b2211d4e03',
 'TARGET_VIEW_MARKER': 'ff3140e9f11647e4224df7a27c51239d2257517bf0dd9070cabd66c5e58477ba',
 'TARGET_VIEW_SCHEMA': '41ed744b5faddc3bdf71c25d95973174349832976a78f851d6967c142ca74289',
 '_ACTION_GUIDANCE': 'a5eb576cbf8c561bdd6527aaf177ab37162f3469ee0d652f83a308964e3770d7',
 '_DEPENDENCY_GUIDANCE': '48cee48a61a58f331be663419a0680afe493a66eb460c8e79413f8f6e8dcdaf8',
 '_QUEUE_GUIDANCE': '4f3a89306bd2e98973cbec59052c13ea18e823318abaa64f1f6037b4515df13d',
 '_YIELD_GUIDANCE': 'ada4328fcfae186a1288c1264c9e555bb69f5b2c47a1de1f8c3ed8e66b81e2c8',
 '_bounded_cards': 'e6069d2bc1eef371c2a30e62e9138fbee624362759e962d0526d1c5ce460862b',
 '_clip': '6af71624a37aef63dd835992de15016aac71236537d330ceb2697426449b3944',
 '_clip_json': '2aeb83eff2d6564cd6b617c3767a0782eff9b7c0be8c29484643e741bfef21ec',
 '_event': '61031bacda0a66c1637c8dc3388f46305d569b705d5fcaf0295449c783331cf1',
 '_failure_context': '54e43f5cc71f27fdd9abde3ea78930b13e49d035f0806d4b0086baa2d86685a5',
 '_failure_feedback': '617c2ed24014d7a47fa19b639ce3b4b16ffccc2a1a72135d17debf81998219a5',
 '_feedback_card': 'b2d69090b0bcd001d10a9b952b0cd73004efcd4449764a8cbf4f1664d553b47f',
 '_finite_json': '33061f9b95c7a781c055b5339d83c99b5424de77911e35a087960200e10aacbb',
 '_followups_prompt': 'ff1b88c9978f00e5a833bf005234b36cb533b903002949a94486ce4dd864f47d',
 '_head_tail': '602aff7f2f5f21d2f169411d1682fa1960da93c4a391ab6f9637a7538d09e231',
 '_known_jobs_prompt': 'cc58a32a7991f74f6e5142516cb575bc7070f0d925371a7ae0057b838c57817e',
 '_read_artifact': 'b80e8a24a80f14966c4820b2c6a6ab68528e65b124544e08fdd23ab461727048',
 '_replay_feedback_prompt': '751f16921f04ff2d2d8dad02032b5b2ee9cf5d8bd295e1b5665f37976bd96815',
 '_research_results_prompt': '447b23d2311abfef1b337fd44b8bd1a7b9e3c2942e75194a17b794632f7d9277',
 '_sha': '7c440b89dd13b70263169cd385fb728be46150f888c45214d2b561e3662a317c',
 'context': 'bcf984e95fb227f94b76de12c34d8de256c76df2c58707e543b38ddd85f9c7c5',
 'format_context': '5aa8d164833d39305eeff6fe1d89ff7e0ce811aa9836e124a4ebdd550d5c43bb'}

_DEPENDENCY_FILES = {'engine/evaluation_timing.py': 'a1727619e9587b48d3f2e21ede68c635fb4a0c2e2cd378bf4b8f85b9f680d9af',
 'engine/research_artifacts.py': 'b42ae891db6789a87b5f47a2c7079f405153d66ad200b49c03b2f703a25cbca3',
 'engine/research_cells_context.py': '42ba4ba67c221d311ddbc340e9903e9a279d0cedddff9d3df48a9c6133f5c6fb',
 'engine/research_targets.py': 'd52308129dce1c3443da28d16c26635ed15558f3a46e290ede949b915c96addd',
 'engine/types.py': '27ad6e447022043274312ee94436db0c42755a1315a1d994b98b75cb56d71c4c',
 'methods/models.py': '6824605cd044e431b3477a53010e2be40644badbbdce4b095edf94fe202d8433',
 'research_tools/__init__.py': '345e01553a9a36104e74dd990452d0aab2b5b97974fc9040997b549f57f19043',
 'research_tools/artifact_manifest.py': '337c4c4d0fcffe15d1d4f92c8fb194ef09593b0fe967b19ed43358ff2e4246ba',
 'research_tools/canonical_diagnostics.py': 'bfe9e4d8c787bacf002bda071bf6c34adb731d8eecafeff71019cad3130a9a8f',
 'research_tools/canonical_evidence.py': 'f9074af6bb5195e5248014eab4d0d041ef144c53ff7ec761de523deccb5c76db',
 'research_tools/cells.py': '17a46001d9221fbff028f71d53f0033578beeb279e05cd9268af657959a9e8f5',
 'research_tools/config.py': '731f8a1a964816f19427afafd40c6b37fd0259f86d4ee74c77969d2fccd2d119',
 'research_tools/continuations.py': '7a846f6cf3d5a5cc990e94b16928355da5d2858eecb5d554e3b9f1cb93c896b7',
 'research_tools/controller_jobs.py': '5aa756d8dd7e1d8b657849d1f4e26dfd18ee3e2c718ec50dea74e471180d317f',
 'research_tools/decision_projection.py': '76bf4e6b786af6bc9ee97ee403b13639ab5bd6793806d5e7665dc139e24453f6',
 'research_tools/experiment_protocol.py': 'cb3b48570a5263fa631e0b4ce17f12800cf537e8a47de7955282f6dd925c6038',
 'research_tools/followups.py': '588a7878b44f0581ffe9a2ec886efc97178dacc5916a568408fd33dde7632490',
 'research_tools/job_worker.py': '2fe9cc20401a4f59e7d6d2af3d894dd813c3c52b6854e93824099e50fc62cc6b',
 'research_tools/jobs.py': 'aa5d94ced3a87f63259ba4759d6e3874759cb1668bf5eaba1016f2b6448843c5',
 'research_tools/library.py': '6f045622c58348d48a4f3ea50701b158417ab230a3fe51fc7faac11ffdea16ea',
 'research_tools/lifecycle.py': '50349c27da1af7141e3fc62db040cc129cb29dc809414730d40b11069abab3cf',
 'research_tools/replay_feedback.py': '03ec8e863996e60a744c052b400f7c9af48ea7a3ccd829f50cfd503093725925',
 'research_tools/research_steps.py': '2665466561f5000a38e2010ce221b9cfe1a276eab4429c42b4dfb48dcb7f3340',
 'research_tools/result_contract.py': 'f38fba4a2de7a6271bf5add6fdb212fc724566c2011ec1d6b6795f82c0eb0fcf',
 'research_tools/result_projection.py': 'b9a48b02bc258dadffe8c8fa3172d10b2717f9d29fa147268357b8d1494891c5',
 'research_tools/shared_evidence.py': '41b1488250c7fa6a4d357883e95b4cfbc6f2565468c03b928c1daaf00dbd289a',
 'research_tools/stage_data.py': 'd2293d0101124a2809d73340267e51296660175478d38f0497cd284e3a340bf5',
 'research_tools/transfer_jobs.py': '3234e49fb107b26dc9435f136adc874ddf10a24f2b88317b46c6ea6ac02cd4b0'}

_SHARED_GLOBALS = ('PROMPT_MARKER',
 'TARGET_VIEW_MARKER',
 'TARGET_VIEW_SCHEMA',
 '_ACTION_GUIDANCE',
 '_DEPENDENCY_GUIDANCE',
 '_QUEUE_GUIDANCE',
 '_YIELD_GUIDANCE',
 '_clip',
 '_clip_json',
 '_feedback_card',
 '_research_results_prompt',
 '_sha',
 'add_canonical_diagnostics',
 'canonical_bytes',
 'experiment_protocol',
 'shared_evidence',
 'state_prompt',
 'validate_targets',
 'artifact_manifest')

_PYTHON_CODE_GUIDANCE = 'Five-field return; JSON code syntax (replace observations with your check):\n' + json.dumps({'code': "def run(payload):\n    return dict(schema='sciel-tool-result/1',status='success',observations={},diagnostics={},error=None)\n"}, separators=(',', ':')) + '\nPreserve newlines/indentation; Python None, not null.\n'

_RESEARCH_PERMISSION = 'CPU search, comparisons and candidate transforms are allowed with existing source/results.\nReuse known values; compute for new evidence. Prior model-only closures are interpretations, not tool policy.\n'
