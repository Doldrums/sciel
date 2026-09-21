"""Wire schemas and bounds for saved initializer evidence."""
SCHEMA = 'sciel-initializer-evidence/1'
VIEW_SCHEMA = 'sciel-initializer-evidence-view/3'
ORIGIN_VIEW_SCHEMA = 'sciel-initializer-evidence-view/2'
LEGACY_VIEW_SCHEMA = 'sciel-initializer-evidence-view/1'
COMPARISON_SCHEMA = 'sciel-initializer-evidence-comparison/2'
LEGACY_COMPARISON_SCHEMA = 'sciel-initializer-evidence-comparison/1'
FIXTURE_SHA256 = 'ed291e3bbe8632b09303a06ceb4f9a4415cd311775cee6da448f35339f938d4f'
MAX_EVIDENCE_CHARS = 128 * 1024
MAX_RECEIPT_CHARS = 16 * 1024**2
SCOPE = ('Fixed finite initializer witnesses: input constraints compatible, trajectory reachability unproved. '
         'Candidate change/pair facts compare effective output to fixed pre-call best_params. '
         'Not universal equivalence, a semantic claim verdict, packing quality, independent support, '
         'lesson evidence or authorization to reject/deduplicate a candidate. Hashes bind retained '
         'data, not externally authenticate execution; full raw receipt stays in the journal.')
ORIGIN_VIEW_SCOPE = (SCOPE + ' Per-case origin return means an accepted helper vector; saved_ordinary means '
              'the helper returned None and the caller used its saved ordinary guess. Facts describe '
              'that effective output versus best_params, not necessarily helper edits. Handoff counts '
              'are attempts/accepted/None/errors, not improvements. Missing origin/counts stay null.')
VIEW_SCOPE = (ORIGIN_VIEW_SCOPE + ' Case rows align with case_ids; handoff_ref indexes handoffs.')
_ROLES = ('candidate','parent','prior')
_IDENTITY = {'checker_sha256','builder_files','builder_contract','control_source_sha256',
    'common_scaffold_sha256','proposal_runtime_sha256','handoff_sha256','initial_parameters_sha256',
    'python','python_executable_sha256','platform','numpy_version','numpy_init_sha256','numpy_core_sha256'}
_SOURCE = {'source_sha256','body_sha256','helper_sha256','complete','status','error_code','observations'}
_OBS = {'case_id','status','effective_output','effective_origin','raw_return_kind','handoff_counts','error_code'}
