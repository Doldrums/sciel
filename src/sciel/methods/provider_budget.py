"""Methods-only provider accounting without a total dollar ceiling.

The shared engine CostLedger remains unchanged. Per-request reservations and
actual charges are finite; unknown charges and request overruns still fail closed.
"""
from __future__ import annotations

from decimal import Decimal
from pathlib import Path
import time
import uuid

from ..campaign.budget import CostLedger, CampaignStopped, RequestNotDispatched, money


class UnlimitedCostLedger(CostLedger):
    """Null total ceiling, with the existing durable provider receipt transport."""

    def __init__(self, path, limit_usd=None):
        if limit_usd is not None:
            raise ValueError('UnlimitedCostLedger requires an explicit null dollar ceiling')
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.limit = None
        with self._locked() as state:
            if not state:
                state.update(schema='sciel-cost-ledger/1', limit_usd=None, calls={}, halted=None)
            elif (state.get('schema') != 'sciel-cost-ledger/1' or 'limit_usd' not in state
                    or state['limit_usd'] is not None or not isinstance(state.get('calls'), dict)):
                raise ValueError('existing provider ledger limit cannot change on resume')

    def reserve(self, bound_usd, *, run_id, model):
        bound = money(bound_usd)
        with self._locked() as state:
            # Validate and retain finite existing receipt totals. No replacement
            # ceiling, periodically reset balance, or discarded history is used.
            self.totals(state)
            if state.get('halted'):
                raise RequestNotDispatched(state['halted'])
            if any(c.get('status') == 'unknown' or (c.get('status') == 'settled' and c.get('cost_usd') is None)
                   for c in state['calls'].values()):
                raise RequestNotDispatched('provider request outcome is unknown; reservation retained')
            ident = uuid.uuid4().hex
            state['calls'][ident] = dict(run_id=run_id, model=model,
                reservation_usd=str(bound), cost_usd=None, status='reserved', created_at=time.time())
        return ident

    def snapshot(self):
        with self._locked() as state:
            known, held = self.totals(state)
            return dict(limit_usd=None, known_cost_usd=str(known), reserved_or_unknown_usd=str(held),
                        remaining_usd=None, calls=len(state['calls']), halted=state.get('halted'))

    def resume_local_pause(self, expected_reason, *, resolution):
        """Preserve the shared audit rule, omitting only its total-ceiling test."""
        if (not isinstance(expected_reason, str) or not expected_reason.startswith('local research pause:')
                or not isinstance(resolution, str) or not resolution.strip()):
            raise ValueError('need the exact local pause reason and its resolution')
        with self._locked() as state:
            if state.get('halted') != expected_reason:
                raise CampaignStopped('the local pause reason changed; halt preserved')
            if any(c.get('status') != 'settled' or c.get('cost_usd') is None
                   or money(c['cost_usd']) > money(c['reservation_usd']) for c in state['calls'].values()):
                raise CampaignStopped('unsettled, unknown or overrun charges; halt preserved')
            known, held = self.totals(state)
            if held != Decimal(0):
                raise CampaignStopped('invalid accounting balance; halt preserved')
            state.setdefault('local_pause_resolutions', []).append(dict(
                reason=expected_reason, resolution=resolution, resumed_at=time.time(),
                known_cost_usd=str(known), limit_usd=None))
            state['halted'] = None
