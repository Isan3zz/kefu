from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


OrderCheck = Literal['track_shipment', 'check_refund_eligibility']
OrderId = Annotated[str, Field(pattern=r'^O[0-9]+$', max_length=100)]


class CreateTaskPlan(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)

    order_ids: list[OrderId] = Field(default_factory=list, max_length=4)
    recent_order_count: int | None = Field(default=None, ge=2, le=4)
    checks: list[OrderCheck] = Field(min_length=1, max_length=2)

    @model_validator(mode='after')
    def validate_scope(self):
        if bool(self.order_ids) == (self.recent_order_count is not None):
            raise ValueError('Provide order_ids or recent_order_count')
        if len(set(self.order_ids)) != len(self.order_ids) or len(set(self.checks)) != len(self.checks):
            raise ValueError('Orders and checks must be unique')
        if len(self.order_ids) == len(self.checks) == 1:
            raise ValueError('A single check for a single order does not need a plan')
        return self


@dataclass(frozen=True)
class TaskCheck:
    tool: str
    arguments: dict
    observation: dict | None

    @property
    def status(self):
        if self.observation is None:
            return 'PENDING'
        return 'COMPLETED' if self.observation['ok'] else 'FAILED'

    def progress(self):
        return {
            'tool': self.tool, **self.arguments, 'status': self.status,
            **({'evidence_id': self.observation['request_id']} if self.observation else {}),
            **({'error_code': self.observation['error']['code']} if self.status == 'FAILED' else {}),
        }


def task_checks(plan, observations):
    """Build checks from this invocation's server-validated tool observations."""
    checks = []

    def check(tool, arguments, matches):
        candidates = [row['result'] for row in observations if row['name'] == tool and matches(row['args'])]
        result = next((candidate for candidate in candidates if candidate['ok']), candidates[-1] if candidates else None)
        checks.append(TaskCheck(tool, arguments, result))
        return result

    order_ids = plan['order_ids']
    if count := plan['recent_order_count']:
        selection = check('get_orders', {'limit': count}, lambda args: args['limit'] >= count)
        if not selection or not selection['ok']:
            return checks
        order_ids = list(dict.fromkeys(row['id'] for row in selection['data']))[:count]
    for order_id in order_ids:
        for tool in plan['checks']:
            check(tool, {'order_id': order_id}, lambda args: args['order_id'] == order_id)
    return checks
